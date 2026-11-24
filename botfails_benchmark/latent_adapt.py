"""Small, split-aware adaptation experiment; reuses the existing VLM and classifiers.

Test episode index modulo five: 0 trains the supervised head, 1 selects the
configuration, 2/3/4 are evaluation only. Nominal train episode 0 is context;
1..4 fit heads and 5/6 are pilot-only normal recordings. No VLM training.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .data import TASKS, episodes, load_image


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def causal_context_frames(frames, cutoff, limit=8):
    """A past-only context independent of the previous classifier/query history."""
    available = [f for f in frames if 1. <= f[1] <= cutoff]
    if not available:
        raise ValueError('No camera frames available at this query time.')
    indices = np.linspace(0, len(available)-1, min(limit, len(available)), dtype=int)
    return [available[i] for i in indices]


def check_joint_head_context(metadata, task, reference, adapter_identity, model_id, context_sampling, representation):
    signature = metadata['comparison_signature']
    if (metadata.get('representation_id') != representation or metadata.get('vector_dimension') != 4096
            or metadata.get('input_mode') != 'raw' or signature.get('model_id') != model_id
            or signature.get('lora_adapter_sha256') != adapter_identity):
        raise ValueError('Joint head uses a different model, adapter or representation.')
    if (signature.get('context_sampling') != context_sampling or signature.get('num_frames') != 8
            or signature.get('reference_contexts', {}).get(task) != reference):
        raise ValueError('Joint head uses a different task/reference context or sampling protocol.')


def labels(episode):
    if episode.split == 'normal_train':
        return None
    return np.loadtxt(episode.dataset.parents[1] / 'failure_labels' /
                      (episode.task + '_anomaly') / (episode.name + '_labels.csv'), ndmin=1).astype(int)


def outcome(record, truth, threshold):
    stop = next((x['timestamp_sec'] for x in record['observations'] if x['score'] >= threshold), None)
    failure = truth is not None and bool(truth.any())
    onset = float(np.argmax(truth) / 30) if failure else None
    return dict(failure=failure, onset=onset, stop=stop,
                correct=(stop is None if not failure else stop is not None and stop >= onset))


def metrics(rows):
    failures = [x for x in rows if x['failure']]
    normal = [x for x in rows if not x['failure']]
    recall = np.mean([x['correct'] for x in failures]) if failures else 0.
    specificity = np.mean([x['correct'] for x in normal]) if normal else 0.
    return dict(recordings=len(rows), failures=len(failures), normals=len(normal),
                detected=sum(x['correct'] for x in failures),
                failures_alarmed_anywhere=sum(x['stop'] is not None for x in failures),
                early=sum(x['stop'] is not None and not x['correct'] for x in failures),
                missed=sum(x['stop'] is None for x in failures),
                false_stops=sum(not x['correct'] for x in normal),
                recall=float(recall), specificity=float(specificity),
                balanced_recording_accuracy=float((recall + specificity) / 2),
                recording_classification_accuracy=float(np.mean([x['stop'] is not None if x['failure'] else x['stop'] is None for x in rows])),
                recording_accuracy=float(np.mean([x['correct'] for x in rows])))


def score(head, vector):
    values = np.asarray(vector, np.float64)
    values = values / np.linalg.norm(values)
    if 'weights' in head:
        logit = float((values - head['feature_mean']) @ head['weights'] + head['intercept'])
        return float(1 / (1 + np.exp(-np.clip(logit, -700, 700))))
    distances = np.linalg.norm(head['memory'] - values, axis=1)
    # One nearest prefix per bag; long nominal recordings cannot dominate kNN.
    per_bag = [min(distances[head['bag_ids'] == bag]) for bag in np.unique(head['bag_ids'])]
    distance = float(np.mean(sorted(per_bag)[:3]))
    return min(1., .75 * distance / float(head['distance_threshold']))


def fit(output, tasks, dataset, frames, variants=('lora', 'base'), feature_directory='features', c_value=100.):
    from rynnbrain_vlm.cop_classifier import fit_logistic_classifier
    checks = output / 'fit_checks.json'
    times = json.loads(checks.read_text()) if checks.exists() else {}
    for task in tasks:
        for variant in variants:
            folder = output / feature_directory / variant / task
            records = [json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))]
            records = [r for r in records if r['role'] == 'train']
            vectors, targets, normal_vectors, bags = [], [], [], []
            for bag_index, record in enumerate(records):
                ep = next(e for e in episodes(dataset, frames, record['split'], task) if e.name == record['episode'])
                truth = labels(ep)
                v = np.load(folder / record['vector_file'])['vectors']
                vectors.extend(v)
                targets.extend([0 if truth is None else int(truth[x['frame_index']]) for x in record['observations']])
                if record['split'] == 'normal_train':
                    normal_vectors.extend(v)
                    bags.extend([bag_index] * len(v))
            started = time.perf_counter()
            weights, intercept, mean, diagnostics = fit_logistic_classifier(vectors, targets, c_value=c_value)
            head = output / 'heads' / variant / task
            head.mkdir(parents=True, exist_ok=True)
            np.savez(head / 'logistic.npz', weights=weights, intercept=intercept, feature_mean=mean)
            memory = np.asarray(normal_vectors, np.float64)
            memory /= np.linalg.norm(memory, axis=1)[:, None]
            bags = np.asarray(bags)
            peaks = []
            for bag in np.unique(bags):
                other = memory[bags != bag]
                other_bags = bags[bags != bag]
                distances = []
                for vector in memory[bags == bag]:
                    d = np.linalg.norm(other - vector, axis=1)
                    distances.append(np.mean([min(d[other_bags == b]) for b in np.unique(other_bags)]))
                peaks.append(max(distances))
            cutoff = float(np.nextafter(max(peaks), np.inf))
            np.savez(head / 'knn.npz', memory=memory, bag_ids=bags, distance_threshold=cutoff)
            times[task + '/' + variant] = dict(seconds=time.perf_counter() - started,
                prefixes=len(targets), failure_prefixes=sum(targets), logistic=diagnostics,
                knn_nominal_leave_one_recording_out_peaks=peaks,
                training_recordings=[r['split'] + '/' + r['episode'] for r in records])
    dump(output / 'fit_checks.json', times)


def select(output, tasks, dataset, frames):
    candidates = []
    for variant in ('lora', 'base'):
        for method, thresholds in (('logistic', (.5, .6, .75)), ('knn', (.75,))):
            for threshold in thresholds:
                rows = []
                for task in tasks:
                    with np.load(output / 'heads' / variant / task / (method + '.npz')) as a:
                        head = dict(a)
                    folder = output / 'features' / variant / task
                    for path in sorted(folder.glob('*_validation_*.json')):
                        record = json.loads(path.read_text())
                        ep = next(e for e in episodes(dataset, frames, record['split'], task) if e.name == record['episode'])
                        vectors = np.load(folder / record['vector_file'])['vectors']
                        for obs, vector in zip(record['observations'], vectors):
                            obs['score'] = score(head, vector)
                        rows.append(outcome(record, labels(ep), threshold))
                candidates.append(dict(variant=variant, method=method, threshold=threshold, **metrics(rows)))
    candidates.sort(key=lambda x: (x['balanced_recording_accuracy'], x['specificity'],
                    x['variant'] == 'lora', x['method'] == 'logistic', x['threshold']), reverse=True)
    dump(output / 'development_selection.json', candidates)
    print('DEVELOPMENT CANDIDATES', json.dumps(candidates), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('features', 'onpolicy', 'replay', 'fit', 'pilot', 'evaluate'), required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--variants', nargs='+', choices=('lora', 'base'), default=['lora', 'base'])
    p.add_argument('--training-only', action='store_true')
    p.add_argument('--context-sampling', choices=('sent_history', 'causal_uniform'), default='sent_history')
    a = p.parse_args()
    if a.stage == 'fit':
        fit(a.output, a.tasks, a.dataset, a.frames)
        select(a.output, a.tasks, a.dataset, a.frames)
        return
    from pipeline_settings import load_pipeline_config
    from rynnbrain_vlm.model import RynnBrainModel, COP_REPRESENTATION_ID
    from rynnbrain_vlm.replay_policy import observation_interval
    config = load_pipeline_config(a.config)['rynnbrain']
    choice = json.loads((a.output / 'choice.json').read_text()) if a.stage != 'features' else None
    complete = Path(config['cop_classifier']['model_paths']['raw'])
    signature = json.loads(complete.with_suffix('.json').read_text())['comparison_signature']
    model = RynnBrainModel({**config['model'], 'device_map': {'': 'cuda:0'}})
    expected_adapter = choice['adapter_sha256'] if choice and choice.get('jointly_trained') else signature['lora_adapter_sha256']
    assert model.adapter_identity == expected_adapter
    task_config = json.loads((Path(__file__).parent / 'tasks_final.json').read_text())
    if choice:
        assert choice.get('context_sampling', 'sent_history') == a.context_sampling
    if a.stage in ('pilot', 'evaluate'):
        if a.stage == 'evaluate':
            gate = json.loads((a.output / 'pilot_gate.json').read_text())
            assert gate['passed'], 'Pilot gate failed; full evaluation prohibited.'
            assert gate['choice_sha256'] == hashlib.sha256((a.output / 'choice.json').read_bytes()).hexdigest()
        hashes = {t: hashlib.sha256((a.output / 'heads' / choice['variant'] / t /
                  (choice['method'] + '.npz')).read_bytes()).hexdigest() for t in a.tasks}
        frozen = dict(choice=choice, head_sha256=hashes,
            config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),context_sampling=a.context_sampling,
            head_metadata_sha256={t:hashlib.sha256((a.output/'heads'/choice['variant']/t/(choice['method']+'.json')).read_bytes()).hexdigest()
                                  for t in a.tasks if (a.output/'heads'/choice['variant']/t/(choice['method']+'.json')).exists()},
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if a.stage == 'evaluate':
            assert json.loads((a.output/'pilot_manifest.json').read_text()) == frozen, 'Evaluation differs from the validated pilot.'
        manifest = a.output / ('pilot_manifest.json' if a.stage=='pilot' else 'evaluation_manifest.json')
        if manifest.exists():
            assert json.loads(manifest.read_text()) == frozen, 'Evaluation configuration changed.'
        else:
            dump(manifest, frozen)
    for task in a.tasks:
        train = episodes(a.dataset, a.frames, 'normal_train', task)
        test = episodes(a.dataset, a.frames, 'test', task)
        camera = task_config[task].get('camera', 'logitech_1')
        reference = train[0]
        turn1 = signature['turn1_prompt'].replace(signature['task_description'], task_config[task]['instruction'])
        turn2 = signature['turn2_prompt'].replace(signature['task_description'], task_config[task]['instruction'])
        ref_frames = [x for x in reference.frame_table(camera) if x[1] >= 1.]
        ids = np.linspace(0, len(ref_frames) - 1, 8, dtype=int)
        topic = signature['test_camera_topics'][0]
        def image(frame, index):
            return (f'Frame {index+1}, camera {topic}, time {frame[1]:.3f} s from this bag start:', load_image(frame[2]))
        ref_turn = dict(role='user', images=[image(ref_frames[i], k) for k, i in enumerate(ids)], text=turn1)
        ref_path = a.output / 'references' / (task + '.json')
        if choice and choice.get('jointly_trained') and not ref_path.exists():
            raise ValueError('The jointly trained head requires its saved reference context.')
        if ref_path.exists():
            response = json.loads(ref_path.read_text())['response']
        else:
            response = model.generate_nominal(ref_turn, config['generation'])
            if not response.strip():
                raise ValueError('Empty nominal response')
            dump(ref_path, dict(reference=reference.key, camera=camera, response=response,
                images=[str(ref_frames[i][2]) for i in ids], turn1=turn1, turn2=turn2))
        if a.stage in ('features', 'onpolicy'):
            jobs = [(e, 'train') for e in train[1:5]] + [(e, 'train') for e in test if e.index % 5 == 0]
            if a.stage == 'features' and not a.training_only:
                jobs += [(e, 'validation') for e in train[5:7]] + [(e, 'validation') for e in test if e.index % 5 == 1]
        elif a.stage in ('pilot', 'replay'):
            jobs = [(e, 'pilot') for e in train[7:9]] + [(e, 'pilot') for e in test if e.index % 5 == 1]
        else:
            jobs = [(e, 'evaluation') for e in test if e.index % 5 in (2, 3, 4)]
        variants = a.variants if a.stage in ('features', 'onpolicy', 'replay') else (choice['variant'],)
        if a.stage == 'onpolicy':
            assert variants[0] == choice['variant'], 'Extract the scheduling-policy variant first.'
        for episode, split_role in jobs:
            folders = {v: a.output / ('features' if a.stage == 'features' else a.stage) / v / task for v in variants}
            name = f'{episode.split}_{split_role}_{episode.name}'
            if all((folder / (name + '.json')).exists() for folder in folders.values()):
                continue
            observations = {v: [] for v in variants}
            vectors = {v: [] for v in variants}
            head = None
            if choice and a.stage != 'replay':
                head_path = a.output / 'heads' / choice['variant'] / task / (choice['method'] + '.npz')
                if choice.get('jointly_trained'):
                    metadata = json.loads(head_path.with_suffix('.json').read_text())
                    saved_reference = json.loads(ref_path.read_text())
                    check_joint_head_context(metadata,task,saved_reference,model.adapter_identity,model.model_id,
                                             a.context_sampling,COP_REPRESENTATION_ID)
                    if (saved_reference['turn1'] != turn1 or saved_reference['turn2'] != turn2
                            or saved_reference['images'] != [str(ref_frames[i][2]) for i in ids]
                            or metadata['comparison_signature']['model_config']['dtype'] != config['model']['dtype']
                            or metadata['comparison_signature']['model_config']['max_image_size'] != model.max_image_size
                            or metadata['comparison_signature'].get('enable_thinking',False) != bool(config['generation'].get('enable_thinking',False))):
                        raise ValueError('Runtime prompts, reference frames or precision differ from joint training.')
                with np.load(head_path) as h:
                    head = dict(h)
            replay = None
            if a.stage == 'replay':
                original = json.loads((a.output / 'pilot' / choice['variant'] / task / (name + '.json')).read_text())
                replay = {x['frame_index']: x['next_interval_sec'] for x in original['observations']}
            history, due = [], 1.
            camera_frames = episode.frame_table(camera)
            for frame in camera_frames:
                if (replay is not None and frame[0] not in replay) or (replay is None and frame[1] + 1e-9 < due):
                    continue
                history.append(image(frame, len(history)))
                context_frames = causal_context_frames(camera_frames, frame[1]) if a.context_sampling == 'causal_uniform' else None
                current_images = [image(f,k) for k,f in enumerate(context_frames)] if context_frames is not None else history
                for variant in variants:
                    started = time.perf_counter()
                    context = model.model.disable_adapter() if variant == 'base' else nullcontext()
                    with context:
                        vector = model.extract_multiturn_cop_vector([ref_turn, dict(role='user', images=current_images, text=turn2)],
                            config['generation'], reference_response=response).numpy()
                    assert vector.shape == (4096,) and np.isfinite(vector).all()
                    value = score(head, vector) if head is not None and variant == choice['variant'] else None
                    if variant == variants[0]:
                        interval = replay[frame[0]] if replay is not None else observation_interval({'interval_mode': 'adaptive'}, value, 10.) if choice else (10., 7., 5., 3.)[(episode.index + len(history) - 1) % 4]
                    observations[variant].append(dict(timestamp_sec=frame[1], frame_index=frame[0], score=value,
                        next_interval_sec=interval, elapsed_sec=time.perf_counter() - started,
                        context_frame_indices=[f[0] for f in context_frames] if context_frames is not None else None))
                    vectors[variant].append(vector)
                due = frame[1] + interval
                if choice and a.stage not in ('onpolicy', 'replay') and value >= choice['threshold']:
                    break  # A live execution ends at its first alarm.
            for variant, folder in folders.items():
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / (name + '.npz'), vectors=np.stack(vectors[variant]))
                record = dict(task=task, episode=episode.name, split=episode.split, role=split_role,
                    reference=reference.key, camera=camera, observations=observations[variant],
                    vector_file=name + '.npz', completed=True, context_sampling=a.context_sampling)
                if choice and a.stage not in ('onpolicy', 'replay'):
                    record.update(outcome(record, labels(episode), choice['threshold']))
                dump(folder / (name + '.json'), record)
            print(a.stage, task, name, 'observations', len(history), flush=True)
    if a.stage == 'pilot':
        rows = [json.loads(p.read_text()) for p in (a.output / 'pilot' / choice['variant']).glob('*/*.json')]
        m = metrics(rows)
        # Quality gate: real failures detected and unseen nominal recordings mostly continue.
        passed = m['recall'] >= .5 and m['specificity'] >= .75 and m['balanced_recording_accuracy'] >= .625
        dump(a.output / 'pilot_gate.json', dict(passed=bool(passed),
             choice_sha256=hashlib.sha256((a.output / 'choice.json').read_bytes()).hexdigest(), **m))
        print('PILOT GATE', passed, m, flush=True)


if __name__ == '__main__':
    main()
