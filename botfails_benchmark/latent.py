"""Frozen robot-trained latent detector transfer to BotFails, using adaptive sampling.

Task/reference/camera context changes are explicit transfer, not classifier retraining.
Existing GUI compatibility guards are not modified. Model/adapter/classifier identities
are checked before scoring the new domain with the frozen logistic parameters.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import TASKS, episodes, load_image


def summarize(output, dataset):
    from .report import rates
    records = [json.loads(p.read_text()) for p in sorted((output / 'episodes').glob('*/*.json'))]
    rows, task_metrics = [], []
    for record in records:
        if record['split'] != 'test':
            continue
        full = np.loadtxt(dataset.parents[1] / 'failure_labels' / (record['task']+'_anomaly') /
                          (record['episode']+'_labels.csv'), ndmin=1).astype(int)
        if (full < 0).any():
            raise ValueError('Resolve unknown failure labels before this evaluation.')
        failure = bool(full.any())
        onset = float(np.argmax(full)/30) if failure else None
        stop = record['first_alarm_sec']
        correct = stop is None if not failure else stop is not None and stop >= onset
        rows.append(dict(task=record['task'], episode=record['episode'], failure=failure, onset_sec=onset,
                         first_alarm_sec=stop, correct_recording_outcome=correct,
                         alarm_anywhere=stop is not None, early_stop=failure and stop is not None and stop < onset,
                         detected_after_onset=failure and stop is not None and stop >= onset,
                         observations=len(record['observations']), peak_score=max(x['score'] for x in record['observations'])))
    table = pd.DataFrame(rows)
    if table.empty:
        return
    table.to_csv(output / 'recordings.csv', index=False)
    summaries = []
    for subset in ('all', 'heldout_round2'):
        selected = table if subset == 'all' else table[table.episode.map(lambda e: int(e.split('_')[1]) % 5 not in (0, 1))]
        if selected.empty:
            continue
        failures, normal = selected[selected.failure], selected[~selected.failure]
        task_metrics = []
        for task, part in selected.groupby('task'):
            truth, prediction = [], []
            for row in part.itertuples():
                full = np.loadtxt(dataset.parents[1] / 'failure_labels' / (task+'_anomaly') /
                                  (row.episode+'_labels.csv'), ndmin=1).astype(int)
                indices = np.arange(0, len(full), 15)
                truth.extend(full[indices])
                prediction.extend(indices/30 >= row.first_alarm_sec if pd.notna(row.first_alarm_sec) else np.zeros(len(indices), bool))
            task_metrics.append(dict(task=task, **rates(truth, prediction)))
        frame_table = pd.DataFrame(task_metrics)
        frame_table.to_csv(output / f'frame_metrics_{subset}.csv', index=False)
        frame_macro = frame_table[['TPR', 'TNR', 'FPR', 'Acc', 'WAcc']].mean().to_dict()
        summaries.append(dict(subset=subset, recordings=len(selected), failures=len(failures), normals=len(normal),
            recording_accuracy=100*selected.correct_recording_outcome.mean(),
            detected_after_onset=int(failures.detected_after_onset.sum()),
            early_stops=int(failures.early_stop.sum()), missed=int((~failures.alarm_anywhere).sum()),
            false_stops=int(normal.alarm_anywhere.sum()), episode_FPR=100*normal.alarm_anywhere.mean() if len(normal) else None,
            frame_metrics_macro=frame_macro))
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    from pipeline_settings import load_pipeline_config
    from rynnbrain_vlm.model import RynnBrainModel, COP_REPRESENTATION_ID
    from rynnbrain_vlm.cop_classifier import load_classifier
    from rynnbrain_vlm.cop_onset import load_onset_classifier
    from rynnbrain_vlm.replay_policy import replay_settings, observation_interval
    from rynnbrain_vlm.feature_verification import vector_difference
    import torch

    config = load_pipeline_config(args.config)
    vlm = config['rynnbrain']
    policy = replay_settings(config)
    if policy['interval_mode'] != 'adaptive' or policy['max_observations'] is not None:
        raise ValueError('This benchmark requires the configured unlimited adaptive policy.')
    path = Path(vlm['cop_classifier']['model_paths']['raw'])
    complete = load_classifier(path)
    classifier, onset_path = load_onset_classifier(path, complete, bag_name='BotFails_transfer')
    threshold = float(vlm['cop_classifier']['training']['threshold'])
    model = RynnBrainModel({**vlm['model'], 'device_map': {'': 'cuda:0'}})
    signature = complete.comparison_signature
    assert model.adapter_identity == signature['lora_adapter_sha256']
    assert classifier.representation_id == COP_REPRESENTATION_ID
    assert model.model_id == signature['model_id']
    assert model.temporal_profile == signature['temporal_profile']
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = dict(config=config, classifier=str(onset_path), complete_classifier=str(path),
        adapter_sha256=model.adapter_identity, classifier_sha256=hashlib.sha256(onset_path.read_bytes()).hexdigest(),
        representation_id=COP_REPRESENTATION_ID, threshold=threshold, adaptive_policy=policy,
        feature_backend='uncached_full_forward_training_readout',
        transfer_changes=['task instruction', 'nominal demonstration and response', 'external camera images'],
        camera_label_protocol='Training camera role labels retained; physical BotFails camera recorded per task.',
        training_on_botfails=False, pilot=args.pilot)
    frozen['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run_path = args.output / 'run.json'
    if run_path.exists() and json.loads(run_path.read_text()) != frozen:
        raise ValueError('Output already contains a different configuration; choose another output.')
    run_path.write_text(json.dumps(frozen, indent=2))
    task_configs = json.loads((Path(__file__).parent / 'tasks_final.json').read_text())
    selected_tasks = ('domotic_groceriesSorting', 'domotic_pouringCoffee', 'industrial_robothon_buttons', 'industrial_soldering') if args.pilot else TASKS
    topic = signature['test_camera_topics'][0]
    old_task = signature['task_description']
    for task in selected_tasks:
        camera = task_configs[task].get('camera', 'logitech_1')
        train = episodes(args.dataset, args.frames, 'normal_train', task)
        reference = train[0]
        test = episodes(args.dataset, args.frames, 'test', task)
        if args.pilot:
            failure = next(e for e in test if e.index % 5 in (0, 1) and e.labels().any())
            test = [train[1], failure]
        folder = args.output / 'episodes' / task
        folder.mkdir(parents=True, exist_ok=True)
        pending = [e for e in test if not (folder / (e.name+'.json')).exists()]
        if not pending:
            continue
        instruction = task_configs[task]['instruction']
        turn1 = signature['turn1_prompt'].replace(old_task, instruction)
        turn2 = signature['turn2_prompt'].replace(old_task, instruction)
        frames = [f for f in reference.frame_table(camera) if f[1] >= float(vlm['sampling_start_sec'])]
        ids = np.linspace(0, len(frames)-1, int(vlm['num_frames']), dtype=int)
        images = [(f'Frame {k+1}, camera {topic}, time {frames[i][1]:.3f} s from this bag start:', load_image(frames[i][2])) for k, i in enumerate(ids)]
        reference_turn = dict(role='user', images=images, text=turn1)
        response_path = args.output / f'{task}_reference.json'
        if response_path.exists():
            response = json.loads(response_path.read_text())['response']
        else:
            response = model.generate_nominal(reference_turn, vlm['generation'])
            if not response.strip():
                raise ValueError('Empty nominal response.')
            response_path.write_text(json.dumps(dict(reference=reference.key, camera=camera, images=[str(frames[i][2]) for i in ids],
                turn1_prompt=turn1, turn2_prompt=turn2, response=response), indent=2))
        for episode in pending:
            assert episode.key != reference.key
            history = []
            due = float(vlm['sampling_start_sec'])
            observations, vectors, checks = [], [], []
            for frame_index, stamp, image_path in episode.frame_table(camera):
                if stamp + 1e-9 < due:
                    continue
                started = time.perf_counter()
                history.append((f'Frame {len(observations)+1}, camera {topic}, time {stamp:.3f} s from this bag start:', load_image(image_path)))
                turns = [reference_turn, dict(role='user', images=history, text=turn2)]
                vector = model.extract_multiturn_cop_vector(turns, vlm['generation'], reference_response=response).numpy()
                # Frozen cross-domain scoring: identities validated above; the changed task/reference
                # is explicitly recorded rather than misrepresented as an identical signature.
                score = classifier.predict_failure_probability(vector, input_mode='raw')
                if args.pilot and len(observations) < 2:
                    baseline = model.extract_multiturn_cop_vector(turns, vlm['generation'], reference_response=response).numpy()
                    comparison = vector_difference(baseline, vector)
                    baseline_score = classifier.predict_failure_probability(baseline, input_mode='raw')
                    comparison.update(score_difference=abs(score-baseline_score), decisions_match=(score >= threshold)==(baseline_score >= threshold))
                    if not comparison['within_tolerance'] or comparison['score_difference'] > .001 or not comparison['decisions_match']:
                        raise ValueError(f'Full-forward repeatability verification failed: {comparison}')
                    checks.append(comparison)
                interval = observation_interval(policy, score, float(vlm['online_replay']['interval_sec']))
                observations.append(dict(timestamp_sec=stamp, frame_index=frame_index, score=score,
                    threshold=threshold, next_interval_sec=interval, elapsed_sec=time.perf_counter()-started))
                vectors.append(vector)
                due = stamp + interval
            if not observations:
                raise ValueError('No observations scored.')
            first = next((x['timestamp_sec'] for x in observations if x['score'] >= threshold), None)
            result = dict(task=task, episode=episode.name, split=episode.split, reference=reference.key,
                camera=camera, first_alarm_sec=first, observations=observations, cache_verification=checks,
                completed=True, monitor_continues_for_diagnostics=not policy['stop_on_alarm'])
            np.savez_compressed(folder / (episode.name+'.npz'), vectors=np.stack(vectors))
            (folder / (episode.name+'.json')).write_text(json.dumps(result, indent=2))
            print(task, episode.split, episode.name, 'observations', len(observations), 'first alarm', first, flush=True)
        del history
        gc.collect()
        torch.cuda.empty_cache()
    summarize(args.output, args.dataset)


if __name__ == '__main__':
    main()
