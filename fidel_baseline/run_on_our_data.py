"""Run FIDeL (Rolland et al., ICRA 2026) on our pick-and-place bags, to compare it with our cascade.

FIDeL's own modules do the work, imported unmodified from a separate checkout (--fidel-src, commit 33f67be; the repository
has no licence, so its code is not copied here): Representation memory and score, conformal thresholds over time and space
(best configuration in the paper, settings from its yaml files), heatmap overlay and Qwen2.5-VL-7B filter.

Same data as our benchmark: memory and calibration from config nominal_bags, test bags and labels from bag_results.csv, the
top camera at 2 fps, from stage 1 until stage 3 + 5 s.

What had to change, because the released code cannot run on this data as it is:
- Bags instead of LeRobot datasets. Two modules FIDeL imports are not in its repository, and the LeRobot loader is not
  needed, so those imports are stubbed.
- The whole image is resized to 224 x 224, as our own DINO gets it. FIDeL's resize-and-center-crop would keep only the middle
  44 % of the width of our 960 x 480 top camera and cut off the pick location and most of the basket.
- The time threshold has one value per frame of the shortest normal run; a longer test run uses the last value.
- Its loop expects a (decision, threshold) pair, but this threshold class returns only the decision.
- The VLM prompt says "A red robot arm"; ours is not red, so that says "A robot arm".
- The VLM generates 3 tokens instead of 128: FIDeL only reads the first character of the answer, so the decision is the same.
- The VLM is pinned to cuda:0 (on Jetson Thor device_map "auto" offloads to the CPU).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
DEVICE = 'cuda'
PATCHES, FEATURES = 256, 768  # DINOv2-base at 224 px.
POST_STAGE3_SEC = 5.


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def load_fidel(src: Path):
    sys.path.insert(0, str(src))
    for name, attribute in (('threshold.conformal_prediction_time_2', 'ConformalPredictionTime2'),
                            ('threshold.conformal_prediction_space_all_2', 'ConformalPredictionSpaceAll2')):
        module = types.ModuleType(name)
        setattr(module, attribute, object)
        sys.modules[name] = module
    for name in ('lerobot', 'lerobot.datasets', 'lerobot.datasets.factory'):
        sys.modules[name] = types.ModuleType(name)
    sys.modules['lerobot.datasets.factory'].make_dataset = None
    from anomaly_detection.Representation import representation
    from threshold.conformal_prediction_time_and_space import ConformalPredictionTimeAndSpace
    from failure_detection_and_recovery import superpose_heatmap
    from VLM_qwen import VLM_SemanticFilter
    return SimpleNamespace(representation=representation, Jury=ConformalPredictionTimeAndSpace,
                           superpose_heatmap=superpose_heatmap, VLM=VLM_SemanticFilter)


def fidel_config(src: Path, prompt: str, distance: str | None):
    import yaml
    threshold = yaml.safe_load((src / 'cfgs/threshold_type/conformal_prediction_time_and_space.yaml').read_text())
    representation = yaml.safe_load((src / 'cfgs/ad_type/Representation.yaml').read_text())
    if distance:
        representation['distance_type'] = distance
    return SimpleNamespace(device=DEVICE, proj_net=False, encoder='dinoV2', prompt=prompt,
                           anomaly_detection=SimpleNamespace(**representation), threshold=SimpleNamespace(**threshold))


class Encoder:
    """DINOv2-base patch features of the whole image resized to 224 x 224 (FIDeL's normalization, no center crop)."""
    def __init__(self, path):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(path, use_fast=False)
        self.model = AutoModel.from_pretrained(path).to(DEVICE).eval()

    @torch.inference_mode()
    def __call__(self, tensor):
        pixels = self.processor(images=[tensor], return_tensors='pt', do_rescale=False, do_center_crop=False,
                                size=dict(height=224, width=224))['pixel_values'].to(DEVICE)
        return self.model(pixel_values=pixels).last_hidden_state[:, 1:, :].squeeze(0)


def to_tensor(image):
    return torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2, 0, 1).float() / 255.


def bag_frames(bag, topic, stage_topic):
    """(frame id, time in the bag, PIL image) from stage 1 until stage 3 + 5 s, at 2 fps, as our benchmark samples them."""
    from rosbag_io import RosbagImageSource, iter_rosbag_image_frames
    from rynnbrain_vlm.visual_matching import read_recorded_stages
    starts = read_recorded_stages(bag, stage_topic).starts
    yield from iter_rosbag_image_frames(RosbagImageSource(bag, topic), sample_fps=2., start_sec=starts[0],
                                        end_sec=starts[-1] + POST_STAGE3_SEC)


def slug(bag):
    return '__'.join(Path(bag).parts[-3:]).replace(' ', '_')


@torch.inference_mode()
def build_memory(fidel, cfg, config, encoder, output, log):
    """Memory, expert scores and heatmaps of the normal bags, following store_memory_data.py and dataset_eval.py."""
    path = output / 'memory.pt'
    if path.is_file():
        log('memory reused')
        return torch.load(path, weights_only=False)
    topic, stage_topic = config['camera_topics'][0], config.get('stage_topic', '/recording_stage')
    episodes, expert_frames = [], []
    (output / 'expert_frames').mkdir(exist_ok=True)
    for bag in config['nominal_bags']:
        features = []
        for _, _, image in bag_frames(bag, topic, stage_topic):
            features.append(encoder(to_tensor(image)).float())
            name = f'{len(expert_frames):06d}.jpg'
            image.convert('RGB').save(output / 'expert_frames' / name, quality=95)
            expert_frames.append(name)
        episodes.append(features)
        log(f'  memory {Path(bag).name}: {len(features)} frames')
    longest = max(len(e) for e in episodes)
    network, _, _ = fidel.representation(len(episodes), longest, PATCHES, FEATURES, cfg, True)
    network = network.to(DEVICE)
    network.train()

    def feed():
        for index, episode in enumerate(episodes):
            for t, features in enumerate(episode):
                network(dict(features=features, episode_index=index, frame_index=t))

    feed()
    network.memory.compute_mean()
    network.memory.counter.zero_()
    network.memory.flag_avg_has_been_computed = True
    feed()
    network.memory.compute_std()
    network.train_mode = False
    network.eval()
    scores, heatmaps = [], []
    for episode in episodes:  # FIDeL calibrates on the same demonstrations that form the memory.
        outputs = [network(features) for features in episode]
        scores.append([float(score) for score, _ in outputs])
        heatmaps.append([patch.squeeze().float().cpu() for _, patch in outputs])
    shortest = min(len(s) for s in scores)
    state = dict(mean=network.memory.memory_avg.cpu(), std=network.memory.memory_std.cpu(),
                 scores=torch.tensor([s[:shortest] for s in scores], dtype=torch.float32),
                 heatmaps=torch.stack([torch.stack(h[:shortest]) for h in heatmaps]),
                 longest=longest, shortest=shortest, expert_frames=expert_frames)
    torch.save(state, path)
    log(f'memory: {len(episodes)} normal bags, {longest} time steps (shortest run {shortest})')
    return torch.load(path, weights_only=False)


def load_vlm(fidel, path):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    log('loading Qwen2.5-VL-7B')
    vlm = fidel.VLM.__new__(fidel.VLM)
    vlm.device, vlm.model_name = DEVICE, str(path)
    vlm.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype='auto', device_map={'': 'cuda:0'}).eval()
    vlm.processor = AutoProcessor.from_pretrained(path)
    released_prompt, released_generate = vlm.build_prompt, vlm.model.generate
    vlm.build_prompt = lambda task: released_prompt(task).replace('A red robot arm', 'A robot arm')
    vlm.model.generate = lambda **kwargs: released_generate(**{**kwargs, 'max_new_tokens': 3})
    return vlm


@torch.inference_mode()
def score_bag(fidel, cfg, network, jury, encoder, vlm, state, row, config, output, expert_dir, vlm_pause=0.):
    topic, stage_topic = config['camera_topics'][0], config.get('stage_topic', '/recording_stage')
    last = state['shortest'] - 1
    columns = {k: [] for k in ('frame_id', 'timestamp_sec', 'score', 'detect_time', 'space_fraction', 'space_fraction_per_patch',
                               'flag', 'vlm', 'vlm_sec', 'feature_sec', 'score_sec')}
    heatmaps = []
    for position, (frame_id, timestamp, image) in enumerate(bag_frames(row['bag_path'], topic, stage_topic)):
        tensor, started = to_tensor(image), time.perf_counter()
        with torch.autocast(device_type='cuda'):  # As FIDeL's eval_ad.
            features = encoder(tensor)
            feature_sec = time.perf_counter() - started
            started = time.perf_counter()
            score, heatmap = network(features.float())
        score_sec = time.perf_counter() - started
        t = min(position, last)
        flag = bool(jury(score, heatmap.float(), dict(frame_index=torch.tensor(t))))
        detect_time = bool(float(score) > float(jury.threshold_time[t]))
        released = float((heatmap.float() >= jury.threshold_space).float().mean())
        per_patch = float((heatmap.squeeze().float() >= jury.threshold_space).float().mean())
        answer, vlm_sec = None, 0.
        if flag and vlm is not None:
            started = time.perf_counter()
            overlay = fidel.superpose_heatmap(tensor, heatmap.float())
            from PIL import Image
            from torchvision.transforms.functional import to_pil_image
            name = state['expert_frames'][min(position, len(state['expert_frames']) - 1)]  # FIDeL: dataset[t].
            with Image.open(expert_dir / name) as reference:
                expert = to_tensor(reference)
            answer, _ = vlm.analyze(cfg, expert, to_pil_image(overlay.squeeze()), cfg.prompt, '')
            vlm_sec = time.perf_counter() - started
            if vlm_pause:
                time.sleep(vlm_pause)  # Leaves GPU time for the robot stack on this shared machine.
        for key, value in (('frame_id', frame_id), ('timestamp_sec', timestamp), ('score', float(score)), ('detect_time', detect_time),
                           ('space_fraction', released), ('space_fraction_per_patch', per_patch), ('flag', flag),
                           ('vlm', -9 if answer is None else answer), ('vlm_sec', vlm_sec), ('feature_sec', feature_sec), ('score_sec', score_sec)):
            columns[key].append(value)
        heatmaps.append(heatmap.squeeze().float().cpu().numpy().astype(np.float16))
    temporary = output / 'bags' / f'{slug(row["bag_path"])}.tmp.npz'
    np.savez(temporary, heatmap=np.stack(heatmaps), **{k: np.asarray(v) for k, v in columns.items()})
    temporary.replace(temporary.with_name(temporary.name.replace('.tmp.npz', '.npz')))
    return len(heatmaps), sum(columns['flag']), sum(v == 1 for v in columns['vlm'])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', type=Path, default=Path('/config/pipeline_config.json'))
    parser.add_argument('--fidel-src', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--bag-results', type=Path, required=True, help='bag_results.csv of our VLM check benchmark')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-folder', type=int, help='Quick check: first N test bags of each folder.')
    parser.add_argument('--distance', help='Override distance_type of FIDeL\'s yaml (e.g. optimal_transport).')
    parser.add_argument('--no-vlm', action='store_true')
    parser.add_argument('--throttle-sec', type=float, default=0.)
    parser.add_argument('--vlm-pause-sec', type=float, default=0., help='Pause after each VLM call (shared GPU).')
    args = parser.parse_args()
    from pipeline_settings import load_pipeline_config
    config = load_pipeline_config(args.config)
    fidel = load_fidel(args.fidel_src)
    cfg = fidel_config(args.fidel_src, config['rynnbrain']['task_description'], args.distance)
    (args.output / 'bags').mkdir(parents=True, exist_ok=True)
    (args.output / 'run.json').write_text(json.dumps(dict(
        command=sys.argv, fidel_commit=(args.fidel_src / 'COMMIT.txt').read_text().strip(), prompt=cfg.prompt, camera=config['camera_topics'],
        threshold=vars(cfg.threshold), distance=cfg.anomaly_detection.distance_type, nominal_bags=config['nominal_bags'],
        image='whole image resized to 224 x 224',
        started=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1))
    encoder = Encoder(args.models / 'dinov2-base')
    state = build_memory(fidel, cfg, config, encoder, args.output, log)
    network, _, _ = fidel.representation(0, 0, 0, 0, cfg, False)
    network.memory.memory_avg, network.memory.memory_std = state['mean'].to(DEVICE), state['std'].to(DEVICE)
    network = network.to(DEVICE).eval()
    random.seed(2)  # FIDeL's config seed; the threshold class splits the demonstrations at random.
    jury = fidel.Jury(cfg, state['scores'], state['heatmaps'].to(DEVICE))
    log(f'threshold: time h={jury.h_time:.3f}, space h={jury.h_space:.3f}, patches needed {cfg.threshold.percentage_patch_detection:.0%}')
    with args.bag_results.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['metric'] == 'euclidean' and r['operating_point'] == 'dino_vlm_check' and r['status'] == 'ok']
    if args.per_folder:
        seen, picked = {}, []
        for row in rows:
            seen[row['folder']] = seen.get(row['folder'], 0) + 1
            if seen[row['folder']] <= args.per_folder:
                picked.append(row)
        rows = picked
    vlm = None if args.no_vlm else load_vlm(fidel, args.models / 'Qwen2.5-VL-7B-Instruct')
    for number, row in enumerate(rows, 1):
        if (args.output / 'bags' / f'{slug(row["bag_path"])}.npz').is_file():
            continue
        started = time.perf_counter()
        frames, flagged, failures = score_bag(fidel, cfg, network, jury, encoder, vlm, state, row, config, args.output, args.output / 'expert_frames', args.vlm_pause_sec)
        log(f"{number}/{len(rows)} {row['folder']}/{row['bag_name']} ({row['ground_truth_label']}): {frames} frames, "
            f"{flagged} flagged, {failures} VLM failure, {time.perf_counter() - started:.0f} s")
        if args.throttle_sec:
            time.sleep(args.throttle_sec)
    log('done')


if __name__ == '__main__':
    main()
