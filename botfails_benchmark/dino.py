"""Our DINO detector on BotFails, one task at a time (memory in RAM, scores on disk).

Same code path as vision_dinov2/benchmark.py: nominal memory of every 2 fps frame
of the normal training episodes, leave-one-episode-out progress calibration
(95 % quantiles, smoothed), CLS top-5 matching, attention weights and the pooled
3x3 score. Only the source differs: JPEGs extracted from the AV1 videos instead
of compressed ROS images. BotFails has no stage messages, so matching is not
stage-constrained (our pipeline's ignore_recorded_stages mode); nominal progress
is the frame position in its episode.

Optional: a feature cache (float16, per episode) so that re-scoring needs no GPU,
and a subset of the test episodes (the development split first). Every scored
frame also records where the anomaly is (peak and box of the top 1 % pooled
patches, as fractions of the image) and its matched nominal frame.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import episodes, load_image, frame_labels


def settings(config):
    from vision_dinov2 import run_vision_test as defaults
    from vision_dinov2.model_settings import model_settings
    model, size = model_settings(config, defaults.DINO_INPUT_SIZE)
    return dict(model_name=model, dino_input_size=size, sample_fps=defaults.SAMPLE_FPS,
                top_k_cls=defaults.TOP_K_CLS, attention_power=defaults.ATTENTION_POWER,
                min_weight=defaults.MIN_WEIGHT, stage_constrained_matching=False,
                n_bins=defaults.THRESHOLD_BINS, min_points_per_bin=defaults.MIN_POINTS_PER_BIN,
                smooth_window=defaults.THRESHOLD_SMOOTH_WINDOW,
                threshold_floor_quantile=defaults.THRESHOLD_FLOOR_QUANTILE,
                threshold_scale=float(config.get('dino_threshold_scale', 1.)))


def episode_features(get_extractor, episode, camera, cache=None):
    """Frame table and float32 features, attention and CLS of an episode, from the cache if present."""
    frames = episode.frame_table(camera)
    path = Path(cache) / episode.split / episode.task / camera / f'{episode.name}.npz' if cache else None
    if path is not None and path.is_file():
        with np.load(path) as arrays:
            return (frames, arrays['features'].astype(np.float32), arrays['attention'].astype(np.float32),
                    arrays['cls'].astype(np.float32), int(arrays['grid']))
    extractor = get_extractor()
    features, attention, cls, grid = [], [], [], None
    for _, _, image_path in frames:
        f, a, c, grid = extractor.extract(load_image(image_path))
        features.append(f), attention.append(a), cls.append(c)
    features, attention, cls = np.stack(features), np.stack(attention), np.stack(cls)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp.npz')
        np.savez(temporary, features=features.astype(np.float16), attention=attention.astype(np.float16),
                 cls=cls, grid=np.asarray(grid))
        temporary.replace(path)
    return frames, features, attention, cls, grid


def build_memory(get_extractor, train, camera, log, cache=None):
    from vision_dinov2.anomaly_detection import add_progress_to_nominal_memory
    memory, grid, frames = [], None, []
    for video_id, episode in enumerate(train):
        table, features, attention, cls, size = episode_features(get_extractor, episode, camera, cache)
        if grid not in (None, size):
            raise ValueError('All nominal frames must have the same patch grid.')
        grid = size
        for frame_id, (frame_index, timestamp, _) in enumerate(table):
            memory.append(dict(video_id=video_id, video_path=episode.key, frame_id=frame_id, frame_index=frame_index,
                               timestamp_sec=timestamp, features=features[frame_id], attention=attention[frame_id],
                               cls=cls[frame_id]))
            frames.append(dict(video_id=video_id, episode=episode.key, frame_id=frame_id,
                               frame_index=frame_index, timestamp_sec=timestamp))
        log(f'  memory: {episode.key} ({len(table)} frames)')
    add_progress_to_nominal_memory(memory)  # frame position / last frame, per episode
    for item, frame in zip(memory, frames):
        frame['progress'] = item['progress']
    return memory, grid, pd.DataFrame(frames)


def calibrate(memory, metric, s):
    from vision_dinov2.threshold import calibrate_attention_leave_one_video_out
    return calibrate_attention_leave_one_video_out(
        memory, stage_constrained_matching=False, top_k_cls=s['top_k_cls'],
        attention_power=s['attention_power'], min_weight=s['min_weight'], metric=metric,
        n_bins=s['n_bins'], min_points_per_bin=s['min_points_per_bin'], smooth_window=s['smooth_window'],
        threshold_floor_quantile=s['threshold_floor_quantile'])


def anomaly_box(evidence):
    """Peak and bounding box of the top 1 % of the 3x3-pooled patch evidence, as image fractions."""
    grid = int(np.sqrt(evidence.size))
    padded = np.pad(np.asarray(evidence, float).reshape(grid, grid), 1, mode='edge')
    pooled = np.lib.stride_tricks.sliding_window_view(padded, (3, 3)).mean(axis=(-1, -2))
    count = max(1, int(np.ceil(.01 * pooled.size)))
    rows, columns = np.unravel_index(np.argsort(pooled, axis=None)[-count:], pooled.shape)
    peak_row, peak_column = np.unravel_index(np.argmax(pooled), pooled.shape)
    return dict(peak_x=(peak_column + .5) / grid, peak_y=(peak_row + .5) / grid, box_x0=columns.min() / grid,
                box_y0=rows.min() / grid, box_x1=(columns.max() + 1) / grid, box_y1=(rows.max() + 1) / grid)


def score_episode(get_extractor, episode, camera, memory, table, s, throttle_sec=0., cache=None):
    from vision_dinov2.anomaly_detection import compare_frame_to_nominal_memory
    from vision_dinov2.threshold import score_matched_frame
    frames, features, attention, cls, _ = episode_features(get_extractor, episode, camera, cache)
    labels = frame_labels(episode, [f for f, _, _ in frames]) if episode.split == 'test' else None
    rows = []
    for position, (frame_index, timestamp, _) in enumerate(frames):
        started = time.perf_counter()
        raw, patches, _, weights, best, cls_distance = compare_frame_to_nominal_memory(
            features[position], attention[position], cls[position], memory, top_k_cls=s['top_k_cls'],
            attention_power=s['attention_power'], min_weight=s['min_weight'], required_stage=None)
        decision = score_matched_frame(raw, float(best['progress']), table, patches=patches,
                                       attention_weights=weights, query_features=features[position],
                                       reference_features=best['features'])
        variants = decision['score_variants']
        rows.append(dict(task=episode.task, episode=episode.name, frame_index=frame_index,
                         timestamp_sec=timestamp, label=None if labels is None else int(labels[position]),
                         pooled_score=variants['pooled']['score'], base_threshold=variants['pooled']['threshold'],
                         weighted_score=variants['weighted']['score'],
                         weighted_threshold=variants['weighted']['threshold'],
                         raw_global_score=float(raw), cls_distance=float(cls_distance),
                         matched_progress=float(best['progress']), matched_episode=best['video_path'],
                         matched_frame_id=int(best['frame_id']), matched_frame_index=int(best['frame_index']),
                         comparison_sec=time.perf_counter() - started, **anomaly_box(decision['heatmap'])))
        if throttle_sec:
            time.sleep(throttle_sec)
    return rows


def load_calibration(directory):
    from vision_dinov2.patch_normality import validate_patch_profile
    table = pd.read_csv(directory / 'threshold.csv')
    with np.load(directory / 'patch_profile.npz', allow_pickle=False) as arrays:
        profile = {key: arrays[key] for key in arrays.files}
    validate_patch_profile(profile)
    table.attrs.update(patch_profile=profile, score_mode='attention')
    return table


def run_task(config, task, dataset, frames, output, metric='euclidean', camera='logitech_1',
             max_train=None, max_test=None, throttle_sec=0., extractor=None, log=print, cache=None, subset=None):
    """Memory + calibration from normal_train, then the selected test episodes.

    Resumable: calibration and every scored episode are kept. subset: test episode indices (None: all).
    """
    from vision_dinov2.dinov2_features import DINOv2FeatureExtractor
    s = settings(config)
    directory = Path(output) / task
    episode_directory = directory / 'episodes'
    episode_directory.mkdir(parents=True, exist_ok=True)
    train = episodes(dataset, frames, 'normal_train', task)[:max_train]
    test = episodes(dataset, frames, 'test', task)[:max_test]
    if subset is not None:
        test = [episode for episode in test if episode.index in subset]
    pending = [episode for episode in test if not (episode_directory / f'{episode.name}.csv').is_file()]
    if not pending:
        log(f'{task}: selected episodes already scored')
        return extractor
    while missing := [e.key for e in train + pending if not (e.frame_dir(camera) / 'frames.csv').is_file()]:
        log(f'{task}: waiting for extracted frames of {len(missing)} episodes, e.g. {missing[0]}')
        time.sleep(60)
    holder = [extractor]

    def get_extractor():
        if holder[0] is None:
            holder[0] = DINOv2FeatureExtractor(s['model_name'], s['dino_input_size'])
        return holder[0]

    timings, started = {}, time.perf_counter()
    log(f'{task}: memory from {len(train)} normal training episodes ({camera})')
    memory, grid, memory_frames = build_memory(get_extractor, train, camera, log, cache)
    timings['memory_sec'] = time.perf_counter() - started
    if (directory / 'threshold.csv').is_file() and (directory / 'patch_profile.npz').is_file():
        table = load_calibration(directory)
        log(f'{task}: calibration reused')
    else:
        log(f'{task}: {len(memory)} nominal frames, grid {grid}; leave-one-episode-out calibration ({metric})')
        started = time.perf_counter()
        table, rows = calibrate(memory, metric, s)
        timings['calibration_sec'] = time.perf_counter() - started
        table.to_csv(directory / 'threshold.csv', index=False)
        np.savez_compressed(directory / 'patch_profile.npz', **table.attrs['patch_profile'])
        # Leave-one-episode-out scores of the normal training frames (alarms in normal runs).
        rows = rows.merge(memory_frames.rename(columns={'video_id': 'held_out_video_id', 'frame_id': 'query_frame_id'}),
                          on=['held_out_video_id', 'query_frame_id'], how='left')
        rows.to_csv(directory / 'calibration_rows.csv', index=False)
        del rows
    started = time.perf_counter()
    for number, episode in enumerate(pending, 1):
        episode_rows = score_episode(get_extractor, episode, camera, memory, table, s, throttle_sec, cache)
        pd.DataFrame(episode_rows).to_csv(episode_directory / f'{episode.name}.csv', index=False)
        log(f'{task}: test {number}/{len(pending)} {episode.name}: {len(episode_rows)} frames')
    timings['test_sec'] = time.perf_counter() - started
    frames_written = sorted(episode_directory.glob('*.csv'))
    pd.concat([pd.read_csv(path) for path in frames_written]).to_csv(directory / 'test_scores.csv', index=False)
    (directory / 'done.json').write_text(json.dumps(dict(
        task=task, metric=metric, camera=camera, train_episodes=[e.key for e in train],
        test_episodes=[p.stem for p in frames_written], nominal_frames=len(memory), grid=grid, settings=s,
        timings=timings), indent=1))
    del memory, table
    gc.collect()
    return holder[0]
