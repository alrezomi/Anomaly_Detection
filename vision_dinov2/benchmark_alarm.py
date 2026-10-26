"""One benchmark alarm policy shared by decisions, plots and scale sweeps."""
import numpy as np
import json

ALARM_POLICY = 'two_consecutive_same_camera_before_stage3_v1'


def read_execution_annotation(bag, stage_topic='/recording_stage'):
    """Use first failure at/after corrected stage 1; retain raw messages for audit."""
    from rosbag_io import read_stage_events
    from rynnbrain_vlm.visual_matching import final_recorded_stages
    from build_dataset_manifest import FAIL_PATTERN
    result = dict(status='unavailable', time_sec=None, marker=None, marker_count=0,
                  error=None, recorded_stage_messages='', stage1_start_sec=None, stage3_start_sec=None)
    try:
        events = read_stage_events(bag, stage_topic)
        result.update(recorded_stage_messages=json.dumps([
            dict(time_sec=float(row.time), value=str(row.stage)) for row in events.itertuples(index=False)]),
            stage_message_count=len(events))
        timeline = final_recorded_stages(events)
        result.update(stage1_start_sec=timeline.starts[0], stage3_start_sec=timeline.starts[2],
                      ignored_pre_stage1_failure_count=sum(float(row.time)<timeline.starts[0] and
                          FAIL_PATTERN.search(str(row.stage)) is not None for row in events.itertuples(index=False)))
        return {**result, **timeline.failure_annotation()}
    except Exception as error:
        return {**result, 'error': f'{type(error).__name__}: {error}'}


def ensure_stage_windows(manifest, stage_topic='/recording_stage'):
    """Old saved runs need only stage metadata, never new feature extraction."""
    from rynnbrain_vlm.visual_matching import read_recorded_stages
    result = []
    for item in manifest:
        row = dict(item)
        if row['status'] == 'ok':
            value = row.get('stage3_start_sec')
            if value is None or value == '':
                value = read_recorded_stages(row['bag_path'], stage_topic).starts[2]
            value = float(value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid stage-3 start for {row["bag_path"]}.')
            row['stage3_start_sec'] = value
        result.append(row)
    return result


def eligible_frames(manifest, frames):
    result = frames.copy().reset_index(drop=True)
    windows = {item['bag_path']: item.get('stage3_start_sec') for item in manifest}
    if len(result):
        limits = result.bag_path.map(windows)
        if limits[result.bag_path.isin([x['bag_path'] for x in manifest if x['status']=='ok'])].isna().any():
            raise ValueError('Stage-3 timestamps are required to evaluate benchmark alarms.')
        result['alarm_eligible'] = result.timestamp_sec < limits
    else:
        result['alarm_eligible'] = np.zeros(0, dtype=bool)
    return result


def confirmation_pairs(frames):
    """Return adjacent row positions within each camera, never across cameras/bags."""
    first, second = [], []
    keys = [key for key in ['metric', 'bag_path', 'camera_topic'] if key in frames]
    for indices in frames.groupby(keys, sort=False).indices.values():
        order = indices[np.argsort(frames.iloc[indices].timestamp_sec.to_numpy(float), kind='stable')]
        if len(order) < 2:
            continue
        eligible = frames.iloc[order].alarm_eligible.to_numpy(bool)
        times = frames.iloc[order].timestamp_sec.to_numpy(float)
        pairs = eligible[1:] & eligible[:-1] & (times[1:] > times[:-1])
        first.extend(order[:-1][pairs])
        second.extend(order[1:][pairs])
    return np.asarray(first, dtype=int), np.asarray(second, dtype=int)


def confirmed_alarms(frames, scale):
    with np.errstate(over='ignore', under='ignore'):
        crossings = frames.pooled_score.to_numpy(float) > frames.base_threshold.to_numpy(float)*scale
    first, second = confirmation_pairs(frames)
    confirmed = np.zeros(len(frames), dtype=bool)
    confirmed[second] = crossings[first] & crossings[second]
    return confirmed
