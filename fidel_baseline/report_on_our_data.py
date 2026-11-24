"""Compare FIDeL with our DINO -> VLM cascade on our pick-and-place bags: same frames, same labels.

Frames: top camera, 2 fps, from stage 1 until stage 3 + 5 s. A frame is positive in a failure bag from the recorded failure
onset on. Failure bags without a recorded onset count only at the episode level.
Episode level: an alarm anywhere in that window; recall over failure bags, false alarms over normal bags.
Both W.Acc formulas are given, because FIDeL's paper and its released code differ:
paper: beta * TNR + (1 - beta) * TPR; code: beta * TPR + (1 - beta) * TNR; beta = normal frames / all frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

POST_STAGE3_SEC = 5.
RUN = 3  # Consecutive frames, the rule of our alarm.


def slug(bag):
    return '__'.join(Path(bag).parts[-3:]).replace(' ', '_')


def rates(labels, predictions):
    labels, predictions = np.asarray(labels, bool), np.asarray(predictions, bool)
    tpr = (predictions & labels).sum() / max(labels.sum(), 1)
    tnr = (~predictions & ~labels).sum() / max((~labels).sum(), 1)
    beta = (~labels).sum() / len(labels)
    return dict(TPR=100 * tpr, TNR=100 * tnr, Acc=100 * (tpr + tnr) / 2, WAcc_paper=100 * (beta * tnr + (1 - beta) * tpr),
                WAcc_code=100 * (beta * tpr + (1 - beta) * tnr), frames=len(labels), positive_frames=int(labels.sum()))


def threshold_free(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return dict(AUROC=100 * roc_auc_score(labels, scores), F1_opt=100 * f1.max())


def first_run(flags, times, run=RUN):
    """Time of the first `run` consecutive True flags (run=1: the first True), or None."""
    streak = 0
    for flag, t in zip(flags, times):
        streak = streak + 1 if flag else 0
        if streak >= run:
            return float(t)
    return None


def load_ours(bag_results, frame_results):
    rows = pd.read_csv(bag_results)
    rows = rows[(rows.metric == 'euclidean') & (rows.status == 'ok')]
    stops = {point: part.set_index('bag_path').first_alarm_time_sec for point, part in rows.groupby('operating_point')}
    bags = rows[rows.operating_point == 'dino_vlm_check'].set_index('bag_path')
    frames = pd.read_csv(frame_results, usecols=['bag_path', 'metric', 'timestamp_sec', 'pooled_score', 'base_threshold'])
    return bags, stops, frames[frames.metric == 'euclidean']


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fidel-results', type=Path, required=True)
    parser.add_argument('--bag-results', type=Path, required=True)
    parser.add_argument('--frame-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bags, stops, our_frames = load_ours(args.bag_results, args.frame_results)
    our_frames = {bag: part.sort_values('timestamp_sec') for bag, part in our_frames.groupby('bag_path')}
    rows, frames = [], []
    for bag, info in bags.iterrows():
        path = args.fidel_results / 'bags' / f'{slug(bag)}.npz'
        if not path.is_file():
            continue
        with np.load(path) as data:
            fidel = pd.DataFrame({k: data[k] for k in ('timestamp_sec', 'score', 'flag', 'vlm', 'detect_time', 'space_fraction_per_patch')})
        onset = info.recorded_failure_time_sec if info.ground_truth_label == 'fail' else np.nan
        fail = info.ground_truth_label == 'fail'
        fidel['label'] = (fidel.timestamp_sec >= onset).astype(int) if np.isfinite(onset) else 0
        fidel['usable'] = (not fail) or bool(np.isfinite(onset))  # Failure bags without onset: episode level only.
        fidel['e2e'] = fidel.flag.astype(bool) & (fidel.vlm == 1)
        ours = our_frames[bag]
        ours = ours[ours.timestamp_sec <= float(info.stage3_start_sec) + POST_STAGE3_SEC]
        merged = pd.merge_asof(fidel.sort_values('timestamp_sec'), ours[['timestamp_sec', 'pooled_score', 'base_threshold']],
                               on='timestamp_sec', direction='nearest', tolerance=.08)
        time = merged.timestamp_sec.to_numpy()
        stop_vlm, stop_dino = stops['dino_vlm_check'][bag], stops['dino_alone'][bag]
        merged['ours_vlm'] = time >= stop_vlm if np.isfinite(stop_vlm) else False
        merged['ours_dino'] = time >= stop_dino if np.isfinite(stop_dino) else False
        merged['folder'], merged['bag'] = info.folder, bag
        frames.append(merged)
        row = dict(bag=bag, folder=info.folder, label=info.ground_truth_label, onset=onset, frames=len(merged),
                   flagged=int(merged.flag.sum()), vlm_failure=int(merged.e2e.sum()), matched=int(merged.pooled_score.notna().sum()),
                   ours_vlm=stop_vlm, ours_dino=stop_dino)
        for name, flags in (('fidel_ad', merged.flag), ('fidel_e2e', merged.e2e)):
            row[f'{name}_any'] = first_run(flags.astype(bool), time, 1)
            row[f'{name}_run3'] = first_run(flags.astype(bool), time)
        rows.append(row)
    per_bag = pd.DataFrame(rows)
    per_bag.to_csv(args.output / 'per_bag.csv', index=False)
    all_frames = pd.concat(frames, ignore_index=True)
    usable = all_frames[all_frames.usable & all_frames.pooled_score.notna()]
    table = []
    labels = usable.label.astype(int)
    detector = {'FIDeL anomaly score': usable.score, 'Our DINO score': usable.pooled_score,
                'Our DINO score / threshold': usable.pooled_score / usable.base_threshold}
    auc = pd.DataFrame([dict(detector=name, **threshold_free(labels, s)) for name, s in detector.items()])
    systems = {'FIDeL detector (flag per frame)': usable.flag.astype(bool), 'FIDeL full (flag and VLM failure, per frame)': usable.e2e,
               'Our DINO alone (latched stop)': usable.ours_dino.astype(bool), 'Our DINO + VLM check (latched stop)': usable.ours_vlm.astype(bool)}
    table = pd.DataFrame([dict(system=name, **rates(labels, p)) for name, p in systems.items()])

    def episode(column):
        fail, normal = per_bag[per_bag.label == 'fail'], per_bag[per_bag.label == 'normal']
        recall, fpr = fail[column].notna().mean() * 100, normal[column].notna().mean() * 100
        timed = fail[fail.onset.notna() & fail[column].notna()]
        return dict(recall=recall, false_alarm=fpr, caught=int(fail[column].notna().sum()), failures=len(fail),
                    false_alarms=int(normal[column].notna().sum()), normals=len(normal),
                    after_onset=int((timed[column] >= timed.onset).sum()), median_delay_sec=(timed[column] - timed.onset).median())

    episodes = {'FIDeL detector, any flagged frame': 'fidel_ad_any', 'FIDeL detector, 3 frames in a row': 'fidel_ad_run3',
                'FIDeL full, any confirmed frame': 'fidel_e2e_any', 'FIDeL full, 3 frames in a row': 'fidel_e2e_run3',
                'Our DINO alone': 'ours_dino', 'Our DINO + VLM check': 'ours_vlm'}
    episode_table = pd.DataFrame([dict(system=name, **episode(column)) for name, column in episodes.items()])
    folders = []
    for folder, part in per_bag.groupby('folder', sort=False):
        row = dict(folder=folder, label=part.label.iloc[0], bags=len(part))
        for name, column in (('FIDeL full any', 'fidel_e2e_any'), ('FIDeL full 3 in a row', 'fidel_e2e_run3'), ('Our DINO + VLM', 'ours_vlm'), ('Our DINO alone', 'ours_dino')):
            row[name] = f'{int(part[column].notna().sum())}/{len(part)}'
        folders.append(row)
    folder_table = pd.DataFrame(folders)
    timing = dict(frames=len(all_frames), flagged_share=float(all_frames.flag.mean()), matched_share=float(all_frames.pooled_score.notna().mean()),
                  failure_bags_without_onset=int(((per_bag.label == 'fail') & per_bag.onset.isna()).sum()))
    for name, frame in (('detector_metrics', auc), ('frame_metrics', table), ('episode_metrics', episode_table), ('per_folder', folder_table)):
        frame.to_csv(args.output / f'{name}.csv', index=False, float_format='%.2f')
    (args.output / 'summary.json').write_text(json.dumps(timing, indent=1))
    pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_colwidth', 60)
    for title, frame in (('Detector, threshold-free', auc), ('Frames', table), ('Episodes', episode_table), ('Per folder (bags with an alarm / bags)', folder_table)):
        print(f'\n=== {title}\n' + frame.round(1).to_string(index=False))
    print('\n', timing)


if __name__ == '__main__':
    main()
