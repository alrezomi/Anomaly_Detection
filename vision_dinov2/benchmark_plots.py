"""Readable metric folders and per-bag plots from already computed DINO scores."""
from pathlib import Path
import hashlib
import json
import re
import shutil

import numpy as np
import pandas as pd


def _bag_directory(path):
    name = re.sub(r'[^a-zA-Z0-9_-]+', '_', Path(path).name)[:80] or 'bag'
    return name + '_' + hashlib.sha256(path.encode()).hexdigest()[:10]


def _bag_plot(item, timeline, decisions, target, scale, metric, plt):
    cameras = sorted(timeline.camera_topic.unique()) if len(timeline) else []
    figure, axes = plt.subplots(max(1, len(cameras)), 1, squeeze=False,
                                figsize=(12, max(4, 3.5*len(cameras))))
    configured = decisions[decisions.operating_point.eq('configured')].iloc[0]
    onset = item.get('recorded_failure_time_sec')
    has_onset = onset is not None and pd.notna(onset) and np.isfinite(float(onset))
    for camera, axis in zip(cameras, axes.ravel()):
        rows = timeline[timeline.camera_topic.eq(camera)].sort_values('timestamp_sec')
        axis.plot(rows.timestamp_sec, rows.pooled_score, label='Anomaly score', color='#2878b5')
        axis.plot(rows.timestamp_sec, rows.configured_threshold, '--',
                  label=f'Threshold (scale {scale:g})', color='#c87925')
        alarms = rows.configured_alarm
        axis.scatter(rows.loc[alarms, 'timestamp_sec'], rows.loc[alarms, 'pooled_score'],
                     color='#c43838', s=10, label='Above threshold', zorder=3)
        if has_onset:
            axis.axvline(float(onset), color='black', linestyle=':', label='Recorded failure onset')
        if pd.notna(configured.first_alarm_time_sec):
            axis.axvline(configured.first_alarm_time_sec, color='#c43838', linestyle=':',
                         label='First alarm (any camera)')
        if 'best_threshold' in rows and rows.best_threshold.notna().any():
            axis.plot(rows.timestamp_sec, rows.best_threshold, '--', alpha=.6,
                      color='#6c58a0', label='Exploratory best threshold')
        axis.set(title=camera, xlabel='Observation time from bag start (s)', ylabel='Pooled anomaly score')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc='best')
    if not cameras:
        axes[0, 0].text(.5, .5, f'No scored frames. Status: {item["status"]}',
                        ha='center', va='center', transform=axes[0, 0].transAxes)
    predicted = configured.predicted_decision if pd.notna(configured.predicted_decision) else 'not evaluated'
    note = 'Recorded onset unavailable' if not has_onset else f'Recorded onset: {float(onset):.3f} s'
    if pd.notna(configured.failure_time_error_sec):
        note += f'; first alarm minus onset: {configured.failure_time_error_sec:+.3f} s'
    if item['status'] != 'ok':
        note += f'; incomplete evaluation: {item["status"]}'
    figure.suptitle(f'{metric.title()} - attention + 3x3 pooling | {item["bag_name"]}\n'
                    f'Ground truth: {item["ground_truth_label"]}; prediction: {predicted}. {note}', fontsize=10)
    figure.tight_layout()
    figure.savefig(target/'score_threshold.png', dpi=140)
    plt.close(figure)


def _scale_plot(frames, sweep_path, target, scale, plt):
    figure, axis = plt.subplots(figsize=(9, 4))
    if sweep_path.is_file():
        sweep = pd.read_csv(sweep_path)
        ratios = []
        for _, rows in frames.groupby('bag_path'):
            positive = rows.base_threshold > 0
            ratios.extend((rows.loc[positive, 'pooled_score']/rows.loc[positive, 'base_threshold']).tolist())
        values = [float(x) for x in ratios if np.isfinite(x) and x > 0] + [float(scale)]
        low, high = min(values)/2, max(values)*1.2
        view = sweep[sweep.threshold_scale.between(low, high)]
        if len(view):
            axis.plot(view.threshold_scale, view.alarm_rate, label='All evaluated bags', color='#2878b5')
        axis.axvline(scale, color='#c87925', linestyle='--', label=f'Configured scale {scale:g}')
        axis.set_xscale('log')
        axis.legend()
    else:
        axis.text(.5, .5, 'No evaluated bags available.', ha='center', transform=axis.transAxes)
    axis.set(xlabel='Threshold scale (log axis)', ylabel='Fraction of bags with any alarm', ylim=(-.02, 1.02),
             title='Alarm rate versus threshold scale (not an ROC)')
    axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(target/'alarm_rate_vs_scale.png', dpi=140)
    plt.close(figure)


def write_metric_reports(manifest, frames, results, metrics, roc, output, scale, plt):
    overview = ['# DINO benchmark results', '',
                'Decision: any sampled pooled-score crossing in any selected camera is an alarm.',
                'Ground-truth task outcomes and visual anomalies can differ. Labels are preserved.', '',
                'Open `euclidean/summary.md` or `mahalanobis/summary.md` for evaluation and plot links.', '']
    for metric in ('euclidean', 'mahalanobis'):
        target = output/metric
        target.mkdir(exist_ok=True)
        subset = results[results.metric.eq(metric)]
        subset.to_csv(target/'bag_results.csv', index=False)
        selected_metrics = metrics[metrics.metric.eq(metric)]
        selected_metrics.to_csv(target/'metrics.csv', index=False)
        selected_frames = frames[frames.metric.eq(metric)]
        selected_frames.to_csv(target/'frame_results.csv', index=False)
        for name in ['alarm_diagnostics.csv', 'runtime_summary.csv']:
            table = pd.read_csv(output/name)
            if 'metric' in table:
                table[table.metric.eq(metric)].to_csv(target/name, index=False)
        for path in output.glob(f'{metric}_failure_timing_*.png'):
            shutil.copy2(path, target/path.name.replace(f'{metric}_', '', 1))
        for suffix in ['scale_sweep.csv', 'roc.csv']:
            source = output/f'{metric}_{suffix}'
            if source.is_file():
                shutil.copy2(source, target/suffix)
        (target/'roc_summary.json').write_text(json.dumps(roc[metric], indent=2, allow_nan=False), encoding='utf-8')
        _scale_plot(selected_frames, output/f'{metric}_scale_sweep.csv', target, scale, plt)
        figure, axis = plt.subplots(figsize=(7, 5))
        if roc[metric]['status'] == 'created':
            points = pd.read_csv(target/'roc.csv')
            axis.plot(points.false_positive_rate, points.recall, label=f'AUC={roc[metric]["auc"]:.3f}')
            best = roc[metric]['best']
            axis.scatter(best['false_positive_rate'], best['recall'], marker='*', s=140,
                         label=f'Exploratory best scale {best["threshold_scale"]:.6g}')
            current = pd.read_csv(target/'scale_sweep.csv')
            current = current[current.threshold_scale.eq(scale)].iloc[0]
            axis.scatter(current.false_positive_rate, current.recall, label=f'Configured scale {scale:g}')
            axis.plot([0, 1], [0, 1], '--', color='gray', alpha=.5)
            axis.set(xlabel='False-positive rate', ylabel='Recall', xlim=(-.02, 1.02), ylim=(-.02, 1.02))
            axis.legend(fontsize=8)
        else:
            axis.text(.5, .5, 'ROC unavailable: both ground-truth classes are required.\n'
                      'Use alarm_rate_vs_scale.png to inspect sensitivity.\nNo best scale can be selected from this single-class set.',
                      ha='center', va='center', transform=axis.transAxes, fontsize=9)
            axis.set_axis_off()
        axis.set_title(f'{metric.title()} - attention + 3x3 pooling')
        figure.tight_layout()
        figure.savefig(target/'roc.png', dpi=140)
        plt.close(figure)
        index = []
        grouped = {bag: rows for bag, rows in selected_frames.groupby('bag_path')}
        for item in manifest:
            if item['status'] == 'excluded_training':
                continue
            bag = item['bag_path']
            relative = Path('bags')/_bag_directory(bag)
            destination = target/relative
            destination.mkdir(parents=True, exist_ok=True)
            timeline = grouped.get(bag, selected_frames.iloc[:0])
            bag_results = subset[subset.bag_path.eq(bag)]
            timeline.to_csv(destination/'frame_results.csv', index=False)
            bag_results.to_csv(destination/'decision.csv', index=False)
            _bag_plot(item, timeline, bag_results, destination, scale, metric, plt)
            index.append(dict(bag_path=bag, bag_name=item['bag_name'], folder=item['folder'], status=item['status'],
                              plot=str(relative/'score_threshold.png').replace('\\', '/')))
        pd.DataFrame(index, columns=['bag_path', 'bag_name', 'folder', 'status', 'plot']).to_csv(target/'bag_index.csv', index=False)
        overall = selected_metrics[(selected_metrics.folder == 'ALL') & selected_metrics.operating_point.eq('configured')].iloc[0]
        configured = subset[subset.operating_point.eq('configured')]
        lines = [f'# {metric.title()} benchmark', '', f'Threshold scale: {scale:g}. Attention + 3x3 pooling.', '',
                 f'- Selected: {int(overall.selected_bags)}; evaluated: {int(overall.evaluated_bags)}.',
                 f'- Status counts: {configured.status.value_counts().to_dict()}.',
                 f'- Ground-truth labels: {int(overall.normal_count)} normal, {int(overall.failure_count)} failure.',
                 f'- Bags with alarms: {int(overall.alarm_bags)}; without alarms: {int(overall.no_alarm_bags)}.',
                 f'- Comparable failure onset/alarm pairs: {int(overall.timing_compared)}.', '']
        if not overall.failure_count:
            lines += ['Recall and ROC are unavailable: no failure-labelled cases were evaluated.',
                      'Alarms on normal-labelled cases count as false positives against these labels.', '']
        lines += ['| Folder | Evaluated | Alarm bags | Alarm rate | Recall | False-positive rate |',
                  '|---|---:|---:|---:|---:|---:|']
        def rate(value):
            return 'N/A' if pd.isna(value) else f'{100*value:.1f}%'
        for row in selected_metrics[selected_metrics.operating_point.eq('configured')].itertuples():
            lines.append(f'| {row.folder} | {row.evaluated_bags} | {row.alarm_bags} | {rate(row.alarm_rate)} | '
                         f'{rate(row.recall)} | {rate(row.false_positive_rate)} |')
        lines += ['', '## Plots', '', '- [Alarm rate versus threshold scale](alarm_rate_vs_scale.png)',
                  '- [ROC / availability explanation](roc.png)',
                  '- [Per-bag plot index](bag_index.csv)', '']
        lines += [f'- [Timing page {page.stem}]({page.name})' for page in sorted(target.glob('failure_timing_*.png'))]
        lines += ['', '## Per-bag score and threshold plots', '']
        lines += [f'- [{row["folder"]}/{row["bag_name"]}]({row["plot"]}) ({row["status"]})' for row in index]
        (target/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        overview += [f'## {metric.title()}', '', f'Alarm bags: {int(overall.alarm_bags)}/{int(overall.evaluated_bags)}. '
                     f'Recall: {rate(overall.recall)}; false-positive rate: {rate(overall.false_positive_rate)}.',
                     f'[Full evaluation and plots]({metric}/summary.md)', '']
    overview += ['Timing differences require an actual recorded onset. Missing timestamps remain unavailable.',
                 'For single-class datasets, alarm-rate sweeps are shown; ROC and best-scale selection are undefined.',
                 'Suggested scales from two-class ROC are exploratory tuning on this set, not independent validation.', '']
    (output/'SUMMARY.md').write_text('\n'.join(overview), encoding='utf-8')
