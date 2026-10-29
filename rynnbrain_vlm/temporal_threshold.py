"""Select a cutoff using first-alarm delay and balanced per-execution costs.

This is an application-specific normalized cost, not the official NAB score.
It needs neither a delay deadline nor an assumed cost in seconds per false alarm.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .runtime import write_json
from .threshold_sweep import THRESHOLDS


def _population(scans, annotations):
    eligible, excluded = [], []
    for record, _, scan in scans:
        name, label = record['bag_name'], record['label']
        reason = None
        times = np.asarray([row['timestamp_sec'] for row in scan['rows']], dtype=float)
        scores = np.asarray([row['anomaly_score'] for row in scan['rows']], dtype=float)
        onset = annotations.get(name, {}).get('time_sec')
        if label not in {'normal', 'fail'}:
            reason = 'unknown_ground_truth'
        elif not scan['complete']:
            reason = 'incomplete_scan'
        elif (not len(times) or not np.isfinite(times).all() or (times < 0).any()
              or (np.diff(times) <= 0).any() or not np.isfinite(scores).all()
              or ((scores < 0) | (scores > 1)).any()):
            reason = 'invalid_or_empty_timeline'
        elif label == 'fail':
            annotation = annotations.get(name, {})
            if annotation.get('status') != 'available' or onset is None or not np.isfinite(onset):
                reason = 'missing_failure_onset'
            elif onset < times[0] or onset >= times[-1]:
                reason = 'failure_onset_outside_scored_window'
        if reason:
            excluded.append(dict(bag_name=name, ground_truth_label=label, reason=reason,
                                 annotation_error=annotations.get(name, {}).get('error')))
        else:
            eligible.append(dict(bag_name=name, ground_truth_label=label, times=times, scores=scores,
                                 onset=float(onset) if label == 'fail' else None))
    return eligible, excluded


def first_alarm_cost(item, threshold):
    """One first alarm per execution, matching stop-on-alarm replay semantics."""
    times, scores, onset = item['times'], item['scores'], item['onset']
    indices = np.flatnonzero(scores >= threshold)
    alarm = float(times[indices[0]]) if len(indices) else None
    delay, normalized = None, None
    if item['ground_truth_label'] == 'normal':
        status, cost = ('false_alarm', 1.) if alarm is not None else ('normal_no_alarm', 0.)
    elif alarm is None:
        status, cost = 'missed_failure', 1.
    elif alarm < onset:
        status, cost = 'premature_alarm', 1.
        delay = alarm - onset
    else:
        status = 'post_onset_detection'
        delay = alarm - onset
        normalized = float(np.clip(delay / (times[-1] - onset), 0., 1.))
        cost = normalized
    return dict(bag_name=item['bag_name'], ground_truth_label=item['ground_truth_label'],
        threshold=threshold, threshold_percent=round(100 * threshold), first_alarm_sec=alarm,
        recorded_failure_time_sec=onset, last_scored_time_sec=float(times[-1]),
        post_onset_window_sec=float(times[-1] - onset) if onset is not None else None,
        status=status, delay_sec=delay, normalized_delay=normalized, cost=cost)


def write_temporal_cost_report(scans, annotations, root):
    """Evaluate saved, complete, uncensored histories; never call a model."""
    root = Path(root) / 'benchmark_temporal_cost'
    root.mkdir(parents=True, exist_ok=True)
    eligible, excluded = _population(scans, annotations)
    normal_count = sum(item['ground_truth_label'] == 'normal' for item in eligible)
    failure_count = len(eligible) - normal_count
    metrics, details = [], []
    usable = bool(normal_count and failure_count)
    for threshold in THRESHOLDS:
        costs = [first_alarm_cost(item, threshold) for item in eligible]
        details.extend(costs)
        normal = [row for row in costs if row['ground_truth_label'] == 'normal']
        failure = [row for row in costs if row['ground_truth_label'] == 'fail']
        detected = [row['delay_sec'] for row in failure if row['status'] == 'post_onset_detection']
        normal_cost = float(np.mean([row['cost'] for row in normal])) if normal else None
        failure_cost = float(np.mean([row['cost'] for row in failure])) if failure else None
        metrics.append(dict(threshold=threshold, threshold_percent=round(100 * threshold),
            normal_count=normal_count, failure_count=failure_count,
            normal_false_alarm_rate=normal_cost, failure_cost=failure_cost,
            balanced_cost=.5 * (normal_cost + failure_cost) if usable else None,
            post_onset_recall=len(detected)/failure_count if failure_count else None,
            premature_alarm_rate=sum(row['status'] == 'premature_alarm' for row in failure)/failure_count if failure_count else None,
            missed_failure_rate=sum(row['status'] == 'missed_failure' for row in failure)/failure_count if failure_count else None,
            mean_detected_delay_sec=float(np.mean(detected)) if detected else None,
            median_detected_delay_sec=float(np.median(detected)) if detected else None,
            p90_detected_delay_sec=float(np.quantile(detected, .9)) if detected else None,
            is_minimum_cost=False))
    best = min(row['balanced_cost'] for row in metrics) if usable else None
    tied = [row['threshold'] for row in metrics if usable and np.isclose(row['balanced_cost'], best, rtol=0, atol=1e-12)]
    for row in metrics:
        row['is_minimum_cost'] = row['threshold'] in tied
    metrics_path, details_path = root/'threshold_costs.csv', root/'per_bag_costs.csv'
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(details).to_csv(details_path, index=False)
    report = dict(status='created' if usable else 'skipped', selected_threshold=min(tied) if tied else None,
        minimum_cost=best, tied_thresholds=tied, normal_count=normal_count, failure_count=failure_count,
        no_alarm_baseline_cost=.5, improvement_over_no_alarm=.5-best if usable else None,
        excluded_bags=excluded, metrics_csv=str(metrics_path), per_bag_csv=str(details_path),
        formula='0.5 * mean(normal bag alarm costs) + 0.5 * mean(failure bag costs)',
        normal_cost='1 for any alarm, 0 for no alarm.',
        failure_cost='First alarm before onset or no alarm: 1. Otherwise (first_alarm - onset) / (last_scored_time - onset).',
        normalization='Per-bag observed post-onset horizon. Longer recordings penalize the same delay less; this is an explicit assumption, not a universal optimum.',
        timing='Actual observation timestamps in seconds; excludes inference, transport and display latency. Raw delay statistics include post-onset detections only; misses and premature alarms remain full penalties in cost.',
        stopping='Only the first crossing counts. A later correct crossing cannot rescue a premature stop. No annotation-tolerance window is assumed.',
        weighting='Normal and failure classes each receive half the weight, regardless of bag counts. This is a transparent default, not a measured economic cost.',
        tie_rule='Lowest threshold among costs equal within absolute tolerance 1e-12.',
        validation='Choose using held-out validation bags; report final performance on separate test bags.',
        literature=[dict(title='Evaluating Real-time Anomaly Detection Algorithms – NAB', url='https://arxiv.org/abs/1510.03336'),
                    dict(title='Quickest Change Detection', url='https://arxiv.org/abs/1210.5552')],
        implementation_note='Application-specific balanced normalized delay cost; not the official NAB scoring algorithm.')
    plot = root/'threshold_costs.png'
    if usable:
        _plot_costs(metrics, min(tied), plot)
    else:
        from .report_artifacts import unavailable_plot
        unavailable_plot(plot, 'Temporal threshold cost', 'Complete normal histories and failure histories with an onset inside the scored window are required.')
    report['plot_file'] = str(plot)
    write_json(root/'cost_summary.json', report)
    return report


def _plot_costs(metrics, selected, path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    data = pd.DataFrame(metrics)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    specifications = [
        [('balanced_cost', 'Balanced cost'), ('failure_cost', 'Failure cost')],
        [('normal_false_alarm_rate', 'Nominal false alarms'), ('premature_alarm_rate', 'Premature failure-bag stops')],
        [('post_onset_recall', 'Detected after onset'), ('missed_failure_rate', 'No alarm')],
        [('mean_detected_delay_sec', 'Mean delay'), ('p90_detected_delay_sec', '90th percentile delay')]]
    for index, (axis, fields) in enumerate(zip(axes.flat, specifications)):
        for column, label in fields:
            axis.plot(data.threshold_percent, data[column], label=label, linewidth=2)
        if index == 0:
            axis.axhline(.5, color='#777777', linestyle=':', label='No-alarm baseline')
        axis.axvline(100*selected, color='#bd7625', linestyle='--', label=f'Selected {100*selected:g}%')
        axis.set(ylabel='Seconds (post-onset detections only)' if index == 3 else 'Fraction', xlim=(0, 100))
        if index != 3:
            axis.set_ylim(-.02, 1.02)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('Failure-score threshold (%)')
    figure.suptitle('Threshold selection: false alarms and first-alarm delay (lower cost is better)')
    figure.text(.5, .015, 'Misses and premature stops receive full cost. Delay uses each recording’s observed post-onset horizon.', ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .035, 1, .96))
    figure.savefig(path, dpi=160)
    plt.close(figure)
