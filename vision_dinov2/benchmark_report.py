"""Bag-level DINO decisions, scale sweeps, statistics and timing figures.

Failure is positive. Alarms require two consecutive crossings before stage 3.
All decisions use strict score > base_threshold * scale,
including ties and zero thresholds. Scale selection is exploratory on the
selected benchmark, never an automatic config update or independent validation.
"""
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd

from rynnbrain_vlm.benchmark_roc import roc_curve
from .benchmark_alarm import eligible_frames, confirmed_alarms, confirmation_pairs, ALARM_POLICY

REPORT_FORMAT_VERSION = 4

def bag_score(frames):
    scores = frames.pooled_score.to_numpy(float)
    bounds = frames.base_threshold.to_numpy(float)
    if (not len(scores) or not np.isfinite(scores).all() or (scores < 0).any()
            or not np.isfinite(bounds).all() or (bounds < 0).any()):
        raise ValueError('Invalid or empty pooled score timeline.')
    ratios = np.divide(scores, bounds, out=np.zeros_like(scores), where=bounds > 0)
    ratios[(bounds == 0) & (scores > 0)] = np.inf
    first, second = confirmation_pairs(frames)
    return float(np.minimum(ratios[first], ratios[second]).max()) if len(first) else 0.


def decisions(manifest, frames, scale, operating_point):
    """Confirm on the second consecutive eligible crossing in the same camera."""
    frames = eligible_frames(manifest, frames)
    rows = []
    grouped = {key: group for key, group in frames.groupby(['metric', 'bag_path'])} if len(frames) else {}
    for metric in ('euclidean', 'mahalanobis'):
        for item in manifest:
            row = dict(item, metric=metric, operating_point=operating_point, threshold_scale=float(scale),
                       alarm_policy=ALARM_POLICY,
                       predicted_decision=None, correct=None, first_alarm_time_sec=None,
                       failure_time_error_sec=None, timing_status='not_scored', max_score_ratio=None)
            timeline = grouped.get((metric, item['bag_path']))
            if item['status'] != 'ok' or timeline is None or timeline.empty:
                rows.append(row)
                continue
            alarms = confirmed_alarms(timeline, scale)
            first = float(timeline.iloc[np.flatnonzero(alarms)].timestamp_sec.min()) if alarms.any() else None
            predicted = 'failure' if first is not None else 'success'
            row.update(predicted_decision=predicted, first_alarm_time_sec=first,
                       max_score_ratio=bag_score(timeline), frames_scored=len(timeline),
                       feature_sec_total=float(timeline.feature_sec.sum()),
                       comparison_sec_total=float(timeline.comparison_sec.sum()),
                       metric_score_sec_total=float(timeline.metric_score_sec.sum()),
                       processing_sec_mean=float(timeline.processing_sec.mean()),
                       processing_sec_p95=float(timeline.processing_sec.quantile(.95)))
            label = item['ground_truth_label']
            if label in {'normal', 'fail'}:
                row['correct'] = (predicted == 'failure') == (label == 'fail')
            onset = item.get('recorded_failure_time_sec')
            row['label_onset_conflict'] = label=='normal' and onset is not None and np.isfinite(onset)
            if onset is None or not np.isfinite(onset):
                row['timing_status'] = ('missing_failure_marker' if label=='fail' else
                    'false_alarm' if first is not None else 'not_failure')
            elif onset >= float(item['stage3_start_sec']):
                row['timing_status'] = 'annotation_after_alarm_window'
            elif not float(item.get('stage1_start_sec') or 0.) <= onset <= timeline.timestamp_sec.max():
                row['timing_status'] = 'annotation_outside_scored_window'
            elif first is None:
                row['timing_status'] = 'missed_failure'
            else:
                row.update(timing_status='compared', failure_time_error_sec=first - onset)
            rows.append(row)
    return pd.DataFrame(rows)


def wilson(successes, count):
    if not count:
        return None, None
    z = 1.959963984540054
    rate = successes / count
    center = (rate + z*z / (2*count)) / (1 + z*z/count)
    radius = z * np.sqrt(rate*(1-rate)/count + z*z/(4*count*count)) / (1 + z*z/count)
    return float(center - radius), float(center + radius)


def metrics_table(results, folders):
    rows = []
    for (metric, point), group in results.groupby(['metric', 'operating_point']):
        for folder in ['ALL', *folders]:
            selected = group if folder == 'ALL' else group[group.folder == folder]
            valid = selected[(selected.status == 'ok') & selected.ground_truth_label.isin(['normal', 'fail'])
                             & selected.predicted_decision.notna()]
            positive, predicted = valid.ground_truth_label.eq('fail'), valid.predicted_decision.eq('failure')
            tp, fp = int((positive & predicted).sum()), int((~positive & predicted).sum())
            fn, tn = int((positive & ~predicted).sum()), int((~positive & ~predicted).sum())
            n, p = tn + fp, tp + fn
            recall, fpr = tp / p if p else None, fp / n if n else None
            errors = valid.loc[valid.timing_status.eq('compared'), 'failure_time_error_sec'].astype(float)
            recall_low, recall_high = wilson(tp, p)
            fpr_low, fpr_high = wilson(fp, n)
            rows.append(dict(metric=metric, operating_point=point, folder=folder,
                threshold_scale=float(group.threshold_scale.iloc[0]), selected_bags=len(selected),
                excluded_or_failed_bags=len(selected)-len(valid), evaluated_bags=len(valid),
                excluded_training_bags=int(selected.status.eq('excluded_training').sum()),
                error_bags=int(selected.status.eq('error').sum()),
                alarm_bags=int(predicted.sum()), no_alarm_bags=int((~predicted).sum()),
                alarm_rate=float(predicted.mean()) if len(valid) else None,
                normal_count=n, failure_count=p, true_positive=tp, false_positive=fp,
                true_negative=tn, false_negative=fn, recall=recall, false_positive_rate=fpr,
                recall_ci95_low=recall_low, recall_ci95_high=recall_high,
                fpr_ci95_low=fpr_low, fpr_ci95_high=fpr_high,
                precision=tp/(tp+fp) if tp+fp else None,
                accuracy=(tp+tn)/(p+n) if p+n else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                balanced_accuracy=(recall+1-fpr)/2 if p and n else None,
                timing_compared=len(errors), missed_failures=fn,
                early_alert_count=int((errors < 0).sum()),
                timing_bias_sec=float(errors.mean()) if len(errors) else None,
                timing_median_sec=float(errors.median()) if len(errors) else None,
                timing_mae_sec=float(errors.abs().mean()) if len(errors) else None,
                processing_sec_mean=float(valid.processing_sec_mean.mean()) if len(valid) else None))
    return pd.DataFrame(rows)


def scale_sweep(results, configured_scale, frames=None):
    """One point per distinct bag score, using valid positive scales and exact > decisions."""
    valid = results[(results.status == 'ok') & results.ground_truth_label.isin(['normal', 'fail'])
                    & results.predicted_decision.notna()]
    if valid.empty:
        return dict(status='unavailable', reason='ROC needs both evaluated normal and failure bags.'), pd.DataFrame(), None
    scores = valid.max_score_ratio.to_numpy(float)
    targets = valid.ground_truth_label.eq('fail').to_numpy(int)
    # Dense ranks preserve ties and ordering, including infinite ratios from a
    # positive score with zero threshold. AUC is the standard ranking AUC.
    both_classes = valid.ground_truth_label.nunique() == 2
    if both_classes:
        ranks = pd.Series(scores).rank(method='dense').to_numpy(float)
        fpr, tpr, _, area = roc_curve(targets, ranks)
    unique = np.unique(scores[np.isfinite(scores) & (scores > 0)])
    candidates = {float(configured_scale), 1., float(np.nextafter(0., 1.)), float(np.finfo(float).max)}
    representative = {float(configured_scale), 1.}
    if len(unique):
        low = float(unique[0] / 2)
        high = float(min(unique[-1], np.finfo(float).max / 1.1) * 1.1)
        representative.update(value for value in [low, high] if value > 0 and np.isfinite(value))
        representative.update(float(np.exp((np.log(a)+np.log(b))/2)) for a, b in zip(unique, unique[1:]))
        candidates.update(representative)
    for value in unique:
        candidates.add(float(value))  # equality means no alarm
        lower = float(np.nextafter(value, 0.))
        if lower > 0:
            candidates.add(lower)
    points = []
    timelines = None
    if frames is not None:
        frames = eligible_frames(results.to_dict('records'), frames)
        timelines = {}
        for path, group in frames.groupby('bag_path'):
            left, right = confirmation_pairs(group)
            values, bounds = group.pooled_score.to_numpy(float), group.base_threshold.to_numpy(float)
            timelines[path] = values[left], bounds[left], values[right], bounds[right]
    for scale in sorted(candidates):
        if timelines is None:
            predictions = scores > scale
        else:
            with np.errstate(over='ignore', under='ignore'):
                predictions = np.array([bool(((timelines[path][0] > timelines[path][1] * scale)
                                             & (timelines[path][2] > timelines[path][3] * scale)).any())
                                        for path in valid.bag_path])
        tp = int((predictions & (targets == 1)).sum())
        fp = int((predictions & (targets == 0)).sum())
        recall = float(tp / targets.sum()) if targets.sum() else None
        false_positive = float(fp / (len(targets)-targets.sum())) if (targets == 0).any() else None
        points.append(dict(threshold_scale=scale, recall=recall, false_positive_rate=false_positive,
                           alarm_rate=float(predictions.mean()),
                           youden_j=recall-false_positive if both_classes else None))
    table = pd.DataFrame(points)
    if not both_classes:
        return dict(status='unavailable', reason='ROC and best scale are undefined with only one ground-truth class.',
                    normal_count=int((targets == 0).sum()), failure_count=int(targets.sum())), table, None
    # Prefer a usable interior/configured value over a nextafter boundary, then
    # the value closest to the configured scale when performance is identical.
    table['boundary_only'] = ~table.threshold_scale.isin(representative)
    table['log_distance_from_configured'] = np.abs(np.log(table.threshold_scale) - np.log(configured_scale))
    best = table.sort_values(['youden_j', 'false_positive_rate', 'boundary_only', 'log_distance_from_configured', 'threshold_scale'],
                              ascending=[False, True, True, True, True], kind='stable').iloc[0].to_dict()
    report = dict(status='created', auc=area, best=best,
                  selection='maximum Youden J; same selected bags used for tuning and reporting',
                  zero_threshold_always_alarm_bags=int(np.isinf(scores).sum()),
                  normal_count=int((targets == 0).sum()), failure_count=int(targets.sum()))
    return report, table, (fpr, tpr)


def write_reports(manifest, frames, output, scale, folders):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    if not len(frames):
        frames = frames.reindex(columns=['metric', 'bag_path', 'camera_topic', 'timestamp_sec',
            'pooled_score', 'base_threshold', 'feature_sec', 'comparison_sec', 'metric_score_sec',
            'processing_sec', 'read_decode_sec'])
    frames = eligible_frames(manifest, frames)
    configured = decisions(manifest, frames, scale, 'configured')
    results = [configured]
    roc_reports = {}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    for metric, axis in zip(('euclidean', 'mahalanobis'), axes.ravel()):
        selected = configured[configured.metric == metric]
        report, sweep, curve = scale_sweep(selected, scale, frames[frames.metric.eq(metric)] if len(frames) else None)
        roc_reports[metric] = report
        if len(sweep):
            sweep.to_csv(output / f'{metric}_scale_sweep.csv', index=False)
        if curve is not None:
            best = report['best']
            axis.plot(*curve, label=f"ROC; AUC={report['auc']:.3f}")
            current = sweep.loc[sweep.threshold_scale.eq(scale)].iloc[0]
            axis.scatter(current.false_positive_rate, current.recall, marker='o', label=f'Configured scale {scale:g}')
            axis.scatter(best['false_positive_rate'], best['recall'], marker='*', s=150,
                         label=f"Exploratory best scale {best['threshold_scale']:.6g}")
            sweep.to_csv(output / f'{metric}_scale_sweep.csv', index=False)
            pd.DataFrame(dict(false_positive_rate=curve[0], recall=curve[1])).to_csv(output / f'{metric}_roc.csv', index=False)
            extra = decisions(manifest, frames, best['threshold_scale'], 'exploratory_best')
            results.append(extra[extra.metric.eq(metric)])
            axis.legend(fontsize=8)
        else:
            axis.text(.5, .5, f"ROC unavailable\nNormal labels: {report.get('normal_count', 0)}; failure labels: {report.get('failure_count', 0)}\n"
                      'See alarm_rate_vs_scale.png in the metric folder.\nTask success is not a visual-anomaly label.',
                      ha='center', va='center', transform=axis.transAxes, fontsize=9)
            axis.set_axis_off()
        if curve is not None:
            axis.plot([0, 1], [0, 1], '--', color='gray', alpha=.5)
        axis.set(title=f'{metric.title()} - attention + 3x3 pooling', xlabel='False-positive rate',
                 ylabel='Recall / true-positive rate', xlim=(-.02, 1.02), ylim=(-.02, 1.02))
        axis.grid(alpha=.2)
    figure.suptitle('Bag-level ROC: two consecutive crossings in one camera, before stage 3')
    figure.tight_layout()
    if any(report['status']=='created' for report in roc_reports.values()):
        figure.savefig(output / 'roc.png', dpi=160)
    plt.close(figure)
    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(output / 'bag_results.csv', index=False)
    audit_columns = ['metric', 'bag_path', 'bag_name', 'folder', 'ground_truth_label', 'onset_source',
        'stage1_start_sec', 'stage3_start_sec', 'failure_marker', 'recorded_failure_time_sec',
        'first_alarm_time_sec', 'failure_time_error_sec', 'timing_status', 'label_onset_conflict',
        'annotation_status', 'annotation_error', 'stage_annotation_refresh_error',
        'stage_message_count', 'ignored_pre_stage1_failure_count', 'recorded_stage_messages']
    configured.reindex(columns=audit_columns).to_csv(output/'timing_diagnostics.csv', index=False)
    metrics = metrics_table(all_results, folders)
    metrics.to_csv(output / 'metrics.csv', index=False)
    (output / 'roc_summary.json').write_text(json.dumps(roc_reports, indent=2, allow_nan=False), encoding='utf-8')
    frames = frames.copy()
    if len(frames):
        frames['configured_threshold'] = frames.base_threshold * scale
        frames['configured_crossing'] = frames.pooled_score > frames.configured_threshold
        frames['configured_alarm'] = confirmed_alarms(frames, scale)
        for metric, report in roc_reports.items():
            mask = frames.metric.eq(metric)
            if report['status'] == 'created':
                best = report['best']['threshold_scale']
                frames.loc[mask, 'best_threshold'] = frames.loc[mask, 'base_threshold'] * best
                frames.loc[mask, 'best_alarm'] = confirmed_alarms(frames.loc[mask], best)
        if 'best_threshold' in frames:
            frames['best_alarm'] = frames.best_alarm.astype('boolean').where(frames.best_threshold.notna())
    frames.to_csv(output / 'frame_results.csv', index=False)
    diagnostics = []
    truth = {row['bag_path']: row for row in manifest}
    if len(frames):
        for (metric, bag), timeline in frames.groupby(['metric','bag_path']):
            item = truth[bag]
            if item['status'] != 'ok':
                continue
            ordered = timeline.sort_values(['timestamp_sec','camera_topic'], kind='stable')
            first = ordered.iloc[0]
            alarms = ordered.configured_alarm
            first_alarm = ordered.loc[alarms].iloc[0] if alarms.any() else None
            diagnostics.append(dict(metric=metric, bag_path=bag, folder=item['folder'],
                ground_truth_label=item['ground_truth_label'], frames_scored=len(ordered),
                alarm_frame_count=int(alarms.sum()), alarm_frame_fraction=float(alarms.mean()),
                first_scored_time_sec=float(first.timestamp_sec), first_score=float(first.pooled_score),
                first_threshold=float(first.configured_threshold), first_frame_alarm=bool(first.configured_alarm),
                first_alarm_time_sec=float(first_alarm.timestamp_sec) if first_alarm is not None else None,
                first_alarm_camera=first_alarm.camera_topic if first_alarm is not None else None,
                max_score_ratio=bag_score(ordered)))
    pd.DataFrame(diagnostics, columns=['metric', 'bag_path', 'folder', 'ground_truth_label', 'frames_scored',
        'alarm_frame_count', 'alarm_frame_fraction', 'first_scored_time_sec', 'first_score', 'first_threshold',
        'first_frame_alarm', 'first_alarm_time_sec', 'first_alarm_camera', 'max_score_ratio']).to_csv(output/'alarm_diagnostics.csv', index=False)
    timing_rows = []
    good_bags = {item['bag_path'] for item in manifest if item['status'] == 'ok'}
    if len(frames):
        for metric in ('euclidean', 'mahalanobis'):
            for folder in ['ALL', *folders]:
                bags = good_bags if folder == 'ALL' else {item['bag_path'] for item in manifest
                                                          if item['folder'] == folder and item['status'] == 'ok'}
                group = frames[frames.metric.eq(metric) & frames.bag_path.isin(bags)]
                row = dict(metric=metric, folder=folder, images_scored=len(group))
                for name in ['feature_sec', 'comparison_sec', 'metric_score_sec', 'processing_sec', 'read_decode_sec']:
                    values = group[name]
                    row.update({name+'_mean': float(values.mean()) if len(values) else None,
                                name+'_median': float(values.median()) if len(values) else None,
                                name+'_p95': float(values.quantile(.95)) if len(values) else None})
                timing_rows.append(row)
    timing_data = pd.DataFrame(timing_rows) if timing_rows else pd.DataFrame(columns=['metric','folder','images_scored'])
    timing_data.to_csv(output / 'runtime_summary.csv', index=False)
    _timing_plots(all_results, output, plt)
    from .benchmark_plots import write_metric_reports
    write_metric_reports(manifest, frames, all_results, metrics, roc_reports, output, scale, plt)
    artifact = dict(report_format_version=REPORT_FORMAT_VERSION, alarm_policy=ALARM_POLICY,
                    writer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    files=sorted(str(path.relative_to(output)).replace('\\', '/') for path in output.rglob('*') if path.is_file()))
    (output/'report_manifest.json').write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(f'DINO report format v{REPORT_FORMAT_VERSION}: {output}/SUMMARY.md; metric folders: euclidean/, mahalanobis/.', flush=True)
    return metrics, roc_reports


def _timing_plots(results, output, plt):
    selected = results[results.operating_point.eq('configured') & ~results.status.eq('excluded_training')]
    has_onsets = selected.loc[selected.status.eq('ok'), 'recorded_failure_time_sec'].notna().any()
    if has_onsets:
        selected = selected[selected.recorded_failure_time_sec.notna() | selected.ground_truth_label.eq('fail')]
    bags = sorted(selected.bag_path.unique())
    figure, axis = plt.subplots(figsize=(12, max(4, len(bags)*.36)))
    labels = []
    for index, bag in enumerate(bags):
        group = selected[selected.bag_path.eq(bag)]
        labels.append(f'{group.iloc[0].folder}/{group.iloc[0].bag_name}')
        for metric, color, offset in [('euclidean', '#2878b5', -.13), ('mahalanobis', '#c87925', .13)]:
            row = group[group.metric.eq(metric)].iloc[0]
            value = row.failure_time_error_sec if has_onsets else row.first_alarm_time_sec
            if pd.notna(value):
                axis.scatter(value, index+offset, color=color, s=25)
            else:
                reason = row.timing_status.replace('_', ' ')
                if row.status != 'ok':
                    reason = 'not evaluated: '+row.status
                elif not has_onsets:
                    reason = 'no confirmed alarm'
                elif pd.isna(row.recorded_failure_time_sec):
                    reason = 'no recorded onset'
                axis.annotate(reason, (0, index+offset), xytext=(5, 0), textcoords='offset points', fontsize=7, color=color)
    for metric, color in [('Euclidean', '#2878b5'), ('Mahalanobis', '#c87925')]:
        axis.scatter([], [], color=color, label=metric)
    if not bags:
        axis.text(.5, .5, 'No evaluated demos.', ha='center', transform=axis.transAxes)
    axis.axvline(0, color='black', linewidth=1)
    axis.set_yticks(range(len(bags)), labels, fontsize=8)
    axis.set_ylim(len(bags)-.6, -.6)
    axis.set(xlabel='Confirmed alarm time - recorded failure onset (s)' if has_onsets else 'First confirmed alarm time from bag start (s)',
             title='All folders: signed detection delay (negative = early, positive = late)' if has_onsets
                   else 'All folders: first confirmed alarm (no recorded onset available for a delay comparison)')
    axis.legend(loc='best')
    axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output/('failure_time_difference.png' if has_onsets else 'first_alarm_time.png'), dpi=160)
    plt.close(figure)
