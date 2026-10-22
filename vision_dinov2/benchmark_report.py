"""Bag-level DINO decisions, scale sweeps, statistics and timing figures.

Failure is positive. All decisions use strict score > base_threshold * scale,
including ties and zero thresholds. Scale selection is exploratory on the
selected benchmark, never an automatic config update or independent validation.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from rynnbrain_vlm.benchmark_roc import roc_curve


def bag_score(frames):
    scores = frames.pooled_score.to_numpy(float)
    bounds = frames.base_threshold.to_numpy(float)
    if (not len(scores) or not np.isfinite(scores).all() or (scores < 0).any()
            or not np.isfinite(bounds).all() or (bounds < 0).any()):
        raise ValueError('Invalid or empty pooled score timeline.')
    ratios = np.divide(scores, bounds, out=np.zeros_like(scores), where=bounds > 0)
    ratios[(bounds == 0) & (scores > 0)] = np.inf
    return float(ratios.max())


def decisions(manifest, frames, scale, operating_point):
    """Any crossing across sampled frames/cameras is failure; incomplete bags aren't successes."""
    rows = []
    grouped = {key: group for key, group in frames.groupby(['metric', 'bag_path'])} if len(frames) else {}
    for metric in ('euclidean', 'mahalanobis'):
        for item in manifest:
            row = dict(item, metric=metric, operating_point=operating_point, threshold_scale=float(scale),
                       predicted_decision=None, correct=None, first_alarm_time_sec=None,
                       failure_time_error_sec=None, timing_status='not_scored', max_score_ratio=None)
            timeline = grouped.get((metric, item['bag_path']))
            if item['status'] != 'ok' or timeline is None or timeline.empty:
                rows.append(row)
                continue
            alarms = timeline.pooled_score > timeline.base_threshold * scale
            first = float(timeline.loc[alarms, 'timestamp_sec'].min()) if alarms.any() else None
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
            if label != 'fail':
                row['timing_status'] = 'false_alarm' if label == 'normal' and first is not None else 'not_failure'
            elif onset is None or not np.isfinite(onset):
                row['timing_status'] = 'missing_failure_marker'
            elif not timeline.timestamp_sec.min() <= onset <= timeline.timestamp_sec.max():
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
    if valid.empty or valid.ground_truth_label.nunique() != 2:
        return dict(status='unavailable', reason='ROC needs both evaluated normal and failure bags.'), pd.DataFrame(), None
    scores = valid.max_score_ratio.to_numpy(float)
    targets = valid.ground_truth_label.eq('fail').to_numpy(int)
    # Dense ranks preserve ties and ordering, including infinite ratios from a
    # positive score with zero threshold. AUC is the standard ranking AUC.
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
        timelines = {path: (group.pooled_score.to_numpy(float), group.base_threshold.to_numpy(float))
                     for path, group in frames.groupby('bag_path')}
    for scale in sorted(candidates):
        if timelines is None:
            predictions = scores > scale
        else:
            with np.errstate(over='ignore', under='ignore'):
                predictions = np.array([bool((timelines[path][0] > timelines[path][1] * scale).any())
                                        for path in valid.bag_path])
        tp = int((predictions & (targets == 1)).sum())
        fp = int((predictions & (targets == 0)).sum())
        recall, false_positive = tp / targets.sum(), fp / (len(targets)-targets.sum())
        points.append(dict(threshold_scale=scale, recall=float(recall), false_positive_rate=float(false_positive),
                           youden_j=float(recall-false_positive)))
    table = pd.DataFrame(points)
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
    configured = decisions(manifest, frames, scale, 'configured')
    results = [configured]
    roc_reports = {}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    for metric, axis in zip(('euclidean', 'mahalanobis'), axes.ravel()):
        selected = configured[configured.metric == metric]
        report, sweep, curve = scale_sweep(selected, scale, frames[frames.metric.eq(metric)] if len(frames) else None)
        roc_reports[metric] = report
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
            axis.text(.5, .5, report['reason'], ha='center', wrap=True)
        axis.plot([0, 1], [0, 1], '--', color='gray', alpha=.5)
        axis.set(title=f'{metric.title()} - attention + 3x3 pooling', xlabel='False-positive rate',
                 ylabel='Recall / true-positive rate', xlim=(-.02, 1.02), ylim=(-.02, 1.02))
        axis.grid(alpha=.2)
    figure.suptitle('Bag-level ROC: any sampled pooled-score crossing is failure')
    figure.tight_layout()
    figure.savefig(output / 'roc.png', dpi=160)
    plt.close(figure)
    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(output / 'bag_results.csv', index=False)
    metrics = metrics_table(all_results, folders)
    metrics.to_csv(output / 'metrics.csv', index=False)
    (output / 'roc_summary.json').write_text(json.dumps(roc_reports, indent=2, allow_nan=False), encoding='utf-8')
    frames = frames.copy()
    if len(frames):
        frames['configured_threshold'] = frames.base_threshold * scale
        frames['configured_alarm'] = frames.pooled_score > frames.configured_threshold
        for metric, report in roc_reports.items():
            mask = frames.metric.eq(metric)
            if report['status'] == 'created':
                best = report['best']['threshold_scale']
                frames.loc[mask, 'best_threshold'] = frames.loc[mask, 'base_threshold'] * best
        if 'best_threshold' in frames:
            frames['best_alarm'] = (frames.pooled_score > frames.best_threshold).astype('boolean').where(frames.best_threshold.notna())
    frames.to_csv(output / 'frame_results.csv', index=False)
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
    return metrics, roc_reports


def _timing_plots(results, output, plt):
    for metric in ('euclidean', 'mahalanobis'):
        failures = results[results.metric.eq(metric) & results.ground_truth_label.eq('fail')]
        bags = sorted(failures.bag_path.unique())
        if not bags:
            continue
        # Keep missed failures visible and paginate large benchmarks.
        for page, start in enumerate(range(0, len(bags), 25), 1):
            names = bags[start:start+25]
            figure, axes = plt.subplots(1, 2, figsize=(15, max(4, len(names)*.35)), sharey=True)
            for index, bag in enumerate(names):
                group = failures[failures.bag_path.eq(bag)]
                onset = group.recorded_failure_time_sec.iloc[0]
                if pd.notna(onset):
                    axes[0].scatter(onset, index, marker='|', color='black', s=100)
                for point, color, offset in [('configured', '#2878b5', -.18), ('exploratory_best', '#c87925', .18)]:
                    selected = group[group.operating_point.eq(point)]
                    if selected.empty:
                        continue
                    row = selected.iloc[0]
                    if pd.notna(row.first_alarm_time_sec):
                        axes[0].scatter(row.first_alarm_time_sec, index+offset, color=color, s=22)
                    if pd.notna(row.failure_time_error_sec):
                        axes[1].scatter(row.failure_time_error_sec, index+offset, color=color, s=22)
                    else:
                        axes[1].annotate(row.timing_status.replace('_', ' '), (0, index+offset),
                                         fontsize=7, color=color)
            axes[0].set_yticks(range(len(names)), ['/'.join(filter(None, [Path(bag).parent.name, Path(bag).name]))
                                                   for bag in names], fontsize=8)
            axes[0].set_ylim(len(names)-.5, -.5)
            axes[0].set(xlabel='Bag-relative observation time (s)', title='Black: actual marker; blue: configured; orange: exploratory best')
            axes[1].axvline(0, color='black', linewidth=1)
            axes[1].set(xlabel='First alarm - recorded failure time (s)', title='Negative = early alarm; positive = late alarm')
            for axis in axes:
                axis.grid(alpha=.2)
            figure.suptitle(f'{metric.title()} - attention + 3x3 pooling')
            figure.tight_layout()
            figure.savefig(output / f'{metric}_failure_timing_{page:02d}.png', dpi=160)
            plt.close(figure)
