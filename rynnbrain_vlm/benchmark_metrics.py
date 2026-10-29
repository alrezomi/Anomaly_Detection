"""Folder-level VLM/classifier metrics without pooling input modes or abstentions."""
from pathlib import Path
import math

import numpy as np
import pandas as pd


def _wilson(successes, total):
    if not total:
        return None, None
    z = 1.959963984540054
    p = successes/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return center-radius, center+radius


def folder_metrics(rows, folders, input_modes=None):
    timeline = any(row.get('benchmark_decision_mode') == 'timeline' for row in rows)
    data = pd.DataFrame(rows).reindex(columns=['input_mode', 'folder', 'ground_truth_label',
        'decision', 'classifier_decision', 'evaluation_error', 'classifier_error',
        'failure_timing_status', 'failure_time_error_sec', 'runtime_total_bag_sec', 'runtime_model_sec'])
    result = []
    sources = [('vlm', 'decision')]
    # Feature-only intentionally generates no language decision. Do not plot it
    # as a classifier failure or as a zero-accuracy generated-response series.
    decisions = data.decision.dropna()
    if decisions.eq('not_generated').any() and decisions.isin(['not_generated', 'failed', 'not_run']).all():
        sources = []
    if data.classifier_decision.notna().any() or timeline:
        sources.append(('classifier_timeline' if timeline else 'classifier', 'classifier_decision'))
    modes = list(dict.fromkeys([*(input_modes or []), *data.input_mode.dropna().tolist()]))
    for mode in modes:
        mode_rows = data[data.input_mode.eq(mode) | data.input_mode.isna()]
        for source, decision_column in sources:
            for folder in ['ALL', *dict.fromkeys(folders)]:
                group = mode_rows if folder == 'ALL' else mode_rows[mode_rows.folder.eq(folder)]
                known = group[group.ground_truth_label.isin(['normal', 'fail'])]
                error = known.evaluation_error.fillna('').astype(str).ne('')
                if source.startswith('classifier'):
                    error |= known.classifier_error.fillna('').astype(str).ne('')
                valid = known[~error & known[decision_column].isin(['success', 'failure'])]
                positive, predicted = valid.ground_truth_label.eq('fail'), valid[decision_column].eq('failure')
                tp, fp = int((positive & predicted).sum()), int((~positive & predicted).sum())
                fn, tn = int((positive & ~predicted).sum()), int((~positive & ~predicted).sum())
                failures, normals = tp+fn, fp+tn
                total_failures = int(known.ground_truth_label.eq('fail').sum())
                recall_low, recall_high = _wilson(tp, failures)
                fpr_low, fpr_high = _wilson(fp, normals)
                timing = valid[valid.failure_timing_status.eq('compared')] if source.startswith('classifier') else valid.iloc[:0]
                errors = pd.to_numeric(timing.failure_time_error_sec, errors='coerce').dropna()
                errors = errors[np.isfinite(errors)]
                def mean_time(column):
                    values = pd.to_numeric(group[column], errors='coerce').dropna()
                    return float(values.mean()) if len(values) else None
                result.append(dict(input_mode=mode, decision_source=source, folder=folder,
                    availability_reason=("No selected bags in this folder after discovery, exclusions and bag filters."
                        if not len(group) else "No selected bags have known normal/fail ground truth."
                        if not len(known) else "No valid binary decisions; inspect errors and abstentions."
                        if not len(valid) else None),
                    selected_bags=len(group), known_label_count=len(known), unknown_label_count=len(group)-len(known),
                    evaluated_bags=len(valid), error_bags=int(error.sum()),
                    abstained_or_unavailable_bags=len(known)-len(valid)-int(error.sum()),
                    decision_coverage=len(valid)/len(known) if len(known) else None,
                    normal_count=normals, failure_count=failures,
                    true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn,
                    recall=tp/failures if failures else None,
                    false_positive_rate=fp/normals if normals else None,
                    recall_ci95_low=recall_low, recall_ci95_high=recall_high,
                    fpr_ci95_low=fpr_low, fpr_ci95_high=fpr_high,
                    precision=tp/(tp+fp) if tp+fp else None,
                    accuracy=(tp+tn)/len(known) if len(known) else None,
                    accuracy_decided=(tp+tn)/len(valid) if len(valid) else None,
                    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                    balanced_accuracy=(tp/failures+tn/normals)/2 if failures and normals else None,
                    failure_recall_all=tp/total_failures if total_failures else None,
                    timing_compared=len(errors), timing_bias_sec=float(errors.mean()) if len(errors) else None,
                    timing_mae_sec=float(errors.abs().mean()) if len(errors) else None,
                    runtime_total_bag_sec_mean=mean_time('runtime_total_bag_sec'),
                    runtime_model_sec_mean=mean_time('runtime_model_sec')))
    return pd.DataFrame(result)


def write_metrics_report(rows, output, folders, input_modes=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    table = folder_metrics(rows, folders, input_modes)
    csv_path, plot_path = output/'benchmark_metrics.csv', output/'benchmark_metrics.png'
    if table.empty:
        table = table.reindex(columns=['input_mode', 'decision_source', 'folder', 'recall', 'false_positive_rate'])
    table.to_csv(csv_path, index=False)
    categories = list(dict.fromkeys(table.folder))
    panels = [('recall', 'Recall (parsed decisions)'), ('false_positive_rate', 'False-positive rate (parsed decisions)'),
               ('precision', 'Precision'), ('accuracy', 'Accuracy (all known labels)'),
               ('f1', 'F1 (parsed decisions)'), ('decision_coverage', 'Decision coverage')]
    figure, axes = plt.subplots(2, 3, figsize=(max(12, len(categories)*1.8), 9))
    series = list(table.groupby(['decision_source', 'input_mode'], sort=True)) if len(table) else []
    colors = plt.get_cmap('tab10')
    width = .8/max(1, len(series))
    for axis, (column, title) in zip(axes.ravel(), panels):
        for index, ((source, mode), group) in enumerate(series):
            values = group.set_index('folder').reindex(categories)[column].to_numpy(dtype=float)
            x = np.arange(len(categories))+(index-(len(series)-1)/2)*width
            axis.bar(x, values*100, width=width, color=colors(index % 10), label=f'{source} / {mode}')
            for position, value in zip(x, values):
                if not np.isfinite(value):
                    axis.text(position, 2, 'N/A', rotation=90, ha='center', fontsize=7, color=colors(index % 10))
        if not series:
            axis.text(.5, .5, 'No completed input-mode results.', ha='center', transform=axis.transAxes)
        axis.set(title=title, ylabel='Percent', ylim=(0, 110), xlim=(-.6, max(.6, len(categories)-.4)))
        axis.set_xticks(range(len(categories)), categories, rotation=30, ha='right', fontsize=8)
        axis.grid(axis='y', alpha=.2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .96), ncol=min(4, len(handles)), fontsize=8)
    timeline = any(row.get('benchmark_decision_mode') == 'timeline' for row in rows)
    figure.suptitle('VLM benchmark: offline timeline alarms (any prefix crossing)' if timeline else
                   'VLM benchmark: all folders, separate generated and classifier decisions')
    figure.tight_layout(rect=(0, 0, 1, .93))
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    note = ('Recall/FPR/precision/F1 use parsed binary decisions only. Coverage shows how many known-labelled '
            'cases were decided. Accuracy includes unavailable/uncertain/error cases as incorrect. '
            'failure_recall_all uses all known failure labels. Unknown labels are excluded. '
            'Timing fields describe the existing classifier timeline only, not generated VLM decisions. '
            'Input modes and generated/classifier decisions are never pooled.')
    if timeline:
        note += (' Classifier decisions use any sampled prefix score >= its saved threshold, including the final prefix. '
                 'No DINO consecutive-alarm or stage cutoff policy is applied. Complete no-alert timelines are success; '
                 'incomplete no-alert timelines are unavailable. Observed crossings on partial timelines still prove failure. '
                 'Whole-demo scores are retained in whole_demo_classifier_* columns. '
                 'Peak prefix scores rank bags for ROC; they are not calibrated probabilities of any bag-level failure.')
    (output/'benchmark_metrics_notes.txt').write_text(note+'\n', encoding='utf-8')
    print(f'Folder metrics: {csv_path}; graph: {plot_path}')
    return dict(csv=str(csv_path), plot=str(plot_path), rows=len(table), definition=note)
