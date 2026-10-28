"""Keep unavailable reports visible without inventing metric values."""
from pathlib import Path


def unavailable_plot(path, title, reason):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    import textwrap

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.set_axis_off()
    axis.set_title(title, fontsize=14, pad=20)
    axis.text(.5, .6, 'Unavailable for this run', ha='center', fontsize=14, transform=axis.transAxes)
    axis.text(.5, .35, textwrap.fill(reason, 90), ha='center', va='center',
              fontsize=10, transform=axis.transAxes)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_report_index(rows, statistics, pca, root, method):
    root = Path(root)
    lines = ['# Benchmark reports', '',
             'Plots labelled unavailable contain no estimated metric or curve. '
             'ROC/AUROC need both ground-truth classes; prediction alarms do not create a ground-truth failure class.', '',
             'PCA uses the saved final feature vector per demo, including in timeline mode. '
             'Per-demo timeline plots show the individual prefix scores.', '',
             '## Per-demo timelines', '', '| Bag | Input | Status | Plot / reason |', '| --- | --- | --- | --- |']
    for row in rows:
        status = row.get(f'{method}_timeline_status') or 'unavailable'
        path = row.get(f'{method}_timeline_plot')
        detail = row.get(f'{method}_timeline_error') or ''
        if path and Path(path).is_file():
            try:
                target = Path(path).resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                target = Path(path).as_posix()
            detail = f'[{Path(path).name}]({target})'
        lines.append(f"| {row.get('bag_name')} | {row.get('input_mode')} | {status} | {detail} |")
    lines += ['', '## ROC availability', '']
    for name in ('roc', 'probability_roc', 'anomaly_roc'):
        for metric in statistics.get(name, {}).get('metrics', []):
            lines.append(f"- {name} / {metric['input_mode']} / {metric['failure_category']}: "
                         f"{metric['status']}. {metric.get('reason') or ''}")
    sweep = statistics.get('probability_roc', {}).get('threshold_sweep', {})
    if sweep:
        lines += ['', '## Threshold sweep (60-90%, step 1%)', '',
                  '[All threshold metrics](benchmark_threshold_sweep/threshold_metrics.csv) · '
                  '[Best thresholds and ties](benchmark_threshold_sweep/threshold_summary.json)', '',
                  'Best accuracy is measured on these scored bags. Select a threshold on validation data '
                  'and evaluate it on separate test bags; the configured threshold is not changed.']
    if any('threshold_selection_peak_score' in row for row in rows):
        lines += ['', '## Selected threshold and processing time', '',
                  '[Selected threshold](selected_threshold.json) · '
                  '[Average processing time](benchmark_frame_runtime.json) · '
                  '[Per-bag frame timings](benchmark_frame_runtime.csv)', '',
                  'Per-bag timelines and PCA use the chosen replay stopping point. ROC and threshold '
                  'selection use the complete uncensored scan, so stopping at one threshold does not '
                  'hide scores needed to assess higher thresholds.']
    lines += ['', '## PCA availability', '']
    for mode in (pca or {}).get('modes', []):
        lines.append(f"- {mode['input_mode']}: {mode['status']}. {mode.get('reason') or mode.get('note') or ''}")
    lines += ['', '## All plot files', '']
    for path in sorted(root.rglob('*.png')):
        relative = path.relative_to(root).as_posix()
        lines.append(f'- [{relative}]({relative})')
    path = root/'REPORTS.md'
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(path)
