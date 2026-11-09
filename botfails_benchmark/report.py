"""FIDeL's metrics for our detector and cascade on BotFails, plus episode-level results.

FIDeL (Rolland et al., ICRA 2026, Sec. IV-C), per task then averaged over tasks:
- anomaly detection: frame-level AUROC and F1 at the best threshold (anomaly labels);
- thresholding: frame-level TPR, TNR, Acc = (TPR + TNR) / 2 and
  W.Acc = beta * TNR + (1 - beta) * TPR with beta = normal frames / all frames;
- end-to-end: the same four metrics after the semantic filter.
We evaluate at the frames our system scores (2 fps; labels taken at those frames).
Our cascade stops the robot at a confirmed failure, so from the stop on every
frame is predicted 'failure' (latched); before it 'normal'.
FIDeL's end-to-end labels relabel benign anomalies as normal; those failure
labels are not public, so the end-to-end metrics here use the anomaly labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TASKS, FPS

FIDEL = {  # Published BotFails numbers (Table I, project-page table, Sec. IV-D, Fig. 6).
    'ad_auroc_mean': 70.50, 'ad_f1_mean': 62.90, 'ad_auroc_domestic': 70.50, 'ad_f1_domestic': 56.25,
    'ad_auroc_industrial': 70.50, 'ad_f1_industrial': 72.88, 'e2e_wacc_mean': 78.65,
    'e2e_acc_mean_approx': 73.6,  # bar in Fig. 6; abstract: +17.38 points over FAIL-Detect (~56 %).
}
BASELINES_AD = {'logpZ0 (FAIL-Detect)': (65.20, 59.87), 'lopO (FAIL-Detect)': (58.30, 53.48), 'AE': (49.90, 52.12)}
# Anomaly type per test episode range, from test/<task>_anomaly/meta/tasks.jsonl (used only for reporting).
ANOMALY_TYPES = {
    'domotic_dishTidyUp': [(0, 4, 'normal'), (5, 9, 'dishes stored the wrong way'), (10, 14, 'dropping dishes')],
    'domotic_groceriesSorting': [(0, 4, 'normal'), (5, 9, 'items in the wrong box')],
    'domotic_makingCoffee': [(0, 4, 'normal'), (5, 9, 'coffee poured on the table'), (10, 14, 'no container in the machine')],
    'domotic_pouringCoffee': [(0, 4, 'poured incorrectly, a lot on the side'), (5, 9, 'poured next to the cup'),
                              (10, 14, 'full cup spilled'), (15, 19, 'normal')],
    'domotic_setTheTable': [(0, 4, 'normal'), (5, 9, 'bad disposition'), (10, 14, 'dropping items')],
    'domotic_vegetablesAndFruitsSorting': [(0, 4, 'normal'), (5, 14, 'items in the wrong box')],
    'industrial_robothon_buttons': [(0, 9, 'slider not pushed fully / red and blue inverted (mixed)')],
    'industrial_robothon_hatchAndProbe': [(0, 14, 'hatch left open / wrong hole / red hole (mixed)')],
    'industrial_screws_sorting': [(0, 9, 'wrong plate / dropped screws (mixed)')],
    'industrial_soldering': [(0, 4, 'normal'), (5, 9, 'iron put on the wood'), (10, 14, 'hand between iron and board')],
}


def anomaly_type(task, episode):
    index = int(str(episode).split('_')[-1])
    return next((name for start, end, name in ANOMALY_TYPES.get(task, []) if start <= index <= end), 'normal')


# End-to-end BotFails bars of FIDeL's Fig. 6 (Acc, W.Acc), read from the figure: approximate.
BASELINES_E2E = {'FIDeL': (73.6, 78.65), 'FAIL-Detect': (56., 59.), 'VLM only (Qwen2.5)': (50.5, 64.5)}


def domain(task):
    return task.split('_')[0]


def rates(labels, predictions):
    labels, predictions = np.asarray(labels, bool), np.asarray(predictions, bool)
    positives, negatives = labels.sum(), (~labels).sum()
    tpr = (predictions & labels).sum() / positives if positives else np.nan
    tnr = (~predictions & ~labels).sum() / negatives if negatives else np.nan
    beta = negatives / len(labels)
    return dict(TPR=100 * tpr, TNR=100 * tnr, Acc=100 * np.nanmean([tpr, tnr]),
                WAcc=100 * (beta * tnr + (1 - beta) * tpr), beta=beta, frames=len(labels),
                anomalous_frames=int(positives))


def threshold_free(labels, scores):
    from sklearn.metrics import precision_recall_curve, roc_auc_score
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)
    return dict(AUROC=100 * roc_auc_score(labels, scores), F1_opt=100 * f1.max())


def first_alarm(frame, scale):
    """DINO alone with our trigger: the first ALARM_RUN consecutive frames above the threshold stop."""
    from rynnbrain_vlm.cascade import AlarmEpisodes
    tracker = AlarmEpisodes()
    for row in frame.sort_values('timestamp_sec').itertuples(index=False):
        if tracker.update('dino', row.pooled_score > row.base_threshold * scale):
            return float(row.timestamp_sec)
    return None


def latched(frame, stop):
    return frame.timestamp_sec >= stop if stop is not None else pd.Series(False, index=frame.index)


def summarize(table: pd.DataFrame, columns) -> pd.DataFrame:
    rows = [table]
    for name, mask in [('mean (10 tasks)', table.task.isin(TASKS)), ('domestic mean', table.task.str.startswith('domotic')),
                       ('industrial mean', table.task.str.startswith('industrial'))]:
        part = table[mask & table.task.isin(TASKS)]
        if len(part):
            rows.append(pd.DataFrame([dict(task=name, **{c: part[c].mean() for c in columns})]))
    return pd.concat(rows, ignore_index=True)


def episode_outcome(anomalous, onset, stop):
    if not anomalous:
        return 'false_stop' if stop is not None else 'correct_continue'
    if stop is None:
        return 'missed'
    return 'detected' if stop >= onset else 'stopped_before_onset'


def run(args, log):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scale = float(args.threshold_scale)
    fallback_tasks = set()
    if args.cascade_run and (Path(args.cascade_run) / 'run.json').is_file():
        tasks_config = json.loads((Path(args.cascade_run) / 'run.json').read_text())['tasks_config']
        fallback_tasks = {task for task, cfg in tasks_config.items() if cfg.get('fallback') == 'dino'}
    split = json.loads(Path(args.split_file).read_text()) if getattr(args, 'split_file', None) and args.subset else None
    ad_rows, threshold_rows, e2e_rows, episode_rows, check_rows = [], [], [], [], []
    for task in TASKS:
        scores_path = Path(args.dino_run) / task / 'test_scores.csv'
        if not scores_path.is_file():
            log(f'{task}: no DINO test scores, skipped')
            continue
        scores = pd.read_csv(scores_path)
        if split:
            scores = scores[scores.episode.isin([f'episode_{i:06d}' for i in split[args.subset][task]])].reset_index(drop=True)
        scores['ratio'] = scores.pooled_score / scores.base_threshold
        labels = scores.label.astype(int)
        ad_rows.append(dict(task=task, **{f'{k}_raw': v for k, v in threshold_free(labels, scores.pooled_score).items()},
                            **{f'{k}_ratio': v for k, v in threshold_free(labels, scores.ratio).items()}))
        threshold_rows.append(dict(task=task, **rates(labels, scores.ratio > scale)))
        cascade = {}
        if args.cascade_run and (Path(args.cascade_run) / f'{task}.json').is_file():
            cascade = {r['episode']: r for r in json.loads((Path(args.cascade_run) / f'{task}.json').read_text())}
        dino_pred, cascade_pred = pd.Series(False, index=scores.index), pd.Series(False, index=scores.index)
        fallback_pred = pd.Series(False, index=scores.index)
        for episode, frame in scores.groupby('episode'):
            dino_stop = first_alarm(frame, scale)
            dino_pred[frame.index] = latched(frame, dino_stop)
            result = cascade.get(episode)
            cascade_stop = result['stop_sec'] if result else None
            if result:
                cascade_pred[frame.index] = latched(frame, cascade_stop)
                fallback_stop = dino_stop if task in fallback_tasks else cascade_stop
                fallback_pred[frame.index] = latched(frame, fallback_stop)
            full = np.loadtxt(Path(args.dataset) / 'labels' / f'{task}_anomaly' / f'{episode}_labels.csv', ndmin=1).astype(int)
            anomalous = bool(full.any())
            onset = float(np.argmax(full) / FPS) if anomalous else None
            # The labels decide: coffee episode 5 lies in the text's 'poured on the table' range but has no anomaly label.
            kind = anomaly_type(task, episode) if anomalous else 'normal'
            kind = 'anomaly (type not listed)' if anomalous and kind == 'normal' else kind
            row = dict(task=task, episode=episode, anomaly_type=kind, anomalous=anomalous, onset_sec=onset,
                       duration_sec=len(full) / FPS, dino_stop_sec=dino_stop,
                       dino_outcome=episode_outcome(anomalous, onset, dino_stop),
                       dino_delay_sec=(dino_stop - onset) if anomalous and dino_stop is not None and dino_stop >= onset else None)
            if result:
                asked = [a for a in result['alarms'] if a['subgoal_probability'] is not None]
                row.update(cascade_stop_sec=cascade_stop, cascade_outcome=episode_outcome(anomalous, onset, cascade_stop),
                           cascade_delay_sec=(cascade_stop - onset) if anomalous and cascade_stop is not None and cascade_stop >= onset else None,
                           fallback_stop_sec=fallback_stop, fallback_outcome=episode_outcome(anomalous, onset, fallback_stop),
                           fallback_delay_sec=(fallback_stop - onset) if anomalous and fallback_stop is not None and fallback_stop >= onset else None,
                           alarms=len(result['alarms']), checks_asked=len(asked),
                           hold_events=sum(e['event'] == 'hold' for e in result['phases']))
                for alarm in result['alarms']:
                    check_rows.append(dict(task=task, episode=episode, time_sec=alarm['time_sec'], phase=alarm['phase'],
                                           after_onset=bool(anomalous and alarm['time_sec'] >= onset),
                                           verdict=alarm['verdict'], subgoal_probability=alarm['subgoal_probability'],
                                           questions=json.dumps(alarm.get('questions', []))))
            episode_rows.append(row)
        e2e_rows.append(dict(task=task, system='DINO alone (first alarm stops)', **rates(labels, dino_pred)))
        if cascade:
            e2e_rows.append(dict(task=task, system='DINO + VLM check (cascade)', **rates(labels, cascade_pred)))
            e2e_rows.append(dict(task=task, system='DINO + VLM check, DINO alone where no check', **rates(labels, fallback_pred)))
    metric_columns = ['TPR', 'TNR', 'Acc', 'WAcc']
    ad = summarize(pd.DataFrame(ad_rows), ['AUROC_raw', 'F1_opt_raw', 'AUROC_ratio', 'F1_opt_ratio'])
    threshold = summarize(pd.DataFrame(threshold_rows), metric_columns)
    e2e = pd.concat([summarize(part, metric_columns).assign(system=system)
                     for system, part in pd.DataFrame(e2e_rows).groupby('system', sort=False)], ignore_index=True)
    episodes_table = pd.DataFrame(episode_rows)
    for name, table in [('ad_metrics', ad), ('dino_threshold_metrics', threshold), ('end_to_end_metrics', e2e),
                        ('episodes', episodes_table), ('vlm_checks', pd.DataFrame(check_rows))]:
        table.to_csv(output / f'{name}.csv', index=False, float_format='%.4f')
    summary = dict(fidel_reference=FIDEL, ad=ad.round(2).to_dict('records'), threshold=threshold.round(2).to_dict('records'),
                   end_to_end=e2e.round(2).to_dict('records'), episode_outcomes=episode_counts(episodes_table))
    (output / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    write_markdown(output / 'summary.md', ad, threshold, e2e, episodes_table, scale)
    plot_comparison(output / 'comparison_with_fidel.png', ad, e2e)
    if args.cascade_run and not getattr(args, 'no_timelines', False):
        plot_timelines(output / 'timelines', Path(args.dino_run), Path(args.cascade_run), episodes_table, scale)
    log(f'Report written to {output}')
    return summary


def table_markdown(frame, columns, names=None):
    names = names or columns
    lines = ['| ' + ' | '.join(names) + ' |', '|' + '---|' * len(names)]
    for row in frame.itertuples(index=False):
        values = [getattr(row, c) for c in columns]
        lines.append('| ' + ' | '.join(f'{v:.1f}' if isinstance(v, float) else str(v) for v in values) + ' |')
    return '\n'.join(lines)


def write_markdown(path, ad, threshold, e2e, episodes_table, scale):
    mean = lambda t: t[t.task == 'mean (10 tasks)'].iloc[0]  # noqa: E731
    lines = ['# BotFails: our DINO -> VLM cascade vs FIDeL', '',
             'Per-task values, then the mean over the 10 tasks (FIDeL averages per task the same way).', '',
             '## 1. Anomaly detection (frame level, anomaly labels, threshold-free)', '',
             f"FIDeL (Representation): AUROC {FIDEL['ad_auroc_mean']:.2f}, F1@Opt {FIDEL['ad_f1_mean']:.2f}. "
             + ', '.join(f'{k}: AUROC {a:.2f}, F1 {f:.2f}' for k, (a, f) in BASELINES_AD.items()) + '.', '',
             f"Ours, raw pooled DINO score: AUROC {mean(ad).AUROC_raw:.2f}, F1@Opt {mean(ad).F1_opt_raw:.2f}. "
             f"Score / progress threshold: AUROC {mean(ad).AUROC_ratio:.2f}, F1@Opt {mean(ad).F1_opt_ratio:.2f}.", '',
             table_markdown(ad, ['task', 'AUROC_raw', 'F1_opt_raw', 'AUROC_ratio', 'F1_opt_ratio']), '',
             f'## 2. DINO threshold decision per frame (score > {scale:g} x threshold, anomaly labels)', '',
             table_markdown(threshold, ['task', 'TPR', 'TNR', 'Acc', 'WAcc']), '',
             '## 3. End to end (latched stop: every frame from the stop on counts as failure)', '',
             f"FIDeL (Fig. 6, failure labels): Acc ~{FIDEL['e2e_acc_mean_approx']:.1f}, W.Acc {FIDEL['e2e_wacc_mean']:.2f}. "
             'Ours use the anomaly labels (FIDeL\'s failure labels are not public).', '']
    for system, part in e2e.groupby('system', sort=False):
        lines += [f'### {system}', '', table_markdown(part, ['task', 'TPR', 'TNR', 'Acc', 'WAcc']), '']
    lines += ['## 4. Episodes', '', '### Per anomaly type (episodes stopped after the onset / all)', '']
    columns = [c for c in ('dino', 'cascade', 'fallback') if f'{c}_outcome' in episodes_table]
    per_type = []
    for (task, kind), part in episodes_table.groupby(['task', 'anomaly_type'], sort=False):
        row = dict(task=task, anomaly_type=kind, episodes=len(part))
        for c in columns:
            good = 'correct_continue' if kind == 'normal' else 'detected'
            row[c] = f"{(part[f'{c}_outcome'] == good).sum()}/{len(part)}"
        per_type.append(row)
    if per_type:
        frame = pd.DataFrame(per_type)
        lines += ['For normal episodes the count is runs that were not stopped.', '',
                  table_markdown(frame, list(frame.columns)), '']
    for system in ('dino', 'cascade', 'fallback'):
        column = f'{system}_outcome'
        if column in episodes_table:
            counts = episodes_table.groupby('task')[column].value_counts().unstack(fill_value=0)
            counts.loc['total'] = counts.sum()
            delays = episodes_table.groupby('task')[f'{system}_delay_sec'].median()
            delays['total'] = episodes_table[f'{system}_delay_sec'].median()
            counts['median_delay_sec'] = delays.reindex(counts.index)
            counts = counts.reset_index()
            counts.columns = [str(c) for c in counts.columns]
            lines += [f'### {system}', '', table_markdown(counts, list(counts.columns)), '']
    path.write_text('\n'.join(lines))


def plot_comparison(path, ad, e2e):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mean = lambda t: t[t.task == 'mean (10 tasks)']  # noqa: E731
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2))
    ours = mean(ad).iloc[0]
    names = ['FIDeL', *BASELINES_AD, 'Ours: DINO score', 'Ours: DINO score/threshold']
    values = [(FIDEL['ad_auroc_mean'], FIDEL['ad_f1_mean']), *BASELINES_AD.values(),
              (ours.AUROC_raw, ours.F1_opt_raw), (ours.AUROC_ratio, ours.F1_opt_ratio)]
    bars(left, names, values, ('AUROC', 'F1@Opt'), 'Anomaly detection (frame level, mean of 10 tasks)')
    names, values = [f'{k} (Fig. 6, approx.)' for k in BASELINES_E2E], list(BASELINES_E2E.values())
    for system, part in e2e.groupby('system', sort=False):
        row = mean(part).iloc[0]
        names.append(f'Ours: {system}')
        values.append((row.Acc, row.WAcc))
    bars(right, names, values, ('Acc', 'W.Acc'), 'End to end (FIDeL: failure labels; ours: anomaly labels)')
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def bars(axis, names, values, metrics, title):
    positions = np.arange(len(names))
    for k, (metric, color) in enumerate(zip(metrics, ('#2878b5', '#c87925'))):
        heights = [v[k] for v in values]
        axis.barh(positions + (k - .5) * .38, heights, .38, color=color, label=metric)
        for y, h in zip(positions + (k - .5) * .38, heights):
            axis.text(h + .5, y, f'{h:.1f}', va='center', fontsize=7)
    axis.set_yticks(positions, names, fontsize=8)
    axis.invert_yaxis()
    axis.set(xlim=(0, 105), title=title)
    axis.title.set_fontsize(9)
    axis.legend(fontsize=8, loc='lower right')


def plot_timelines(directory, dino_run, cascade_run, episodes_table, scale):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for task in episodes_table.task.unique():
        scores = pd.read_csv(dino_run / task / 'test_scores.csv')
        cascade_path = cascade_run / f'{task}.json'
        results = {r['episode']: r for r in json.loads(cascade_path.read_text())} if cascade_path.is_file() else {}
        (directory / task).mkdir(parents=True, exist_ok=True)
        for row in episodes_table[episodes_table.task == task].itertuples(index=False):
            frame = scores[scores.episode == row.episode].sort_values('timestamp_sec')
            figure, axis = plt.subplots(figsize=(11, 3.2))
            axis.plot(frame.timestamp_sec, frame.pooled_score / frame.base_threshold, color='#2878b5', lw=1.2,
                      label='DINO score / threshold')
            axis.axhline(scale, color='#888', ls='--', lw=.8, label=f'alarm level x{scale:g}')
            if row.anomalous:
                axis.axvspan(row.onset_sec, row.duration_sec, color='#d62728', alpha=.08, label='anomaly (label)')
            result = results.get(row.episode)
            if result:
                events = result['phases']
                for start, end in zip([e for e in events if e['event'] == 'hold'], events[1:] + [None]):
                    stop = end['time_sec'] if end else row.duration_sec
                    axis.axvspan(start['time_sec'], stop, color='#2ca02c', alpha=.06)
                for alarm in result['alarms']:
                    color = {'failure': '#d62728', 'safe': '#2ca02c'}[alarm['verdict']]
                    marker = 'o' if alarm['subgoal_probability'] is not None else 'x'
                    axis.plot(alarm['time_sec'], scale, marker=marker, color=color, ms=7)
                if result['stop_sec'] is not None:
                    axis.axvline(result['stop_sec'], color='#d62728', lw=1.5, label='cascade stop')
            if row.dino_stop_sec == row.dino_stop_sec and row.dino_stop_sec is not None:
                axis.axvline(row.dino_stop_sec, color='#ff7f0e', lw=1, ls=':', label='DINO-alone stop')
            axis.set(xlabel='time (s)', ylabel='score / threshold', xlim=(0, row.duration_sec),
                     title=f'{task} {row.episode}  (green band: gripper holds; o asked: green safe / red failure; x no sub-goal)')
            axis.legend(loc='upper left', fontsize=7, ncol=5)
            figure.tight_layout()
            figure.savefig(directory / task / f'{row.episode}.png', dpi=90)
            plt.close(figure)


def episode_counts(table):
    result = {}
    for system in ('dino', 'cascade', 'fallback'):
        column = f'{system}_outcome'
        if column not in table:
            continue
        counts = table[column].value_counts().to_dict()
        delays = table[f'{system}_delay_sec'].dropna()
        result[system] = dict(counts=counts, median_delay_sec=float(delays.median()) if len(delays) else None)
    return result
