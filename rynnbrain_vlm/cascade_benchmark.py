"""Offline benchmark of the DINO -> VLM check on a saved DINO benchmark run.

DINO is not run again. Its saved per-frame crossings (pooled score above the
configured threshold, the DINO benchmark's decision score) are replayed with
the live rule: ALARM_RUN consecutive crossings in one camera, until
POST_STAGE3_SEC after stage 3 on the causal stage clock. Each new alarm
episode asks the VLM exactly as the live replay does (VlmChecker), with the
camera frames the live replay would have buffered. A failure answer is the
bag's alarm; a safe answer lets the execution continue.

Two setups are reported on the same bags; the VLM's reference executions
are left out of both: DINO alone (the saved run's own decision) and DINO with
the VLM check.

Results go to <dino run>/vlm_check/. VLM answers are cached in checks.jsonl,
keyed by bag, alarm and the exact question, so a rerun resumes where it
stopped, both DINO metrics share identical checks, and reports can be
regenerated without loading the VLM.
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .cascade import (ALARM_RUN, CONTEXT_SEC, MAX_VLM_FRAMES, MIN_FRAME_SPACING_SEC,
                      AlarmEpisodes, VlmChecker, current_stage)

SAVED, CHECKED = 'dino_alone', 'dino_vlm_check'
SETUPS = (SAVED, CHECKED)
COLORS = {SAVED: '#2878b5', CHECKED: '#c87925'}
LABELS = {SAVED: 'DINO alone', CHECKED: 'DINO + VLM check'}


def stage_boundaries(bag, stage_topic):
    """Causal stage clock, exactly as the live replay advances it."""
    from rosbag_io import read_stage_events
    from time_series_gmr_scripts.online_gmr import StageClock
    clock = StageClock()
    for row in read_stage_events(bag, stage_topic).itertuples(index=False):
        clock.observe(float(row.time), str(row.stage).strip())
    return dict(clock.boundaries)


def alarm_episodes(timeline, boundaries, scale, post_sec):
    """Frames where the live rule starts a new alarm episode, within the execution window."""
    end = boundaries[3] + post_sec if 3 in boundaries else np.inf
    rule, episodes = AlarmEpisodes(), []
    for row in timeline.sort_values('timestamp_sec', kind='stable').itertuples(index=False):
        t = float(row.timestamp_sec)
        if t > end:
            break
        threshold = float(row.base_threshold) * scale
        if rule.update(row.camera_topic, bool(row.pooled_score > threshold)):
            episodes.append(dict(time_sec=t, camera_topic=row.camera_topic, stage=current_stage(boundaries, t),
                                 dino_score=float(row.pooled_score), dino_threshold=threshold))
    return episodes


def camera_frames(bag, stage_topic, topic, until_sec):
    """Camera messages buffered as in the live replay (same 15 fps sampling), up to until_sec."""
    from rosbag_io import iter_ts_replay_events
    frames, events = [], iter_ts_replay_events(bag, {}, stage_topic, topic)
    try:
        for timestamp, kind, payload in events:
            if timestamp > until_sec + 1e-9:
                break
            if kind == 'camera':
                frames.append((timestamp, ('message', payload)))
    finally:
        events.close()
    return frames


def question_identity(config):
    """Everything that determines the VLM's answer, without loading the model."""
    from time_series_gmr_scripts import run_time_series_test as settings
    vlm = config['rynnbrain']
    return dict(model=vlm['model']['model_id'], reference_bags=list(vlm.get('reference_bags', [])),
                num_frames=int(vlm.get('num_frames', 8)), prompt_variant=vlm.get('prompt_variant', 'A'),
                task=vlm.get('task_description', 'Robot manipulation task'), generation=vlm.get('generation', {}),
                camera=(config.get('camera_topics') or vlm['camera_topics'])[0], alarm_run=ALARM_RUN,
                context_sec=CONTEXT_SEC, min_frame_spacing_sec=MIN_FRAME_SPACING_SEC,
                max_vlm_frames=MAX_VLM_FRAMES, post_stage3_sec=settings.POST_STAGE3_SEC)


def check_key(identity, bag, episode, boundaries):
    from .prompts import cascade_verification_prompt
    payload = dict(identity=identity, bag=bag, time_sec=round(episode['time_sec'], 6),
                   camera=episode['camera_topic'], boundaries={str(k): round(v, 6) for k, v in boundaries.items()},
                   prompt=cascade_verification_prompt(identity['task'], episode['stage']))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def timing(item, first):
    """Same timing categories as the DINO benchmark."""
    onset = item.get('recorded_failure_time_sec')
    if onset is None or not np.isfinite(onset):
        return ('missing_failure_marker' if item['ground_truth_label'] == 'fail' else
                'false_alarm' if first is not None else 'not_failure'), None
    return ('missed_failure', None) if first is None else ('compared', first - onset)


class CheckCache:
    """checks.jsonl: one line per VLM check; also the record of every answer."""
    def __init__(self, path):
        self.path, self.records = path, {}
        if path.is_file():
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record['key']] = record
        self.new = 0

    def add(self, record):
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        self.records[record['key']] = record
        self.new += 1


def run(config, dino_run, metrics=None, limit=None, progress=print):
    from .run import _slug
    from time_series_gmr_scripts import run_time_series_test as settings
    from vision_dinov2.benchmark_report import metrics_table
    started = time.monotonic()
    dino_run = Path(dino_run)
    saved = json.loads((dino_run / 'run.json').read_text(encoding='utf-8'))
    if saved.get('score_variant') != 'pooled':
        raise ValueError('The saved DINO run must decide on the pooled score, as the VLM check trigger does.')
    scale, stage_topic = float(saved['configured_scale']), saved.get('stage_topic', '/recording_stage')
    folders, saved_rule = list(saved['selected_folders']), saved.get('bag_rule', 'saved rule')
    saved_rule = 'two in a row before stage 3' if saved_rule == 'two_consecutive_same_camera_before_stage3_v1' else saved_rule
    saved_results = pd.read_csv(dino_run / 'bag_results.csv')
    saved_results = saved_results[saved_results.operating_point.eq('configured')]
    frames = pd.read_csv(dino_run / 'frame_results.csv')
    metrics = list(metrics or sorted(frames.metric.unique()))
    output = dino_run / 'vlm_check'
    (output / 'frames').mkdir(parents=True, exist_ok=True)
    identity = question_identity(config)
    reference_names = {Path(bag).name for bag in identity['reference_bags']}
    cache = CheckCache(output / 'checks.jsonl')
    checker = None  # Loaded only when an answer is not cached.
    timelines = {key: group for key, group in frames.groupby(['metric', 'bag_path'])}
    manifest = saved_results[saved_results.metric.eq(metrics[0])]
    evaluated = [row for row in manifest.to_dict('records') if row['status'] == 'ok']
    if limit:
        evaluated = evaluated[:limit]
    outcomes, statuses = {}, {}
    for number, item in enumerate(evaluated, 1):
        bag, name = item['bag_path'], f"{item['folder']}/{item['bag_name']}"
        if Path(bag).name in reference_names:
            statuses[bag] = 'excluded_vlm_reference'
            progress(f'[{number}/{len(evaluated)}] {name}: VLM reference execution, left out of every setup')
            continue
        try:
            boundaries = stage_boundaries(bag, stage_topic)
        except Exception as error:
            statuses[bag] = 'error'
            progress(f'[{number}/{len(evaluated)}] {name}: cannot read stages ({error}); left out of every setup')
            continue
        statuses[bag], buffer, lines = 'ok', None, []
        episodes_by_metric = {metric: alarm_episodes(timelines[metric, bag], boundaries, scale, settings.POST_STAGE3_SEC)
                              if (metric, bag) in timelines else [] for metric in metrics}
        for metric, episodes in episodes_by_metric.items():
            checks, alarm = [], None
            for episode in episodes:
                key = check_key(identity, bag, episode, boundaries)
                if key not in cache.records:
                    if checker is None:
                        checker = VlmChecker(config, progress)
                    if buffer is None:  # Read once per bag, up to its last possible check.
                        until = max(e['time_sec'] for found in episodes_by_metric.values() for e in found)
                        buffer = camera_frames(bag, stage_topic, checker.topic, until)
                    check, images = checker.ask([f for f in buffer if f[0] <= episode['time_sec'] + 1e-9],
                                                boundaries, episode['time_sec'])
                    directory = output / 'frames' / _slug(name)
                    directory.mkdir(parents=True, exist_ok=True)
                    files = []
                    for k, (_, image) in enumerate(images, 1):
                        image = image.copy()
                        image.thumbnail((checker.model.max_image_size, checker.model.max_image_size))
                        files.append(str((directory / f"t{episode['time_sec']:07.2f}_frame_{k}.jpg").relative_to(output)))
                        image.save(output / files[-1], quality=85)
                    cache.add(dict(key=key, bag=bag, folder=item['folder'], alarm=episode, boundaries=boundaries,
                                   reference_response=checker.reference_response, frame_files=files,
                                   created_utc=datetime.now(timezone.utc).isoformat(), **check))
                record = cache.records[key]
                checks.append(dict(time_sec=episode['time_sec'], verdict=record['verdict']))
                if record['verdict'] == 'failure':
                    alarm = episode['time_sec']
                    break
            outcomes[metric, bag] = dict(episodes=episodes, checks=checks, alarm=alarm)
            lines.append(f"{metric}: {len(episodes)} episode(s), VLM "
                         + (', '.join(f"{c['verdict']}@{c['time_sec']:.1f}s" for c in checks) or 'not asked'))
        progress(f'[{number}/{len(evaluated)}] {name}: ' + ' | '.join(lines))

    # One row per metric, setup and bag; every setup is scored on the same bags.
    rows = []
    for item in saved_results[saved_results.metric.isin(metrics)].to_dict('records'):
        bag = item['bag_path']
        if item['status'] == 'ok' and bag not in statuses:
            continue  # Outside --limit.
        status = statuses.get(bag, item['status'])
        rows.append(dict(item, operating_point=SAVED, status=status))
        outcome = outcomes.get((item['metric'], bag))
        row = dict(item, operating_point=CHECKED, status=status, alarm_policy=CHECKED, predicted_decision=None,
                   correct=None, first_alarm_time_sec=None, timing_status='not_scored', failure_time_error_sec=None,
                   alarm_episodes=None, vlm_checks=None)
        if status == 'ok' and outcome is not None:
            first = outcome['alarm']
            predicted = 'failure' if first is not None else 'success'
            row.update(predicted_decision=predicted, first_alarm_time_sec=first,
                       correct=(predicted == 'failure') == (item['ground_truth_label'] == 'fail'),
                       alarm_episodes=json.dumps([round(e['time_sec'], 2) for e in outcome['episodes']]),
                       vlm_checks=json.dumps(outcome['checks']))
            row['timing_status'], row['failure_time_error_sec'] = timing(item, first)
        rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(output / 'bag_results.csv', index=False)
    table = metrics_table(results, folders)
    table.insert(2, 'setup', table.operating_point.map(LABELS))
    table.to_csv(output / 'metrics.csv', index=False)
    for metric in metrics:
        plot_metrics(table[table.metric.eq(metric)], output / f'metrics_{metric}.png', metric, scale)
    asked = [r for r in cache.records.values() if r['bag'] in statuses]
    summary = dict(dino_run=str(dino_run), metrics=metrics, configured_scale=scale, dino_trigger_score='pooled',
                   saved_rule=saved_rule, execution_window=f'until stage 3 + {settings.POST_STAGE3_SEC:g} s',
                   question=identity, evaluated_bags=sum(s == 'ok' for s in statuses.values()),
                   excluded_vlm_reference=sorted(b for b, s in statuses.items() if s == 'excluded_vlm_reference'),
                   unreadable_bags=sorted(b for b, s in statuses.items() if s == 'error'),
                   vlm_checks=len(asked), vlm_checks_new=cache.new,
                   vlm_verdicts={v: sum(r['verdict'] == v for r in asked) for v in ('safe', 'failure')},
                   vlm_sec_median=float(np.median([r['vlm_sec'] for r in asked])) if asked else None,
                   limit=limit, elapsed_sec=time.monotonic() - started,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    (output / 'run.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_summary(output, table, summary)
    progress(f'Results: {output}')
    return output


def plot_metrics(table, path, metric, scale):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folders = list(dict.fromkeys(table.folder))
    if table.failure_count.sum() == 0:
        panels = [('false_positive_rate', 'False-positive rate: normal runs alarmed'),
                  ('accuracy', 'Accuracy: normal runs accepted')]
    elif table.normal_count.sum() == 0:
        panels = [('recall', 'Recall'), ('accuracy', 'Accuracy')]
    else:
        panels = [('recall', 'Recall'), ('false_positive_rate', 'False-positive rate'), ('precision', 'Precision'),
                  ('accuracy', 'Accuracy'), ('f1', 'F1'), ('alarm_rate', 'Confirmed alarm rate')]
    columns = 2 if len(panels) == 2 else 3
    figure, axes = plt.subplots(len(panels) // columns, columns, figsize=(max(13, len(folders) * 2.6), 6 if columns == 2 else 9),
                                squeeze=False)
    for axis, (column, title) in zip(axes.ravel(), panels):
        for offset, setup in zip((-.18, .18), SETUPS):
            rows = table[table.operating_point.eq(setup)].set_index('folder').reindex(folders)
            values = rows[column].to_numpy(dtype=float)
            positions = np.arange(len(folders)) + offset
            axis.bar(positions, np.nan_to_num(values) * 100, width=.34, color=COLORS[setup], label=LABELS[setup])
            for position, value in zip(positions, values):
                axis.annotate('N/A' if not np.isfinite(value) else f'{100 * value:.0f}%',
                              (position, 2 if not np.isfinite(value) else 100 * value + 2),
                              ha='center', fontsize=8, rotation=90, color=COLORS[setup])
        axis.set(title=title, ylabel='Percent', ylim=(0, 125), xlim=(-.6, len(folders) - .4))
        axis.set_yticks(np.arange(0, 101, 20))
        counts = table[table.operating_point.eq(CHECKED)].set_index('folder').reindex(folders).evaluated_bags
        axis.set_xticks(range(len(folders)), [f'{f}\n({int(n)} bags)' for f, n in zip(folders, counts)],
                        rotation=30, ha='right', fontsize=8)
        axis.grid(axis='y', alpha=.2)
    handles, names = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .965), ncol=2, fontsize=9)
    figure.suptitle(f'{metric.title()} DINO, configured scale {scale:g} | DINO alone vs DINO + VLM check')
    figure.tight_layout(rect=(0, 0, 1, .92))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_summary(output, table, summary):
    lines = ['# DINO + VLM check on a saved DINO benchmark', '',
             f"Saved DINO run: `{summary['dino_run']}` (DINO was not rerun; its saved pooled-score crossings at "
             f"scale {summary['configured_scale']:g} were replayed).", '',
             f"- **{LABELS[SAVED]}**: the saved run's own decision ({summary['saved_rule']}).",
             f"- **{LABELS[CHECKED]}**: {ALARM_RUN} consecutive DINO crossings in one camera, "
             f"{summary['execution_window']}, ask the base VLM (turn 1: complete nominal execution "
             f"{', '.join(summary['question']['reference_bags'])}; turn 2: up to {MAX_VLM_FRAMES} frames up to the "
             'alarm, stage-aware question). A failure (or unclear) answer is the bag alarm; a safe answer continues.', '',
             f"Evaluated bags: {summary['evaluated_bags']}. VLM reference execution(s) left out of every setup: "
             f"{', '.join(Path(b).name for b in summary['excluded_vlm_reference']) or 'none'}."
             + (f" Unreadable: {', '.join(summary['unreadable_bags'])}." if summary['unreadable_bags'] else ''),
             f"VLM checks: {summary['vlm_checks']} ({summary['vlm_checks_new']} new in this run); verdicts "
             f"{summary['vlm_verdicts']}; median {summary['vlm_sec_median'] or 0:.1f} s per check.", '']
    for metric in summary['metrics']:
        rows = table[table.metric.eq(metric)]
        lines += [f'## {metric.title()}', '', f'![metrics](metrics_{metric}.png)', '',
                  '| Folder | Bags | ' + ' | '.join(LABELS[s] for s in SETUPS) + ' |',
                  '|---|---|---|---|']
        for folder in dict.fromkeys(rows.folder):
            cells = []
            for setup in SETUPS:
                row = rows[rows.folder.eq(folder) & rows.operating_point.eq(setup)].iloc[0]
                text = f"{int(row.alarm_bags)}/{int(row.evaluated_bags)} alarmed"
                if row.failure_count:
                    text += f", recall {100 * row.recall:.0f}%"
                if row.normal_count:
                    text += f", FPR {100 * row.false_positive_rate:.0f}%"
                cells.append(text)
            evaluated = int(rows[rows.folder.eq(folder)].evaluated_bags.iloc[0])
            lines.append(f'| {folder} | {evaluated} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['Per-bag outcomes: `bag_results.csv` (setup in `operating_point`). Every VLM question, answer and '
              'its input frames: `checks.jsonl` and `frames/`. Metrics: `metrics.csv`.',
              'Recall needs failure-labelled bags; when every bag is labelled normal, only false alarms can be measured.']
    (output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
