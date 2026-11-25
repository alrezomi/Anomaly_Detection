"""Offline benchmark of the DINO -> VLM check on a saved DINO benchmark run.

DINO is not run again. Its saved per-frame crossings (pooled score above the
configured threshold, the DINO benchmark's decision score) are replayed with
the live rule: ALARM_RUN consecutive crossings in one camera, again after a
stage change, until POST_STAGE3_SEC after stage 3 on the causal stage clock.
Each new alarm episode asks the VLM exactly as the live replay does
(VlmChecker), with the gripper-camera frames the live replay would have
buffered. A failure answer is the bag's alarm; a safe answer lets the
execution continue.

Two setups are reported on the same bags: DINO alone (the saved run's own
decision) and DINO with the VLM check. With rynnbrain.ts_check.enabled a third
one, DINO + TS with the VLM check, adds the TS alarm checks of the live
cascade: GMR is replayed causally on each bag (prepared exactly as the GUI
prepares it, default signal mode TCP position + force), TsAlarmEpisodes decides
when a TS alarm asks the VLM, and DINO and TS checks are asked in time order
until the first failure answer.

Per alarm (alarm_checks.csv, alarm_metrics.csv): every check of the most
complete setup with its answer and whether its alarm was true. With
all_checks, the replay does not end at the first failure answer, so every
alarm of the execution gets its check (those after the first stop are marked
after_stop); this measures how often the VLM rightly ignores a false alarm
and rightly confirms a true one. An alarm is true from the recorded failure
marker on in a failure bag and false in a normal bag; before the marker, or
without one, it is unclear (alarm_truth).

Results go to <dino run>/vlm_check/ unless another output directory is given.
VLM answers are cached in checks.jsonl, keyed by bag, alarm and the exact
questions, so a rerun resumes where it stopped, both DINO metrics share
identical checks, and reports can be regenerated without loading the VLM.
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .cascade import (ALARM_RUN, CHECK_OFFSETS_SEC, RELEASE_CHECK_DELAY_SEC, SUBGOAL_THRESHOLD, TS_MAX_RECHECKS,
                      TS_RECHECK_SEC, AlarmEpisodes, TsAlarmEpisodes, VlmChecker, current_stage, group_ts_alarms,
                      question_text, ts_trigger)

SAVED, CHECKED, TS_CHECKED = 'dino_alone', 'dino_vlm_check', 'dino_ts_vlm_check'
SETUPS = (SAVED, CHECKED)
COLORS = {SAVED: '#2878b5', CHECKED: '#c87925', TS_CHECKED: '#3a8f5c'}
LABELS = {SAVED: 'DINO alone', CHECKED: 'DINO + VLM check', TS_CHECKED: 'DINO + TS + VLM check'}
def alarm_truth(item, time_sec):
    """'failure' (a true alarm), 'normal' (a false alarm) or 'unclear', from the bag label and failure marker.

    Every alarm in a normal bag is false; in a failure bag an alarm at or after the recorded marker is true. Before
    the marker it is unclear: the operator presses it after seeing the failure (on the project's grasp-miss and
    slip bags, empty-grip alarms came 1-4 s before it, once the gripper had already closed on nothing)."""
    if item['ground_truth_label'] != 'fail':
        return 'normal'
    onset = item.get('recorded_failure_time_sec')
    if onset is None or not np.isfinite(onset):
        return 'unclear'
    return 'failure' if time_sec >= onset else 'unclear'


def alarm_kind(check):
    return 'DINO' if check['source'] == 'dino' else 'TS: ' + ' + '.join(check['alarms'].split('+'))


def alarm_metrics(checks):
    """Per alarm kind: how the VLM answered true and false alarms (checks: alarm_checks.csv rows)."""
    rows = []
    for kind, group in [*checks.groupby(checks.apply(alarm_kind, axis=1), sort=True), ('All alarms', checks)]:
        asked = group[group.asked]
        true, false = asked[asked.alarm_truth.eq('failure')], asked[asked.alarm_truth.eq('normal')]
        rows.append({'alarm': kind, 'checks': len(group),
                     'true alarms asked': len(true), 'true: stopped (right)': int(true.verdict.eq('failure').sum()),
                     'true: ignored (missed)': int(true.verdict.eq('safe').sum()),
                     'false alarms asked': len(false), 'false: ignored (right)': int(false.verdict.eq('safe').sum()),
                     'false: stopped (wrong)': int(false.verdict.eq('failure').sum()),
                     'not asked (before the grasp)': int((~group.asked).sum()),
                     'unclear': int(asked.alarm_truth.eq('unclear').sum())})
    return pd.DataFrame(rows)


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
        threshold, stage = float(row.base_threshold) * scale, current_stage(boundaries, t)
        if rule.update(row.camera_topic, bool(row.pooled_score > threshold), stage):
            episodes.append(dict(time_sec=t, camera_topic=row.camera_topic, stage=stage,
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


def question_identity(config, questions=None):
    """Everything that determines the VLM's answer, without loading the model.

    questions: the prompts.QuestionSet; derived questions (and their thresholds) are part of the identity.
    """
    from time_series_gmr_scripts import run_time_series_test as settings
    vlm = config['rynnbrain']
    check = vlm.get('vlm_check') or {}
    identity = dict(check='subgoal_v2', model=vlm['model']['model_id'], camera=check.get('camera_topic'),
                    item=check.get('item'), destination=check.get('destination'), alarm_run=ALARM_RUN,
                    check_offsets_sec=list(CHECK_OFFSETS_SEC), subgoal_threshold=SUBGOAL_THRESHOLD,
                    post_stage3_sec=settings.POST_STAGE3_SEC)
    if questions is not None and questions.source == 'reference':
        identity['questions'] = questions.identity()
    return identity


def question_set(config, progress, checker_class=None):
    """The checker's questions without loading the VLM when possible: the fixed ones, or the saved derived ones.

    Returns (questions, checker): the checker is loaded only when derived questions must be made first.
    """
    from .prompts import QuestionSet
    check = config['rynnbrain'].get('vlm_check') or {}
    if check.get('questions', 'fixed') != 'reference':
        return QuestionSet(check.get('item'), check.get('destination'), default_threshold=SUBGOAL_THRESHOLD), None
    from .reference_questions import load_questions
    questions = load_questions(config)
    if questions is not None:
        return questions, None
    checker = (checker_class or VlmChecker)(config, progress)
    return checker.questions, checker


def check_key(identity, bag, episode, boundaries, questions):
    payload = dict(identity=identity, bag=bag, time_sec=round(episode['time_sec'], 6),
                   camera=episode['camera_topic'], boundaries={str(k): round(v, 6) for k, v in boundaries.items()},
                   questions=questions.subgoal(episode['stage']))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def ts_check_key(identity, bag, event, boundaries, questions):
    payload = dict(identity=identity, bag=bag, time_sec=round(event['time_sec'], 6), alarms=event['alarms'],
                   rechecks=event['rechecks'], boundaries={str(k): round(v, 6) for k, v in boundaries.items()},
                   questions=questions.for_alarm(event['alarms'][0], event['stage'])[1],
                   ts_rule=dict(recheck_sec=TS_RECHECK_SEC, max_rechecks=TS_MAX_RECHECKS,
                                release_delay_sec=RELEASE_CHECK_DELAY_SEC))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def ts_alarm_events(gmr, bag, questions, progress, stop):
    """The TS alarm checks the live cascade asks in this bag: a causal GMR replay (prepared as in the GUI) and
    TsAlarmEpisodes. Each event is one check (alarms about the same sub-goal at one sample share it)."""
    from rosbag_io import iter_ts_replay_events
    from time_series_gmr_scripts.online_gmr import CausalSamples
    from time_series_gmr_scripts import run_time_series_test as settings
    gmr.reset(bag)
    gmr.prepare_replay(progress, stop)
    sampler = CausalSamples(gmr.sample_kinds, stop, release=gmr.release_check, post_sec=settings.POST_STAGE3_SEC)
    sampler.columns, sampler.sensors, sampler.pose_frame = gmr.columns, gmr.sensors, gmr.pose_frame
    gmr.sampler = sampler
    rule, events = TsAlarmEpisodes(), []
    for timestamp, kind, payload in sampler.packets(iter_ts_replay_events(bag, gmr.topics, gmr.stage_topic, None)):
        if kind == 'release':
            gmr.check_release(payload)
        elif kind == 'sample':
            gmr.pending = payload
            row, _, _ = gmr.append(None, timestamp)
            stage = current_stage(sampler.clock.boundaries, timestamp)
            for alarms in group_ts_alarms(rule.update(row['sensors'], stage, timestamp, gmr.release_result),
                                          stage, questions):
                events.append(ts_trigger(alarms, timestamp, stage))
    return events


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


def run(config, dino_run, metrics=None, limit=None, progress=print, output=None, per_folder=None,
        ts_signal='pose_force', all_checks=False):
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
    output = Path(output) if output else dino_run / 'vlm_check'
    (output / 'frames').mkdir(parents=True, exist_ok=True)
    # The VLM is loaded only when an answer is not cached (or derived questions must be made first).
    questions, checker = question_set(config, progress, VlmChecker)
    identity = question_identity(config, questions)
    cache = CheckCache(output / 'checks.jsonl')
    timelines = {key: group for key, group in frames.groupby(['metric', 'bag_path'])}
    manifest = saved_results[saved_results.metric.eq(metrics[0])]
    evaluated = [row for row in manifest.to_dict('records') if row['status'] == 'ok']
    if per_folder:
        evaluated = [row for _, rows in pd.DataFrame(evaluated).groupby('folder', sort=False)
                     for row in rows.head(per_folder).to_dict('records')]
    if limit:
        evaluated = evaluated[:limit]
    ts_check = bool((config['rynnbrain'].get('ts_check') or {}).get('enabled', False))
    setups = (SAVED, CHECKED, TS_CHECKED) if ts_check else SETUPS
    gmr, ts_notes, stop = None, {}, __import__('threading').Event()
    if ts_check:
        from time_series_gmr_scripts.online_gmr import GmrOnlineDetector
        gmr = GmrOnlineDetector(config, progress)
        gmr.signal_mode = ts_signal
    outcomes, statuses, alarm_rows = {}, {}, []
    for number, item in enumerate(evaluated, 1):
        bag, name = item['bag_path'], f"{item['folder']}/{item['bag_name']}"
        try:
            boundaries = stage_boundaries(bag, stage_topic)
        except Exception as error:
            statuses[bag] = 'error'
            progress(f'[{number}/{len(evaluated)}] {name}: cannot read stages ({error}); left out of every setup')
            continue
        statuses[bag], buffer, lines = 'ok', None, []
        episodes_by_metric = {metric: alarm_episodes(timelines[metric, bag], boundaries, scale, settings.POST_STAGE3_SEC)
                              if (metric, bag) in timelines else [] for metric in metrics}
        ts_events = []
        if gmr is not None:
            try:
                # GMR preparation messages (training, missing topics) are shown for the first bag only.
                ts_events = ts_alarm_events(gmr, bag, questions,
                                            progress if number == 1 else (lambda _message: None), stop)
            except Exception as error:
                ts_notes[bag] = str(error)
                progress(f'[{number}/{len(evaluated)}] {name}: no TS replay ({error})')
        for metric, episodes in episodes_by_metric.items():
            for setup in setups[1:]:
                events = [dict(e, source='dino') for e in episodes]
                if setup == TS_CHECKED:
                    events = sorted(events + ts_events, key=lambda e: (e['time_sec'], e['source']))
                checks, alarm = [], None
                for event in events:
                    dino = event['source'] == 'dino'
                    key = (check_key(identity, bag, event, boundaries, questions) if dino else
                           ts_check_key(identity, bag, event, boundaries, questions))
                    if key not in cache.records:
                        if checker is None:
                            checker = VlmChecker(config, progress)
                        if buffer is None:  # Read once per bag, up to its last possible check.
                            until = max(e['time_sec'] for e in [*ts_events, *[x for found in episodes_by_metric.values()
                                                                              for x in found]])
                            buffer = camera_frames(bag, stage_topic, checker.topic, until)
                        frames = [f for f in buffer if f[0] <= event['time_sec'] + 1e-9]
                        check, images = (checker.ask(frames, boundaries, event['time_sec']) if dino else
                                         checker.ask(frames, boundaries, event['time_sec'], alarm=event['alarms'][0]))
                        directory = output / 'frames' / _slug(name)
                        directory.mkdir(parents=True, exist_ok=True)
                        files = []
                        prefix = 'dino' if dino else 'ts_' + '_'.join(event['alarms'])
                        for k, (_, image) in enumerate(images, 1):
                            image = image.copy()
                            image.thumbnail((checker.model.max_image_size, checker.model.max_image_size))
                            files.append(str((directory / f"t{event['time_sec']:07.2f}_{prefix}_frame_{k}.jpg").relative_to(output)))
                            image.save(output / files[-1], quality=85)
                        cache.add(dict(key=key, bag=bag, folder=item['folder'], alarm=event, boundaries=boundaries,
                                       frame_files=files, created_utc=datetime.now(timezone.utc).isoformat(), **check))
                    record = cache.records[key]
                    checks.append(dict(time_sec=event['time_sec'], source=event['source'],
                                       alarms=event.get('alarms'), verdict=record['verdict']))
                    if setup == setups[-1]:
                        alarm_rows.append(dict(
                            metric=metric, folder=item['folder'], bag_path=bag, bag_name=item['bag_name'],
                            ground_truth_label=item['ground_truth_label'],
                            recorded_failure_time_sec=item.get('recorded_failure_time_sec'),
                            time_sec=event['time_sec'], source=event['source'],
                            alarms='+'.join(event.get('alarms') or ['dino']),
                            recheck=max(event.get('rechecks') or [0]), stage=record['stage'],
                            goal_stage=record.get('goal_stage', record['stage']), asked=bool(record.get('questions')),
                            subgoal_probability=record.get('subgoal_probability'), verdict=record['verdict'],
                            after_stop=alarm is not None, alarm_truth=alarm_truth(item, event['time_sec'])))
                    if record['verdict'] == 'failure' and alarm is None:
                        alarm = event['time_sec']
                        if not all_checks:
                            break
                outcomes[metric, bag, setup] = dict(episodes=events, checks=checks, alarm=alarm)
                asked = [c for c in checks if c['source'] == ('dino' if setup == CHECKED else 'ts')]
                lines.append(f"{metric} {LABELS[setup]}: " + (', '.join(
                    f"{c['verdict']}@{c['time_sec']:.1f}s" + ('' if c['source'] == 'dino' else '(' + '+'.join(c['alarms']) + ')')
                    for c in (checks if setup == CHECKED else asked)) or 'no checks'))
        progress(f'[{number}/{len(evaluated)}] {name}: ' + ' | '.join(lines))

    # One row per metric, setup and bag; every setup is scored on the same bags.
    rows = []
    for item in saved_results[saved_results.metric.isin(metrics)].to_dict('records'):
        bag = item['bag_path']
        if item['status'] == 'ok' and bag not in statuses:
            continue  # Outside --limit or --per-folder.
        status = statuses.get(bag, item['status'])
        rows.append(dict(item, operating_point=SAVED, status=status))
        for setup in setups[1:]:
            outcome = outcomes.get((item['metric'], bag, setup))
            row = dict(item, operating_point=setup, status=status, alarm_policy=setup, predicted_decision=None,
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
        plot_metrics(table[table.metric.eq(metric)], output / f'metrics_{metric}.png', metric, scale, setups)
    alarm_checks = pd.DataFrame(alarm_rows, columns=[
        'metric', 'folder', 'bag_path', 'bag_name', 'ground_truth_label', 'recorded_failure_time_sec', 'time_sec',
        'source', 'alarms', 'recheck', 'stage', 'goal_stage', 'asked', 'subgoal_probability', 'verdict', 'after_stop',
        'alarm_truth'])
    alarm_checks.to_csv(output / 'alarm_checks.csv', index=False)
    per_alarm = pd.concat([alarm_metrics(group).assign(metric=metric)
                           for metric, group in alarm_checks.groupby('metric')]) if len(alarm_checks) else pd.DataFrame()
    per_alarm.to_csv(output / 'alarm_metrics.csv', index=False)
    checked = [r for r in cache.records.values() if r['bag'] in statuses]
    asked = [r for r in checked if r.get('questions')]  # Stage 1 alarms are answered without the VLM.
    summary = dict(dino_run=str(dino_run), metrics=metrics, configured_scale=scale, dino_trigger_score='pooled',
                   saved_rule=saved_rule, execution_window=f'until stage 3 + {settings.POST_STAGE3_SEC:g} s',
                   question=identity, evaluated_bags=sum(s == 'ok' for s in statuses.values()),
                   unreadable_bags=sorted(b for b, s in statuses.items() if s == 'error'),
                   vlm_checks=len(checked), vlm_checks_asked=len(asked), vlm_checks_new=cache.new,
                   vlm_verdicts={v: sum(r['verdict'] == v for r in checked) for v in ('safe', 'failure')},
                   vlm_sec_median=float(np.median([r['vlm_sec'] for r in asked])) if asked else None,
                   setups=list(setups), ts_check=ts_check, ts_signal=ts_signal if ts_check else None,
                   ts_checks=sum(r['alarm'].get('source') == 'ts' for r in checked),
                   ts_not_replayed=ts_notes, all_checks=all_checks,
                   limit=limit, per_folder=per_folder, elapsed_sec=time.monotonic() - started,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    (output / 'run.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_summary(output, table, summary, per_alarm)
    progress(f'Results: {output}')
    return output


def plot_metrics(table, path, metric, scale, setups=SETUPS):
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
        panels = [('recall', 'Recall: failures alarmed'), ('false_positive_rate', 'False-positive rate: normal runs alarmed'),
                  ('accuracy', 'Accuracy')]
    # One panel per row keeps many folders readable.
    figure, axes = plt.subplots(len(panels), 1, figsize=(max(13, len(folders) * 1.3), 4.6 * len(panels)), squeeze=False)
    for axis, (column, title) in zip(axes.ravel(), panels):
        width = .68 / len(setups)
        for offset, setup in zip(np.linspace(-.34 + width / 2, .34 - width / 2, len(setups)), setups):
            rows = table[table.operating_point.eq(setup)].set_index('folder').reindex(folders)
            values = rows[column].to_numpy(dtype=float)
            positions = np.arange(len(folders)) + offset
            axis.bar(positions, np.nan_to_num(values) * 100, width=width, color=COLORS[setup], label=LABELS[setup])
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
    figure.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .965), ncol=len(setups), fontsize=9)
    figure.suptitle(f'{metric.title()} DINO, configured scale {scale:g} | ' + ' vs '.join(LABELS[s] for s in setups))
    figure.tight_layout(rect=(0, 0, 1, .92))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_summary(output, table, summary, per_alarm=None):
    lines = ['# DINO + VLM check on a saved DINO benchmark', '',
             f"Saved DINO run: `{summary['dino_run']}` (DINO was not rerun; its saved pooled-score crossings at "
             f"scale {summary['configured_scale']:g} were replayed).", '',
             f"- **{LABELS[SAVED]}**: the saved run's own decision ({summary['saved_rule']}).",
             f"- **{LABELS[CHECKED]}**: {ALARM_RUN} consecutive DINO crossings in one camera (again after a stage "
             f"change), {summary['execution_window']}, ask the base VLM about the gripper camera "
             f"(`{summary['question']['camera']}`) whether the sub-goal of the current stage holds: after the grasp "
             f"(stage 2) {summary['question']['item']} is held between the fingers, after the release (stage 3) it "
             f"lies on {summary['question']['destination']} below the gripper (`subgoal_questions` in "
             f"`prompts.py`). Each gripper frame {', '.join(f'{o:g}' for o in CHECK_OFFSETS_SEC)} s before the alarm "
             f"is asked alone; a mean probability below {SUBGOAL_THRESHOLD:g} that the sub-goal holds is the bag "
             'alarm, otherwise the execution continues. Stage 1 alarms have no reached sub-goal and continue.', '',
             *([f"  Questions derived by the VLM from the nominal reference runs (`reference_questions.py`), used "
                'instead of the fixed ones above for these stages:', '',
                *(f"  - stage {goal}: \"{question_text(q['question'])}\" (failure below {q['threshold']:.2f})"
                  for goal, q in summary['question']['questions']['derived'].items()), '']
               if summary['question'].get('questions') else []),
             *([f"- **{LABELS[TS_CHECKED]}**: as above, and every TS alarm of the live cascade (GMR signal "
                f"`{summary['ts_signal']}`: position GMR, force safety limit, empty grip; release position "
                f"{RELEASE_CHECK_DELAY_SEC:g} s after the opening) asks the VLM when it starts and every "
                f"{TS_RECHECK_SEC:g} s while it lasts (at most {TS_MAX_RECHECKS}), about the state it concerns "
                "(`ts_alarm_questions` in `prompts.py`: empty grip: held between the fingers; release position: "
                "where the object lies; other alarms: the current stage's sub-goal, nothing before the grasp). "
                "DINO and TS checks are asked in time order; the first failure answer is the bag alarm.", '']
               if summary.get('ts_check') else []),
             f"Evaluated bags: {summary['evaluated_bags']}."
             + (f" Unreadable: {', '.join(summary['unreadable_bags'])}." if summary['unreadable_bags'] else ''),
             f"Checks: {summary['vlm_checks']} ({summary['vlm_checks_new']} new in this run), "
             f"{summary['vlm_checks_asked']} asked the VLM (stage 2 or 3); verdicts {summary['vlm_verdicts']}; "
             f"median {summary['vlm_sec_median'] or 0:.1f} s per VLM check."
             + (f" TS checks: {summary['ts_checks']}." if summary.get('ts_check') else '')
             + (f" Bags without a TS replay: {len(summary['ts_not_replayed'])}." if summary.get('ts_not_replayed') else ''),
             '']
    setups = summary.get('setups') or list(SETUPS)
    for metric in summary['metrics']:
        rows = table[table.metric.eq(metric)]
        lines += [f'## {metric.title()}', '', f'![metrics](metrics_{metric}.png)', '',
                  '| Folder | Bags | ' + ' | '.join(LABELS[s] for s in setups) + ' |',
                  '|---|---|' + '---|' * len(setups)]
        for folder in dict.fromkeys(rows.folder):
            cells = []
            for setup in setups:
                row = rows[rows.folder.eq(folder) & rows.operating_point.eq(setup)].iloc[0]
                text = f"{int(row.alarm_bags)}/{int(row.evaluated_bags)} alarmed"
                if row.failure_count:
                    text += f", recall {100 * row.recall:.0f}%"
                    if row.timing_compared:
                        text += f", median delay {row.timing_median_sec:+.1f} s"
                if row.normal_count:
                    text += f", FPR {100 * row.false_positive_rate:.0f}%"
                cells.append(text)
            evaluated = int(rows[rows.folder.eq(folder)].evaluated_bags.iloc[0])
            lines.append(f'| {folder} | {evaluated} | ' + ' | '.join(cells) + ' |')
        lines.append('')
        if per_alarm is not None and len(per_alarm):
            rows = per_alarm[per_alarm.metric.eq(metric)].drop(columns='metric')
            lines += [f"### Per alarm ({LABELS[setups[-1]]}" + (', every alarm checked, also after the first stop)'
                                                                 if summary.get('all_checks') else
                                                                 ', checks until the first stop only)'), '',
                      'A true alarm comes at or after the recorded failure marker of a failure bag; a false alarm in a '
                      'normal bag. Before the marker the failure may already be going on, so those alarms are unclear. '
                      'Right answers: stop on a true alarm, ignore a false one.', '',
                      '| ' + ' | '.join(rows.columns) + ' |', '|' + '---|' * len(rows.columns)]
            lines += ['| ' + ' | '.join(str(v) for v in values) + ' |' for values in rows.itertuples(index=False)]
            lines.append('')
    lines += ['Per-alarm checks: `alarm_checks.csv` (summary `alarm_metrics.csv`).',
              'Per-bag outcomes: `bag_results.csv` (setup in `operating_point`). Every VLM question, answer and '
              'its input frames: `checks.jsonl` and `frames/`. Metrics: `metrics.csv`.',
              'Recall needs failure-labelled bags; when every bag is labelled normal, only false alarms can be measured. '
              'Delay is first alarm minus the recorded failure marker (negative: alarm before the marker).']
    (output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
