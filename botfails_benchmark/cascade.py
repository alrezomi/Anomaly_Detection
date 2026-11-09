"""Causal replay of our DINO -> VLM cascade on BotFails episodes.

Exactly the live rule (rynnbrain_vlm/cascade.py): ALARM_RUN consecutive DINO
frames above the scaled threshold start an alarm episode; it is re-armed when
the score drops below the threshold or the task phase changes. At an alarm the
base VLM answers the phase's questions (prompts.task_questions) about the camera
frames CHECK_OFFSETS_SEC before the alarm; the mean probability over frames of
each question is its sub-goal probability, and a sub-goal below
SUBGOAL_THRESHOLD stops the execution (a 'failure' verdict). Our robot reports
its stage; here the gripper defines the phase (phases.py).
"""
from __future__ import annotations

import hashlib
import json
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

from .data import episodes, load_image
from .phases import episode_phases
from .prompts import holds_probability, task_questions


def load_scores(dino_run: Path, task: str, split: str) -> pd.DataFrame:
    """DINO scores per frame: test frames, or leave-one-episode-out scores of normal training frames."""
    if split == 'test':
        frame = pd.read_csv(dino_run / task / 'test_scores.csv')
        return frame[['episode', 'frame_index', 'timestamp_sec', 'pooled_score', 'base_threshold', 'label']]
    frame = pd.read_csv(dino_run / task / 'calibration_rows.csv')
    frame['episode'] = frame['episode'].str.rsplit('/', n=1).str[-1]
    return frame.rename(columns={'score_pooled': 'pooled_score', 'threshold_pooled': 'base_threshold'})[
        ['episode', 'frame_index', 'timestamp_sec', 'pooled_score', 'base_threshold']].assign(label=0)


class Checker:
    """Base VLM answers, cached on disk by (image, question) so reruns only ask new questions."""
    def __init__(self, config, tasks_config, camera, cache_path: Path, log, thresholds=None):
        from rynnbrain_vlm.cascade import CHECK_OFFSETS_SEC, SUBGOAL_THRESHOLD
        self.offsets, self.threshold = CHECK_OFFSETS_SEC, SUBGOAL_THRESHOLD
        self.thresholds = thresholds or {}  # {task: {question name: threshold}}, calibrated on normal data
        self.tasks, self.camera, self.log = tasks_config, camera, log
        self.cache_path, self.cache, self.model = cache_path, {}, None
        self.vlm_settings = {**config['rynnbrain']['model'], 'lora_adapter_path': None,
                             **(config['rynnbrain'].get('vlm_check') or {}).get('model', {})}
        if cache_path.is_file():
            for line in cache_path.read_text().splitlines():
                record = json.loads(line)
                self.cache[record['key']] = record['distribution']
        self.asked = 0

    def distribution(self, path: Path, question: dict) -> list[float]:
        crop = question.get('crop')
        identity = [str(path), question['text'], question['options']] + ([crop] if crop else [])
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        if key not in self.cache:
            if self.model is None:
                from rynnbrain_vlm.model import RynnBrainModel
                self.log('Loading the base VLM (no LoRA) for alarm checks')
                self.model = RynnBrainModel(self.vlm_settings)
            turn = dict(role='user', images=[('Camera image:', load_image(path, crop))], text=question['text'])
            self.cache[key] = self.model.answer_distribution(turn, tuple(question['options']))
            self.asked += 1
            with self.cache_path.open('a') as stream:
                stream.write(json.dumps(dict(key=key, image=str(path), question=question['name'],
                                             distribution=self.cache[key])) + '\n')
        return self.cache[key]

    def ask(self, task, phase, frames, alarm_sec, trigger='dino'):
        questions = task_questions(self.tasks[task], phase, 'release' if trigger == 'release' else 'dino')
        if not questions:
            return dict(verdict='safe', subgoal_probability=None, frames=[], questions=[],
                        answer=f'Phase {phase}: no sub-goal to check. Safe to continue.')
        stamps = [t for _, t, _ in frames]
        chosen, used = [], set()
        for offset in self.offsets:
            position = bisect_right(stamps, alarm_sec - offset + 1e-9) - 1
            if position < 0 or position in used:
                continue
            used.add(position)
            chosen.append(frames[position])
        started = time.monotonic()
        results = []
        for question in questions:
            per_frame = [holds_probability(self.distribution(path, question), question['options'], question['holds'])
                         for _, _, path in chosen]
            limit = float(self.thresholds.get(task, {}).get(question['name'], self.threshold))
            results.append(dict(name=question['name'], probability=float(np.mean(per_frame)), threshold=limit,
                                per_frame=[round(p, 4) for p in per_frame]))
        probability = min(r['probability'] for r in results)
        verdict = 'failure' if any(r['probability'] < r['threshold'] for r in results) else 'safe'
        return dict(verdict=verdict, subgoal_probability=probability, questions=results,
                    frames=[dict(frame_index=f, time_sec=t) for f, t, _ in chosen],
                    vlm_sec=time.monotonic() - started)


def replay_episode(task, episode, scores, checker, camera, scale, recheck_sec=None, release_delay=None, confirm_sec=None):
    """recheck_sec: ask again every so often while an alarm persists. release_delay: also check
    this long after every release, whether DINO alarms or not (failures DINO cannot see).
    confirm_sec: a failing check must fail again this much later before the execution stops."""
    from rynnbrain_vlm.cascade import ALARM_RUN, AlarmEpisodes
    phases = episode_phases(episode)
    frames = episode.frame_table(camera)
    tracker, alarms, stop, last_ask, confirm_due = AlarmEpisodes(), [], None, float('-inf'), None
    config = checker.tasks[task]
    has_release_check = any('release' in check.get('triggers', ('dino',)) for check in config.get('checks', []))
    releases = ([t + release_delay for t, kind in phases.events if kind == 'release']
                if release_delay is not None and has_release_check else [])
    dino_decides = config.get('decision') == 'dino'
    for row in scores.sort_values('timestamp_sec').itertuples(index=False):
        t = float(row.timestamp_sec)
        index, kind = phases.at(t)
        above = float(row.pooled_score) > float(row.base_threshold) * scale
        trigger = 'dino' if tracker.update('dino', above, index) else None
        if trigger == 'dino' and dino_decides:
            alarms.append(dict(time_sec=t, frame_index=int(row.frame_index), phase=kind, phase_index=index, trigger=trigger,
                               verdict='failure', subgoal_probability=None, questions=[], frames=[],
                               answer='This task trusts the DINO alarm: no VLM question covers its failure types.'))
            stop = t
            break
        if trigger is None and recheck_sec and above and tracker.streaks['dino'] >= ALARM_RUN and t - last_ask >= recheck_sec:
            trigger = 'recheck'
        while releases and releases[0] <= t + 1e-9:
            releases.pop(0)
            trigger = trigger or 'release'
        if confirm_due is not None and t + 1e-9 >= confirm_due:
            trigger = 'confirm'
        if trigger:
            last_ask = t
            check = checker.ask(task, kind, frames, t, trigger)
            alarms.append(dict(time_sec=t, frame_index=int(row.frame_index), phase=kind, phase_index=index,
                               trigger=trigger, **check))
            if check['verdict'] == 'failure':
                if confirm_sec is None or trigger == 'confirm':
                    stop = t
                    break
                confirm_due = t + confirm_sec
            elif trigger == 'confirm':
                confirm_due = None
    return dict(task=task, episode=episode.name, split=episode.split, stop_sec=stop,
                alarms=alarms, phases=phases.summary())


def run(config, args, log):
    tasks_config = {k: v for k, v in json.loads(Path(args.tasks_config).read_text()).items() if not k.startswith('_')}
    scale = float(args.threshold_scale if args.threshold_scale is not None else config.get('dino_threshold_scale', 1.))
    thresholds = json.loads(Path(args.vlm_thresholds).read_text()) if args.vlm_thresholds else None
    split = json.loads(Path(args.split_file).read_text()) if args.split_file else None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run.json').write_text(json.dumps(dict(
        dino_run=str(args.dino_run), tasks_config=tasks_config, camera=args.camera, split=args.split,
        threshold_scale=scale, recheck_sec=args.recheck_sec, release_delay=args.release_delay, confirm_sec=args.confirm_sec,
        vlm_thresholds=thresholds, subset=args.subset, started=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1))
    cache_path = Path(args.vlm_cache) if args.vlm_cache else output / 'vlm_cache.jsonl'
    checker = Checker(config, tasks_config, args.camera, cache_path, log, thresholds)
    from .data import TASKS
    for task in args.task or TASKS:
        if not (Path(args.dino_run) / task / 'done.json').is_file():
            log(f'{task}: DINO scores missing, skipped')
            continue
        scores = load_scores(Path(args.dino_run), task, args.split)
        selected = episodes(args.dataset, args.frames, args.split, task)[:args.per_task]
        if split and args.subset and args.split == 'test':
            selected = [episode for episode in selected if episode.index in split[args.subset][task]]
        selected = [episode for episode in selected if (scores.episode == episode.name).any()]
        camera = tasks_config[task].get('camera', args.camera)
        results = []
        for episode in selected:
            result = replay_episode(task, episode, scores[scores.episode == episode.name], checker, camera, scale,
                                    args.recheck_sec, args.release_delay, args.confirm_sec)
            results.append(result)
            asked = sum(a['subgoal_probability'] is not None for a in result['alarms'])
            log(f"{task} {episode.name}: {len(result['alarms'])} alarms, {asked} asked, "
                f"stop {result['stop_sec']}")
        (output / f'{task}.json').write_text(json.dumps(results, indent=1))
    log(f'VLM questions asked this run: {checker.asked}')
