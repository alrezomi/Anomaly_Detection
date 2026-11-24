"""DINO + GMR replay that stops for a VLM check when DINO raises an alarm.

DINO (camera) and GMR (TCP position, force limit, gripper release) score the
same execution together. When one DINO camera scores ALARM_RUN consecutive
frames above its threshold (the pooled 3x3 score, the DINO benchmark's
decision score), the replay pauses and the base VLM (no LoRA) checks, on the
gripper camera (rynnbrain.vlm_check), the sub-goal the robot has reported as
reached: after the grasp (stage 2) the object must be held between the
gripper fingers, after the release (stage 3) it must lie at the destination
below the gripper. Before the grasp (stage 1) no sub-goal is reached yet, so
the check answers safe without asking. A failure answer ends the execution; a
safe answer resumes it, and that camera must drop below its threshold, or the
stage must change, before it can trigger another check. The execution, and with
it the replay, ends POST_STAGE3_SEC after stage 3, as for GMR, so the robot
returning home cannot trigger a check.

With rynnbrain.ts_check.enabled, TS alarms ask the VLM too (TsAlarmEpisodes):
each GMR alarm source (position GMR, force safety limit, empty grip, other GMR
sensors) when its alarm starts and again every TS_RECHECK_SEC while it lasts,
and the release-position check RELEASE_CHECK_DELAY_SEC after the gripper opened.
The VLM answers a question about the visible state that alarm concerns
(prompts.ts_alarm_questions) and the execution stops only if the image shows a
failure; otherwise the alarm is ignored. The trigger rules (AlarmEpisodes,
TsAlarmEpisodes) and the check (VlmChecker) are shared with the offline
benchmark in cascade_benchmark.py.
"""
from collections import defaultdict, deque
import heapq
import io
import time

import numpy as np

ALARM_RUN = 3  # Consecutive DINO frames above threshold that stop the execution.
CHECK_OFFSETS_SEC = (0., .5, 1.)  # Gripper-camera frames this long before the alarm.
SUBGOAL_THRESHOLD = .5  # Below this mean probability that the sub-goal holds, the check answers failure.
FRAME_BUFFER = 150  # Gripper-camera frames kept for the check (10 s at 15 fps).
SUBGOALS = {2: 'the object is held between the gripper fingers',
            3: 'the object lies at the destination below the gripper'}
TS_RECHECK_SEC = 2.  # A TS alarm that lasts is checked again this often...
TS_MAX_RECHECKS = 5  # ...at most this many times per alarm episode.
RELEASE_CHECK_DELAY_SEC = 1.  # A release-position alarm is checked once the object has had time to settle.
TS_ALARM_NAMES = {'position': 'TCP position (GMR)', 'force': 'force safety limit', 'gripper_empty': 'empty grip',
                  'release': 'release position', 'gripper': 'gripper (GMR)', 'joints': 'joint positions (GMR)'}


def current_stage(boundaries, timestamp):
    return max((stage for stage, start in boundaries.items() if start <= timestamp), default=0)


class AlarmEpisodes:
    """ALARM_RUN consecutive frames above threshold in one camera start an alarm
    episode; that camera must drop below its threshold, or the stage must change,
    before the next one. A new stage changes what should be true, so an alarm
    that persists into it is checked again."""
    def __init__(self):
        self.streaks = defaultdict(int)
        self.armed = defaultdict(lambda: True)
        self.stages = {}

    def update(self, camera, above, stage=0):
        """Record one frame; True when it starts a new alarm episode."""
        if self.stages.setdefault(camera, stage) != stage:
            self.stages[camera], self.streaks[camera], self.armed[camera] = stage, 0, True
        self.streaks[camera] = self.streaks[camera] + 1 if above else 0
        if not above:
            self.armed[camera] = True
        if self.streaks[camera] >= ALARM_RUN and self.armed[camera]:
            self.armed[camera] = False
            return True
        return False


class TsAlarmEpisodes:
    """When TS alarms ask the VLM.

    A source's (filtered) alarm starts an episode and is checked; while it lasts
    it is checked again every TS_RECHECK_SEC, at most TS_MAX_RECHECKS times; it
    is re-armed when it drops or the stage changes. Re-checks matter: 0.4 s after
    an empty grasp the missed instrument still lies just below the fingers and
    looks held; seconds later it is clearly left behind. A release-position alarm
    is checked once, RELEASE_CHECK_DELAY_SEC after the opening was confirmed.
    """
    def __init__(self):
        self.active = {}
        self.release_at, self.release_done = None, False

    def update(self, sensors, stage, timestamp, release=None):
        """[(source, recheck number)] due at this GMR sample.

        sensors: the GMR row's per-source results (each with 'alarm'); release:
        the release check result (GmrOnlineDetector.release_result) or None.
        """
        due = []
        for name, result in sensors.items():
            state = self.active.get(name)
            if not result.get('alarm'):
                self.active.pop(name, None)
            elif state is None or state['stage'] != stage:
                self.active[name] = dict(stage=stage, last=timestamp, rechecks=0)
                due.append((name, 0))
            elif timestamp - state['last'] >= TS_RECHECK_SEC - 1e-6 and state['rechecks'] < TS_MAX_RECHECKS:
                state['last'], state['rechecks'] = timestamp, state['rechecks'] + 1
                due.append((name, state['rechecks']))
        if (release and release.get('status') == 'released' and release.get('alarm')
                and self.release_at is None and not self.release_done):
            self.release_at = release['time_sec'] + RELEASE_CHECK_DELAY_SEC
        if self.release_at is not None and timestamp >= self.release_at - 1e-9:
            self.release_at, self.release_done = None, True
            due.append(('release', 0))
        return due


def group_ts_alarms(due, stage, item, destination):
    """Split the TS alarms due at one sample into checks: alarms about the same sub-goal share one check.

    Before the grasp a position or force alarm has no visible sub-goal: it is recorded once (not asked, like a
    DINO alarm then) and its re-checks are skipped.
    """
    from .prompts import ts_alarm_questions
    groups = {}
    for source, recheck in due:
        goal, questions = ts_alarm_questions(source, stage, item, destination)
        if questions is None and recheck:
            continue
        groups.setdefault(goal if questions else None, []).append((source, recheck))
    return list(groups.values())


def ts_trigger(alarms, timestamp, stage, details=None):
    return dict(source='ts', time_sec=timestamp, stage=stage, alarms=[a for a, _ in alarms],
                alarm_names=[TS_ALARM_NAMES.get(a, a) for a, _ in alarms],
                rechecks=[r for _, r in alarms], details=details or {})


class VlmChecker:
    """The base VLM (no LoRA) checks the sub-goal of the alarm's stage on the gripper camera.

    Each question of prompts.subgoal_questions is asked about each
    gripper-camera frame CHECK_OFFSETS_SEC before the alarm; the mean
    probability of the answer meaning that the sub-goal holds decides.
    rynnbrain.vlm_check names the camera, how the object and the destination
    look from it, and optional model overrides.
    """
    def __init__(self, config, progress):
        from .model import RynnBrainModel
        vlm = config['rynnbrain']
        settings = vlm.get('vlm_check') or {}
        missing = [key for key in ('camera_topic', 'item', 'destination') if not settings.get(key)]
        if missing:
            raise ValueError(f"The VLM check needs rynnbrain.vlm_check.{', '.join(missing)}.")
        self.topic, self.item, self.destination = settings['camera_topic'], settings['item'], settings['destination']
        progress('Loading the base VLM (no LoRA) for alarm checks')
        # On Jetson Thor device_map "auto" sees only free, not reclaimable, memory and
        # offloads layers to the CPU; vlm_check.model can pin the model to the GPU.
        self.model = RynnBrainModel({**vlm['model'], 'lora_adapter_path': None, **settings.get('model', {})})
        self.signature = dict(vlm_model=vlm['model']['model_id'], vlm_camera=self.topic, item=self.item,
                              destination=self.destination, alarm_run=ALARM_RUN,
                              check_offsets_sec=list(CHECK_OFFSETS_SEC), subgoal_threshold=SUBGOAL_THRESHOLD)

    def ask(self, frames, boundaries, alarm_sec, alarm=None):
        """Check the sub-goal of the stage at alarm_sec (a DINO alarm), or the one TS alarm `alarm` concerns.

        frames: time-ordered (timestamp, (source, payload)) gripper-camera frames
        received so far; source is 'message' (a compressed ROS image) or 'image'
        (a PIL image). Returns the check record and the labelled images asked about.
        """
        from rosbag_io import decode_image_message
        from .prompts import subgoal_questions, ts_alarm_questions
        stage = current_stage(boundaries, alarm_sec)
        goal, questions = ((stage, subgoal_questions(stage, self.item, self.destination)) if alarm is None else
                           ts_alarm_questions(alarm, stage, self.item, self.destination))
        if questions is None:
            return dict(stage=stage, goal_stage=goal, frames=[], camera_topic=self.topic, questions=[], scores=[],
                        subgoal_probability=None, verdict='safe', vlm_sec=0.,
                        answer=f'Stage {stage}: no grasp reported yet, so no sub-goal to check. Safe to continue.'), []
        stamps = [t for t, _ in frames]
        records, images, used = [], [], set()
        for offset in CHECK_OFFSETS_SEC:
            # Latest camera frame at or before the requested time; never a later one, never twice.
            position = int(np.searchsorted(stamps, alarm_sec - offset + 1e-9, side='right')) - 1
            if position < 0 or position in used:
                continue
            used.add(position)
            frame_sec, (source, payload) = frames[position]
            label = 'alarm' if offset == 0 else f'{offset:g} s before the alarm'
            images.append((label, decode_image_message(*payload) if source == 'message' else payload))
            records.append(dict(label=label, requested_sec=alarm_sec - offset, frame_sec=frame_sec))
        if not images:
            raise ValueError('No gripper-camera frames were received before the alarm; cannot ask the VLM.')
        started = time.monotonic()
        scores = [[self.model.answer_probability(dict(role='user', images=[('Gripper camera image:', image)],
                                                      text=question), options) for _, image in images]
                  for question, options in questions]
        probability = float(np.mean(scores))
        verdict = 'failure' if probability < SUBGOAL_THRESHOLD else 'safe'
        answer = '\n'.join([f'Stage {stage}' + (f', checking: {SUBGOALS[goal]}.' if goal != stage else
                                                f' sub-goal: {SUBGOALS[goal]}.'),
                            *(f"P({options[0]}) of the {'/'.join(options)} question per frame: "
                              + ', '.join(f'{p:.2f}' for p in values) for (_, options), values in zip(questions, scores)),
                            f'Mean probability that the sub-goal holds: {probability:.3f} '
                            + (f'< {SUBGOAL_THRESHOLD:g}: failure.' if verdict == 'failure'
                               else f'>= {SUBGOAL_THRESHOLD:g}: safe to continue.')])
        check = dict(stage=stage, goal_stage=goal, frames=records, camera_topic=self.topic,
                     questions=[q for q, _ in questions],
                     scores=scores, subgoal_probability=probability, answer=answer, verdict=verdict,
                     vlm_sec=time.monotonic() - started)
        return check, images


class CascadeDetector:
    kind = 'cascade'
    knn = True
    capture_visual = False
    frame_count = None
    start_sec = 0.
    end_sec = None
    score_kind = 'dino_gmr_cascade'
    score_source = 'dino_gmr_with_vlm_check'
    preview_image = matched_image = None

    def __init__(self, config, progress):
        from .dino_cameras import DinoCameraReplay
        from time_series_gmr_scripts.online_gmr import GmrOnlineDetector
        self.config = config
        self.dino = DinoCameraReplay(config, progress)
        self.gmr = GmrOnlineDetector(config, progress)
        self.cameras = self.dino.cameras
        self.sample_fps = self.dino.sample_fps
        self.path = self.gmr.path.parent / 'cascade'
        self.checker = VlmChecker(config, progress)
        self.vlm_topic = self.checker.topic
        self.excluded = self.dino.excluded | self.gmr.excluded
        self.signal_mode = self.gmr.signal_mode
        self.ts_check = bool((config['rynnbrain'].get('ts_check') or {}).get('enabled', False))
        self.signature = dict(protocol='dino_gmr_ts_vlm_check_v3' if self.ts_check else 'dino_gmr_vlm_subgoal_check_v2',
                              dino=self.dino.signature, dino_trigger_score='pooled', ts_check=self.ts_check,
                              **self.checker.signature)
        if self.ts_check:
            self.signature.update(ts_recheck_sec=TS_RECHECK_SEC, ts_max_rechecks=TS_MAX_RECHECKS,
                                  release_check_delay_sec=RELEASE_CHECK_DELAY_SEC)
        self.note = (f'DINO and GMR score the execution together. {ALARM_RUN} consecutive DINO frames above '
                     'threshold (pooled 3x3 score) stop it; the base VLM then checks on the gripper camera the '
                     'sub-goal of the current stage (object held after the grasp, object at the destination after '
                     'the release). A safe answer resumes the replay. '
                     + ('TS alarms (position, force limit, empty grip, release position) also ask the VLM, about '
                        'the state the alarm concerns, when they start and every '
                        f'{TS_RECHECK_SEC:g} s while they last; the VLM ignores the alarm unless the image shows a '
                        'failure.' if self.ts_check else 'GMR does not trigger checks (rynnbrain.ts_check is off).'))

    # The replay app reads the GMR stage clock and release result through these.
    @property
    def sampler(self):
        return self.gmr.sampler

    @property
    def release_result(self):
        return self.gmr.release_result

    def catalogue(self):
        return self.dino.catalogue()

    def reset(self, bag):
        self.dino.reset(bag)
        self.gmr.signal_mode = self.signal_mode
        self.gmr.reset(bag)
        self.bag = bag
        self.episodes, self.ts_episodes = AlarmEpisodes(), TsAlarmEpisodes()
        self.queries, self.vlm_frame_jpegs, self.frames = [], {}, deque(maxlen=FRAME_BUFFER)
        self.halted, self.termination_reason, self.output_dir = False, None, None

    def prepare_replay(self, progress, stop):
        prepared = self.gmr.prepare_replay(progress, stop)
        self.signature['gmr'] = self.gmr.signature
        return dict(prepared, protocol=self.signature['protocol'])

    def create_feed(self, publish, stop, update):
        from rosbag_io import iter_ts_replay_events
        from time_series_gmr_scripts.online_gmr import CausalSamples
        from time_series_gmr_scripts import run_time_series_test as settings
        gmr = self.gmr
        gmr.sampler = CausalSamples(gmr.sample_kinds, stop, release=gmr.release_check, post_sec=settings.POST_STAGE3_SEC)
        gmr.sampler.columns, gmr.sampler.sensors, gmr.sampler.pose_frame = gmr.columns, gmr.sensors, gmr.pose_frame
        events = iter_ts_replay_events(self.bag, gmr.topics, gmr.stage_topic, self.vlm_topic)
        self.update = update
        self.feed = CascadeFeed(self, gmr.sampler.packets(events), dino_packets(self.dino), publish, stop, update)
        return self.feed

    def append(self, image, timestamp, verify=False):
        kind, payload = self.pending
        if kind == 'gmr':
            self.gmr.pending = payload
            row, vector, baseline = self.gmr.append(image, timestamp, verify)
            row['vlm_check'] = None  # Same columns in every GMR row; filled when this sample triggers a check.
            if self.ts_check and not self.halted:
                stage = current_stage(self.sampler.clock.boundaries, timestamp)
                due = self.ts_episodes.update(row['sensors'], stage, timestamp, self.gmr.release_result)
                if due:
                    row['vlm_check'] = self.check_ts_alarms(due, timestamp, row)
            return row, vector, baseline
        topic = payload
        row, vector, baseline = self.cameras[topic].append(image, timestamp, verify)
        row['vlm_check'] = None  # Same columns in every DINO row; filled when this frame triggers a check.
        # The pooled (3x3) score decides, as in the DINO benchmark.
        stage = current_stage(self.sampler.clock.boundaries, timestamp)
        if self.episodes.update(topic, bool(row['alarm_pooled']), stage):
            row['vlm_check'] = self.check_with_vlm(topic, timestamp, row)
        return row, vector, baseline

    def check_with_vlm(self, topic, timestamp, row):
        """A DINO alarm episode: check the sub-goal of the current stage."""
        boundaries = dict(self.sampler.clock.boundaries)
        trigger = dict(source='dino', time_sec=timestamp, camera_topic=topic, stage=current_stage(boundaries, timestamp),
                       dino_score=row['score_pooled'], dino_threshold=row['threshold_pooled'],
                       consecutive_frames=self.episodes.streaks[topic])
        return self.ask_vlm(trigger, boundaries)

    def check_ts_alarms(self, due, timestamp, row):
        """TS alarms due at this GMR sample: one check per sub-goal they concern; stops at a failure."""
        boundaries = dict(self.sampler.clock.boundaries)
        stage = current_stage(boundaries, timestamp)
        query = None
        for alarms in group_ts_alarms(due, stage, self.checker.item, self.checker.destination):
            details = {}
            for source, recheck in alarms:
                result = (self.gmr.release_result if source == 'release' else row['sensors'].get(source)) or {}
                details[source] = dict(recheck=recheck, value=result.get('distance'), limit=result.get('threshold'))
            trigger = ts_trigger(alarms, timestamp, stage, details)
            query = self.ask_vlm(trigger, boundaries, alarm=alarms[0][0])
            if query['verdict'] == 'failure':
                break
        return query

    def ask_vlm(self, trigger, boundaries, alarm=None):
        """Pause the replay, ask the VLM, keep the record and the frames, stop on a failure answer."""
        paused = time.monotonic()
        index = len(self.queries)
        timestamp = trigger['time_sec']
        self.update(vlm_status=dict(state='asking', **trigger))
        check, images = self.checker.ask(self.frames, boundaries, timestamp, alarm=alarm)
        for k, (_, image) in enumerate(images):
            preview = image.copy()
            preview.thumbnail((480, 360))
            buffer = io.BytesIO()
            preview.save(buffer, format='JPEG', quality=85)
            self.vlm_frame_jpegs[index, k] = buffer.getvalue()
            if self.output_dir is not None:
                image.save(self.output_dir / f'vlm_check_{index + 1}_frame_{k + 1}.jpg')
        query = dict(index=index, trigger=trigger, **check)
        self.queries.append(query)
        if query['verdict'] == 'failure':
            self.halted, self.termination_reason = True, 'vlm_confirmed_failure'
        self.update(vlm_status=dict(state='answered', **trigger), vlm_queries=self.queries)
        # Replay time stood still while the VLM answered.
        self.feed.origin += time.monotonic() - paused
        return query


def dino_packets(dino):
    frames = dino.iter_frames()
    try:
        for frame_id, timestamp, image in frames:
            yield timestamp, 'dino', (frame_id, image)
    finally:
        frames.close()


class CascadeFeed:
    """Pace camera frames, stage markers, GMR samples and DINO frames in bag time."""
    def __init__(self, detector, gmr_packets, dino_frames, publish, stop, update, clock=time.monotonic):
        from time_series_gmr_scripts import run_time_series_test as settings
        self.detector, self.post_sec = detector, settings.POST_STAGE3_SEC
        self.packets = heapq.merge(gmr_packets, dino_frames, key=lambda packet: packet[0])
        self.sources = (gmr_packets, dino_frames)
        self.publish, self.stop, self.update, self.clock = publish, stop, update, clock
        self.origin = clock()
        self.index = 0

    def run(self):
        pass

    def next_frame(self, earliest):
        detector = self.detector
        while not self.stop.is_set() and not detector.halted:
            packet = next(self.packets, None)
            boundaries = detector.sampler.clock.boundaries
            if packet is None or (3 in boundaries and packet[0] > boundaries[3] + self.post_sec):
                # End of the recording or of the execution (stage 3 plus the GMR window).
                if packet is not None and detector.termination_reason is None:
                    detector.termination_reason = 'execution_end'
                sampler = detector.sampler
                if detector.gmr.release_check and sampler.release is None and sampler.clock.stage == 3:
                    self.update(gmr_release=detector.gmr.check_release(None))
                return None
            timestamp, kind, payload = packet
            if self.stop.wait(max(0., self.origin + timestamp - self.clock())):
                return None
            if kind == 'camera':
                detector.frames.append((timestamp, ('message', payload)))  # Gripper camera, compressed, for the VLM check.
            elif kind == 'stage':
                self.update(gmr_stage=payload)
            elif kind == 'release':
                self.update(gmr_release=detector.gmr.check_release(payload))
            elif kind == 'sample':
                detector.pending = ('gmr', payload)
                self.index += 1
                return self.index - 1, timestamp, None
            elif kind == 'dino':
                (topic, frame_id), image = payload
                self.publish(image, timestamp)  # Live view of the DINO camera panel.
                detector.pending = ('dino', topic)
                return (topic, frame_id), timestamp, image
        return None

    def close(self):
        for source in self.sources:
            if hasattr(source, 'close'):
                source.close()
