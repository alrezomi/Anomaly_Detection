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
stage must change, before it can trigger another check. GMR is shown but does
not trigger checks yet. The execution, and with it the replay, ends
POST_STAGE3_SEC after stage 3, as for GMR, so the robot returning home cannot
trigger a check. The trigger rule (AlarmEpisodes) and the check (VlmChecker)
are shared with the offline benchmark in cascade_benchmark.py.
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

    def ask(self, frames, boundaries, alarm_sec):
        """Check the sub-goal of the stage at alarm_sec.

        frames: time-ordered (timestamp, (source, payload)) gripper-camera frames
        received so far; source is 'message' (a compressed ROS image) or 'image'
        (a PIL image). Returns the check record and the labelled images asked about.
        """
        from rosbag_io import decode_image_message
        from .prompts import subgoal_questions
        stage = current_stage(boundaries, alarm_sec)
        questions = subgoal_questions(stage, self.item, self.destination)
        if questions is None:
            return dict(stage=stage, frames=[], camera_topic=self.topic, questions=[], scores=[],
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
            raise ValueError('No gripper-camera frames were received before the DINO alarm; cannot ask the VLM.')
        started = time.monotonic()
        scores = [[self.model.answer_probability(dict(role='user', images=[('Gripper camera image:', image)],
                                                      text=question), options) for _, image in images]
                  for question, options in questions]
        probability = float(np.mean(scores))
        verdict = 'failure' if probability < SUBGOAL_THRESHOLD else 'safe'
        answer = '\n'.join([f'Stage {stage} sub-goal: {SUBGOALS[stage]}.',
                            *(f"P({options[0]}) of the {'/'.join(options)} question per frame: "
                              + ', '.join(f'{p:.2f}' for p in values) for (_, options), values in zip(questions, scores)),
                            f'Mean probability that the sub-goal holds: {probability:.2f} '
                            + (f'< {SUBGOAL_THRESHOLD:g}: failure.' if verdict == 'failure'
                               else f'>= {SUBGOAL_THRESHOLD:g}: safe to continue.')])
        check = dict(stage=stage, frames=records, camera_topic=self.topic, questions=[q for q, _ in questions],
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
        self.signature = dict(protocol='dino_gmr_vlm_subgoal_check_v2', dino=self.dino.signature,
                              dino_trigger_score='pooled', **self.checker.signature)
        self.note = (f'DINO and GMR score the execution together. {ALARM_RUN} consecutive DINO frames above '
                     'threshold (pooled 3x3 score) stop it; the base VLM then checks on the gripper camera the '
                     'sub-goal of the current stage (object held after the grasp, object at the destination after '
                     'the release). A safe answer resumes the replay. GMR does not trigger checks yet.')

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
        self.episodes = AlarmEpisodes()
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
            return self.gmr.append(image, timestamp, verify)
        topic = payload
        row, vector, baseline = self.cameras[topic].append(image, timestamp, verify)
        row['vlm_check'] = None  # Same columns in every DINO row; filled when this frame triggers a check.
        # The pooled (3x3) score decides, as in the DINO benchmark.
        stage = current_stage(self.sampler.clock.boundaries, timestamp)
        if self.episodes.update(topic, bool(row['alarm_pooled']), stage):
            row['vlm_check'] = self.check_with_vlm(topic, timestamp, row)
        return row, vector, baseline

    def check_with_vlm(self, topic, timestamp, row):
        paused = time.monotonic()
        boundaries = dict(self.sampler.clock.boundaries)
        index = len(self.queries)
        trigger = dict(time_sec=timestamp, camera_topic=topic, stage=current_stage(boundaries, timestamp),
                       dino_score=row['score_pooled'], dino_threshold=row['threshold_pooled'],
                       consecutive_frames=self.episodes.streaks[topic])
        self.update(vlm_status=dict(state='asking', **trigger))
        check, images = self.checker.ask(self.frames, boundaries, timestamp)
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
