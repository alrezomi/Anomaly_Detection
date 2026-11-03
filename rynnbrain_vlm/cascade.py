"""DINO + GMR replay that stops for a VLM check when DINO raises an alarm.

DINO (camera) and GMR (TCP position, force limit, gripper release) score the
same execution together. When one DINO camera scores ALARM_RUN consecutive
frames above its threshold (the pooled 3x3 score, the DINO benchmark's
decision score), the replay pauses and the base VLM (no LoRA) is
asked in the main VLM's multi-turn form: turn 1 shows a complete nominal
execution (rynnbrain.reference_bags, num_frames frames, the main VLM's
reference prompt); turn 2 shows raw frames of this execution up to the alarm
and asks whether a failure happened or the robot can safely continue. A failure answer ends the execution; a safe
answer resumes it, and that camera must drop below its threshold before it can
trigger another check. GMR is shown but does not trigger checks yet. The
execution, and with it the replay, ends POST_STAGE3_SEC after stage 3, as for
GMR, so the robot returning home cannot trigger a check. The trigger rule
(AlarmEpisodes) and the question (VlmChecker) are shared with the offline
benchmark in cascade_benchmark.py.
"""
from collections import defaultdict
import heapq
import io
import re
import time

import numpy as np

ALARM_RUN = 3  # Consecutive DINO frames above threshold that stop the execution.
CONTEXT_SEC = 2.5  # Context frames: after stage 1 starts, and before each later stage starts.
MIN_FRAME_SPACING_SEC = 1.0  # An alarm soon after a stage starts gets fewer frames.
MAX_VLM_FRAMES = 8


def select_vlm_frame_times(boundaries, alarm_sec):
    """Causal frame times (<= alarm_sec) and labels for the VLM check.

    One frame CONTEXT_SEC after stage 1 starts and one CONTEXT_SEC before each
    later stage that has started show the earlier progress. The remaining
    frames are spread evenly from the start of the current stage to the alarm,
    at least MIN_FRAME_SPACING_SEC apart, up to MAX_VLM_FRAMES in total.
    """
    started = sorted((stage, start) for stage, start in boundaries.items() if start <= alarm_sec)
    stage, stage_start = started[-1] if started else (0, 0.)
    picks = []
    if stage >= 2:
        context = [(boundaries[1] + CONTEXT_SEC, f'{CONTEXT_SEC:g} s after stage 1 started')]
        context += [(boundaries[s] - CONTEXT_SEC, f'{CONTEXT_SEC:g} s before stage {s} started')
                    for s in range(2, stage + 1)]
        for t, label in context:
            t = min(max(t, boundaries[1]), stage_start)
            if not picks or t - picks[-1][0] >= MIN_FRAME_SPACING_SEC:
                picks.append((t, label))
    span = alarm_sec - stage_start
    count = min(MAX_VLM_FRAMES - len(picks), int(span / MIN_FRAME_SPACING_SEC) + 1)
    times = np.linspace(stage_start, alarm_sec, max(count, 1)) if count > 1 else [alarm_sec]
    for t in times:
        label = 'alarm' if t == alarm_sec else f'stage {stage}, {t - stage_start:.1f} s after it started'
        picks.append((float(t), label))
    return picks


def current_stage(boundaries, timestamp):
    return max((stage for stage, start in boundaries.items() if start <= timestamp), default=0)


def parse_verdict(answer):
    """'failure' or 'safe'. Anything unclear stops the robot."""
    match = re.search(r'^\s*Decision\s*:\s*(.+)$', answer, flags=re.IGNORECASE | re.MULTILINE)
    decision = (match.group(1) if match else answer).lower()
    return 'safe' if 'safe' in decision and 'fail' not in decision else 'failure'


class AlarmEpisodes:
    """ALARM_RUN consecutive frames above threshold in one camera start an alarm
    episode; that camera must drop below its threshold before the next one."""
    def __init__(self):
        self.streaks = defaultdict(int)
        self.armed = defaultdict(lambda: True)

    def update(self, camera, above):
        """Record one frame; True when it starts a new alarm episode."""
        self.streaks[camera] = self.streaks[camera] + 1 if above else 0
        if not above:
            self.armed[camera] = True
        if self.streaks[camera] >= ALARM_RUN and self.armed[camera]:
            self.armed[camera] = False
            return True
        return False


class VlmChecker:
    """The base VLM (no LoRA), asked in the main VLM's multi-turn form.

    Turn 1 is a complete nominal execution (rynnbrain.reference_bags,
    num_frames frames, the main VLM's reference prompt); the model's answer to
    it is generated once. Turn 2 is frames of the current execution up to the
    alarm, with the stage-aware question.
    """
    def __init__(self, config, progress):
        from pathlib import Path
        from .model import RynnBrainModel
        from .prompts import task_context_prompt
        from .run import _raw_inputs
        from .temporal_lora import timestamped_images
        vlm = config['rynnbrain']
        self.task = vlm.get('task_description', 'Robot manipulation task')
        self.generation = vlm.get('generation', {})
        self.topic = (config.get('camera_topics') or vlm['camera_topics'])[0]
        progress('Loading the base VLM (no LoRA) for alarm checks')
        self.model = RynnBrainModel({**vlm['model'], 'lora_adapter_path': None})
        # Turn 1 exactly as the main online VLM builds it: a complete nominal execution.
        progress('Reading the nominal reference execution for the VLM')
        topics = vlm.get('memory_camera_topics') or [self.topic]
        frame_count = int(vlm.get('num_frames', 8))
        self.reference_bags = list(vlm.get('reference_bags', []))
        self.reference_names = {Path(bag).name for bag in self.reference_bags}
        reference = []
        for bag in self.reference_bags:
            images, metadata = _raw_inputs(Path(bag), topics, frame_count)
            reference.extend(timestamped_images(images, metadata, topics, frame_count))
        if not reference:
            raise ValueError('The VLM check needs rynnbrain.reference_bags (a complete nominal execution).')
        self.reference_turn = dict(role='user', images=reference, text=task_context_prompt(
            self.task, temporal=True, variant=vlm.get('prompt_variant', 'A')))
        progress('Generating the VLM response to the nominal reference')
        self.reference_response = self.model.generate_nominal(self.reference_turn, self.generation)
        self.signature = dict(vlm_model=vlm['model']['model_id'], vlm_camera=self.topic,
                              reference_bags=self.reference_bags, reference_response=self.reference_response,
                              alarm_run=ALARM_RUN, context_sec=CONTEXT_SEC,
                              min_frame_spacing_sec=MIN_FRAME_SPACING_SEC, max_vlm_frames=MAX_VLM_FRAMES)

    def ask(self, frames, boundaries, alarm_sec):
        """Ask whether a failure happened by alarm_sec.

        frames: time-ordered (timestamp, (source, payload)) received so far;
        source is 'message' (a compressed ROS image) or 'image' (a PIL image).
        Returns the check record and the labelled images that were sent.
        """
        from rosbag_io import decode_image_message
        from .prompts import cascade_verification_prompt
        stage = current_stage(boundaries, alarm_sec)
        stamps = [t for t, _ in frames]
        records, images, used = [], [], set()
        for requested, label in select_vlm_frame_times(boundaries, alarm_sec):
            # Latest camera frame at or before the requested time; never a later one, never twice.
            position = int(np.searchsorted(stamps, requested + 1e-9, side='right')) - 1
            if position < 0 or position in used:
                continue
            used.add(position)
            frame_sec, (source, payload) = frames[position]
            image = decode_image_message(*payload) if source == 'message' else payload
            images.append((f'Frame {len(images) + 1}, camera {self.topic}, time {frame_sec:.3f} s '
                           f'from this bag start ({label}):', image))
            records.append(dict(label=label, requested_sec=requested, frame_sec=frame_sec))
        if not images:
            raise ValueError('No camera frames were received before the DINO alarm; cannot ask the VLM.')
        prompt = cascade_verification_prompt(self.task, stage)
        started = time.monotonic()
        _, answer = self.model.generate_multiturn(
            [self.reference_turn, dict(role='user', images=images, text=prompt)], self.generation,
            reference_response=self.reference_response)
        check = dict(stage=stage, frames=records, camera_topic=self.topic, reference_bags=self.reference_bags,
                     reference_frames=len(self.reference_turn['images']), prompt=prompt, answer=answer,
                     verdict=parse_verdict(answer), vlm_sec=time.monotonic() - started)
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
        self.excluded = self.dino.excluded | self.gmr.excluded | self.checker.reference_names
        self.signal_mode = self.gmr.signal_mode
        self.signature = dict(protocol='dino_gmr_vlm_check_v1', dino=self.dino.signature,
                              dino_trigger_score='pooled', **self.checker.signature)
        self.note = (f'DINO and GMR score the execution together. {ALARM_RUN} consecutive DINO frames above '
                     'threshold (pooled 3x3 score) stop it and ask the base VLM: turn 1 shows a complete nominal '
                     'execution, turn 2 frames of this execution up to the alarm. A safe answer resumes the replay. '
                     'GMR does not trigger checks yet.')

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
        self.queries, self.vlm_frame_jpegs, self.frames = [], {}, []
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
        if self.episodes.update(topic, bool(row['alarm_pooled'])):
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
        from rosbag_io import decode_image_message
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
                detector.frames.append((timestamp, ('message', payload)))  # Kept compressed for the VLM check.
                self.publish(decode_image_message(*payload), timestamp)
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
                if topic == detector.vlm_topic and (not detector.frames or detector.frames[-1][0] < timestamp - .25):
                    # The 15 fps camera stream has ended with the GMR stream; keep DINO's frame instead.
                    preview = image.copy()
                    size = detector.checker.model.max_image_size
                    preview.thumbnail((size, size))
                    detector.frames.append((timestamp, ('image', preview)))
                detector.pending = ('dino', topic)
                return (topic, frame_id), timestamp, image
        return None

    def close(self):
        for source in self.sources:
            if hasattr(source, 'close'):
                source.close()
