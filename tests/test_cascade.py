from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from rynnbrain_vlm.cascade import AlarmEpisodes, CascadeDetector, TsAlarmEpisodes, VlmChecker


def cascade(above, probability=.2, unpooled=None, boundaries=None):
    """Detector with one fake DINO camera whose pooled scores are above threshold as listed.

    probability: what the fake VLM answers for every sub-goal question.
    """
    detector = CascadeDetector.__new__(CascadeDetector)
    unpooled = above if unpooled is None else unpooled
    rows = iter(dict(anomaly_score=2. if u else .5, threshold=1., above_threshold=u,
                     score_pooled=2. if a else .5, threshold_pooled=1., alarm_pooled=a) for a, u in zip(above, unpooled))
    detector.cameras = {'top': SimpleNamespace(append=lambda *_: (next(rows), None, None))}
    checker = detector.checker = VlmChecker.__new__(VlmChecker)
    checker.topic, checker.item, checker.destination = 'gripper', 'a metal surgical instrument', 'a blue surface'
    checker.model = SimpleNamespace(answer_probability=Mock(return_value=probability), max_image_size=640)
    detector.vlm_topic = 'gripper'
    clock = SimpleNamespace(boundaries=boundaries or {1: 0., 2: 15.})
    detector.gmr = SimpleNamespace(sampler=SimpleNamespace(clock=clock))
    detector.update, detector.feed = Mock(), SimpleNamespace(origin=0.)
    detector.episodes = AlarmEpisodes()
    detector.queries, detector.vlm_frame_jpegs = [], {}
    detector.halted, detector.termination_reason, detector.output_dir = False, None, None
    detector.frames = [(t / 10, ('message', ('message', t / 10))) for t in range(0, 400, 2)]  # 5 fps gripper camera
    return detector


def ts_cascade(samples, probability=.2, boundaries=None, release=None):
    """Cascade with TS checks on; the fake GMR returns rows whose sensors alarm as listed per sample."""
    detector = cascade([], probability=probability, boundaries=boundaries or {1: 0., 2: 15.})
    detector.ts_check, detector.ts_episodes = True, TsAlarmEpisodes()
    rows = iter(dict(sensors={name: dict(alarm=on, distance=1., threshold=1.) for name, on in alarms.items()},
                     alarm_sources=[name for name, on in alarms.items() if on]) for alarms in samples)
    detector.gmr.append = lambda *_: (next(rows), None, None)
    detector.gmr.release_result = release
    return detector


class CascadeTests(unittest.TestCase):
    def test_ts_alarms_are_checked_at_start_and_again_while_they_last(self):
        rule = TsAlarmEpisodes()
        due = [rule.update({'position': {'alarm': t < 13}}, 2 if t < 12 else 3, float(t)) for t in range(30)]
        # Start, re-checks every 2 s, a new episode on the stage change, nothing once the alarm dropped.
        self.assertEqual({t: d for t, d in enumerate(due) if d},
                         {0: [('position', 0)], 2: [('position', 1)], 4: [('position', 2)], 6: [('position', 3)],
                          8: [('position', 4)], 10: [('position', 5)], 12: [('position', 0)]})
        rule = TsAlarmEpisodes()
        release = dict(status='released', alarm=True, time_sec=40.)
        due = [rule.update({}, 3, t, release) for t in (40.5, 41., 41.5)]
        self.assertEqual(due, [[], [('release', 0)], []])  # Once, 1 s after the opening.

    def test_empty_grip_asks_whether_the_object_is_held_even_before_the_grasp_marker(self):
        detector = ts_cascade([{'gripper_empty': True}], boundaries={1: 0., 2: 30.})
        with patch('rosbag_io.decode_image_message', side_effect=lambda _m, t: Image.new('RGB', (8, 8))):
            detector.pending = ('gmr', {})
            row, _, _ = detector.append(None, 20.)
        check = row['vlm_check']
        self.assertEqual((check['trigger']['source'], check['trigger']['alarms'], check['stage'], check['goal_stage']),
                         ('ts', ['gripper_empty'], 1, 2))
        self.assertIn('held between the two white fingers', check['questions'][0])
        self.assertEqual((check['verdict'], detector.halted, detector.termination_reason),
                         ('failure', True, 'vlm_confirmed_failure'))

    def test_a_ts_alarm_is_ignored_when_the_image_shows_the_sub_goal_holds(self):
        detector = ts_cascade([{'position': True, 'force': False}] * 3, probability=.9)
        with patch('rosbag_io.decode_image_message', side_effect=lambda _m, t: Image.new('RGB', (8, 8))):
            for t in (20., 20.1, 20.2):
                detector.pending = ('gmr', {})
                row, _, _ = detector.append(None, t)
        self.assertEqual(len(detector.queries), 1)  # Only at the start; the re-check comes 2 s later.
        self.assertEqual((detector.queries[0]['verdict'], detector.halted), ('safe', False))
        self.assertIsNone(row['vlm_check'])

    def test_a_position_alarm_before_the_grasp_is_noted_but_not_asked(self):
        detector = ts_cascade([{'position': True}] * 30, boundaries={1: 0., 2: 30.})
        for t in range(30):
            detector.pending = ('gmr', {})
            detector.append(None, 20. + t / 10)
        self.assertEqual(len(detector.queries), 1)
        self.assertEqual((detector.queries[0]['verdict'], detector.queries[0]['questions']), ('safe', []))
        detector.checker.model.answer_probability.assert_not_called()

    def test_ts_checks_are_off_unless_configured(self):
        detector = ts_cascade([{'gripper_empty': True}])
        detector.ts_check = False
        detector.pending = ('gmr', {})
        row, _, _ = detector.append(None, 20.)
        self.assertIsNone(row['vlm_check'])
        self.assertEqual(detector.queries, [])

    def test_a_stage_change_rearms_a_persisting_alarm(self):
        episodes = AlarmEpisodes()
        stages = [2] * 5 + [3] * 4
        started = [episodes.update('top', True, stage) for stage in stages]
        # Three in a row start an episode; the persisting alarm is checked again three frames into stage 3.
        self.assertEqual(started, [False, False, True, False, False, False, False, True, False])

    def test_stage_one_alarms_continue_without_asking_the_vlm(self):
        detector = cascade([True] * 3, boundaries={1: 0., 2: 30.})
        for t in (20., 20.5, 21.):
            detector.pending = ('dino', 'top')
            row, _, _ = detector.append(None, t)
        check = row['vlm_check']
        self.assertEqual((check['stage'], check['verdict'], check['frames']), (1, 'safe', []))
        detector.checker.model.answer_probability.assert_not_called()
        self.assertFalse(detector.halted)

    def test_three_frames_in_a_row_trigger_once_per_alarm_episode(self):
        detector = cascade([True, True, True, True, False, True, True, True])
        with patch.object(CascadeDetector, 'check_with_vlm', return_value={'verdict': 'safe'}) as check:
            rows = []
            for t in range(8):
                detector.pending = ('dino', 'top')
                rows.append(detector.append(None, 20. + t * .5)[0])
        self.assertEqual([c.args[1] for c in check.call_args_list], [21., 23.5])
        # Every DINO row has the same columns, so the score CSV accepts triggering rows.
        self.assertEqual(len({tuple(row) for row in rows}), 1)
        self.assertEqual([row['vlm_check'] is not None for row in rows], [False, False, True, False, False, False, False, True])

    def test_trigger_follows_the_pooled_score_like_the_dino_benchmark(self):
        detector = cascade([False, False, False], unpooled=[True, True, True])
        with patch.object(CascadeDetector, 'check_with_vlm') as check:
            for t in range(3):
                detector.pending = ('dino', 'top')
                detector.append(None, 20. + t * .5)
        check.assert_not_called()

    def test_vlm_failure_stops_execution_with_causal_frames(self):
        detector = cascade([True] * 3)
        with tempfile.TemporaryDirectory() as directory, \
             patch('rosbag_io.decode_image_message', side_effect=lambda _m, t: Image.new('RGB', (8, 8))):
            detector.output_dir = Path(directory)
            for t in (20., 20.5, 21.):
                detector.pending = ('dino', 'top')
                row, _, _ = detector.append(None, t)
            saved = sorted(p.name for p in Path(directory).glob('*.jpg'))
        check = row['vlm_check']
        self.assertEqual(check['verdict'], 'failure')
        self.assertTrue(detector.halted)
        self.assertEqual(detector.termination_reason, 'vlm_confirmed_failure')
        # Gripper-camera frames at the alarm and 0.5 s and 1 s before it; never a later one.
        self.assertEqual([f['frame_sec'] for f in check['frames']], [21., 20.4, 20.])
        self.assertEqual(len(saved), 3)
        calls = [call.args for call in detector.checker.model.answer_probability.call_args_list]
        # The alarm is in stage 2 (after the grasp): is the object held? Asked about each frame alone.
        self.assertEqual(len(calls), 3)
        self.assertIn('held between the two white fingers', calls[0][0]['text'])
        self.assertEqual([options for _, options in calls], [('yes', 'no')] * 3)
        self.assertTrue(all(len(turn['images']) == 1 for turn, *_ in calls))
        self.assertGreater(detector.feed.origin, 0.)  # Replay time paused while the VLM answered.

    def test_reached_subgoal_resumes_the_execution(self):
        detector = cascade([True] * 3, probability=.8, boundaries={1: 0., 2: 5., 3: 18.})
        with patch('rosbag_io.decode_image_message', side_effect=lambda _m, t: Image.new('RGB', (8, 8))):
            for t in (20., 20.5, 21.):
                detector.pending = ('dino', 'top')
                row, _, _ = detector.append(None, t)
        check = row['vlm_check']
        self.assertEqual((check['stage'], check['verdict']), (3, 'safe'))
        self.assertAlmostEqual(check['subgoal_probability'], .8)
        # After the release (stage 3): does the object lie on the destination, on another surface, or nowhere?
        self.assertIn('A) A metal surgical instrument lies on a blue surface.', check['questions'][0])
        self.assertIn('C) No metal surgical instrument is visible.', check['questions'][0])
        options = [call.args[1] for call in detector.checker.model.answer_probability.call_args_list]
        self.assertEqual(options, [('A', 'B', 'C')] * 3)
        self.assertFalse(detector.halted)

    def test_gripper_frames_are_buffered_for_the_check_not_shown(self):
        from rynnbrain_vlm.cascade import CascadeFeed
        detector = cascade([])
        detector.frames = []
        detector.gmr = SimpleNamespace(sampler=SimpleNamespace(clock=SimpleNamespace(boundaries={1: 0.}, stage=1)),
                                       release_check=False)
        camera = iter([(t, 'camera', ('message', 'type')) for t in (1., 1.1, 1.2)])
        stop = SimpleNamespace(is_set=lambda: False, wait=lambda _delay: False)
        publish = Mock()
        feed = CascadeFeed(detector, camera, iter([]), publish, stop, Mock())
        self.assertIsNone(feed.next_frame(0.))
        self.assertEqual([t for t, _ in detector.frames], [1., 1.1, 1.2])
        publish.assert_not_called()  # The camera panels show DINO's camera, not the gripper camera.

    def test_replay_ends_when_the_execution_ends(self):
        from rynnbrain_vlm.cascade import CascadeFeed
        detector = cascade([False] * 5)
        detector.sampler_clock = SimpleNamespace(boundaries={1: 0., 2: 1., 3: 2.}, stage=3)
        detector.gmr = SimpleNamespace(sampler=SimpleNamespace(clock=detector.sampler_clock, release=None),
                                       release_check=False)
        image = Image.new('RGB', (8, 8))
        dino = iter([(t, 'dino', (('top', i), image)) for i, t in enumerate((3., 6., 7.5, 9.))])
        stop = SimpleNamespace(is_set=lambda: False, wait=lambda _delay: False)
        feed = CascadeFeed(detector, iter([]), dino, lambda *_: None, stop, Mock())
        times = []
        while (packet := feed.next_frame(0.)) is not None:
            times.append(packet[1])
        self.assertEqual(times, [3., 6.])  # Stage 3 at 2 s plus the 5 s window.

    def test_gui_replay_saves_interleaved_dino_and_gmr_rows(self):
        import csv
        import numpy as np
        from rynnbrain_vlm.online_replay import ReplayApplication
        image = Image.new('RGB', (8, 8))
        packets = iter([(('top', 0), 1., image), (0, 1.1, None), (1, 1.2, None), (('top', 1), 1.5, image)])
        feed = SimpleNamespace(origin=0., run=lambda: None, close=lambda: None,
                               next_frame=lambda _earliest: next(packets, None))
        def append(frame, timestamp, _verify):
            if frame is not None:  # DINO image
                return dict(timestamp_sec=timestamp, anomaly_score=.5, threshold=1., above_threshold=False,
                            cached_feature_sec=.01, verification=None, score_kind='dino_standardized_distance',
                            visual_matching={'status': 'unavailable'}), np.ones(3), None
            return dict(timestamp_sec=timestamp, anomaly_score=.2, threshold=1., above_threshold=False,
                        cached_feature_sec=.001, verification=None, score_kind='gmr_sensor_ratio',
                        alarm_filtered=False, alarm_sources=[], sensors={'position': {'distance': 1.}}), np.ones(2), None
        with tempfile.TemporaryDirectory() as directory:
            config = dict(output_dir=directory, rynnbrain={'camera_topics': ['top'], 'num_frames': 8})
            camera = SimpleNamespace(preview_image=None, matched_image=None)
            detector = SimpleNamespace(kind='cascade', knn=True, capture_visual=False, frame_count=None, start_sec=0.,
                end_sec=None, score_source='dino_gmr_with_vlm_check', score_kind='dino_gmr_cascade', sample_fps=2.,
                signature={'protocol': 'test'}, path=Path(directory), cameras={'top': camera}, release_result=None,
                sampler=SimpleNamespace(clock=SimpleNamespace(boundaries={1: 0.}, failures=[])), queries=[],
                termination_reason=None, reset=Mock(), prepare_replay=Mock(return_value={}),
                create_feed=Mock(return_value=feed), append=append)
            app = ReplayApplication(config, [Path('test')], 10., False)
            app.update(detector_type='cascade')
            app.detector = detector
            with patch('rynnbrain_vlm.online_replay.save_plot'):
                app.run_bag(Path('test'))
            result = app.snapshot()
            self.assertEqual(result['status'], 'completed', result.get('error'))
            output = Path(result['output_directory'])
            for name, count in (('dino', 2), ('gmr', 2)):
                with (output / f'online_scores_{name}.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), count)

    def test_gui_accepts_the_cascade_detector(self):
        from rynnbrain_vlm.online_replay import ReplayApplication
        app = ReplayApplication({'rynnbrain': {'num_frames': 8}}, [Path('heldout')], 10., False)
        app.update(status='ready')
        app.start(0, detector_type='cascade')
        self.assertEqual(app.state['requested_detector'], 'cascade')


if __name__ == '__main__':
    unittest.main()
