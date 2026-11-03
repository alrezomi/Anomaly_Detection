from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from rynnbrain_vlm.cascade import CascadeDetector, parse_verdict, select_vlm_frame_times


def cascade(above, verdict='Decision: failure\nReason: two instruments lifted.'):
    """Detector with one fake DINO camera whose frames are above threshold as listed."""
    detector = CascadeDetector.__new__(CascadeDetector)
    rows = iter(dict(anomaly_score=2. if a else .5, threshold=1., above_threshold=a) for a in above)
    detector.cameras = {'top': SimpleNamespace(append=lambda *_: (next(rows), None, None))}
    detector.task = 'Pick the scissor and place it in the basket.'
    detector.generation, detector.vlm_topic = {}, 'top'
    detector.vlm = SimpleNamespace(generate_multiturn=Mock(return_value=('reference', verdict)))
    detector.reference_turn = dict(role='user', images=[('Frame 1, camera top, time 0.500 s', None)] * 8, text='turn 1')
    detector.reference_response, detector.reference_bags = 'The robot picks the scissor.', ['nominal_1.1']
    detector.gmr = SimpleNamespace(sampler=SimpleNamespace(clock=SimpleNamespace(boundaries={1: 0., 2: 15.})))
    detector.update, detector.feed = Mock(), SimpleNamespace(origin=0.)
    detector.streaks, detector.armed = {'top': 0}, {'top': True}
    detector.queries, detector.vlm_frame_jpegs = [], {}
    detector.halted, detector.termination_reason, detector.output_dir = False, None, None
    detector.frames = [(t / 10, ('message', ('message', t / 10))) for t in range(0, 400, 2)]  # 5 fps camera
    return detector


class CascadeTests(unittest.TestCase):
    def test_frames_use_context_then_the_failure_stage_and_never_the_future(self):
        picks = select_vlm_frame_times({1: 0., 2: 15.}, 30.)
        times = [t for t, _ in picks]
        self.assertEqual(len(picks), 8)
        self.assertEqual(times[:2], [2.5, 12.5])  # After stage 1 starts, before stage 2 starts.
        self.assertEqual(times[2], 15.)
        self.assertEqual(times[-1], 30.)
        self.assertEqual(picks[-1][1], 'alarm')
        # An alarm soon after stage 2 starts gets fewer frames.
        self.assertEqual(len(select_vlm_frame_times({1: 0., 2: 15.}, 16.5)), 4)
        # In stage 1 every frame comes from stage 1; a later boundary is ignored.
        stage_one = select_vlm_frame_times({1: 0., 2: 15.}, 10.)
        self.assertEqual(len(stage_one), 8)
        self.assertTrue(all(0. <= t <= 10. for t, _ in stage_one))

    def test_unclear_answers_stop_the_robot(self):
        self.assertEqual(parse_verdict('Decision: safe to continue\nReason: fine.'), 'safe')
        self.assertEqual(parse_verdict('Decision: failure\nReason: dropped.'), 'failure')
        self.assertEqual(parse_verdict('I am not sure what happened.'), 'failure')

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
        self.assertTrue(all(f['frame_sec'] <= 21. for f in check['frames']))
        self.assertEqual(len(check['frames']), len(saved))
        (reference, turn), _generation = detector.vlm.generate_multiturn.call_args.args
        # Multi-turn: a complete nominal execution first, then this execution up to the alarm.
        self.assertIs(reference, detector.reference_turn)
        self.assertEqual(detector.vlm.generate_multiturn.call_args.kwargs['reference_response'], detector.reference_response)
        self.assertIn('stopped in the middle of the task', turn['text'])
        # The alarm is in stage 2 (after the grasp): the prompt says the object should already be held.
        self.assertIn('stopped in stage 2, so the grasp is already complete', turn['text'])
        self.assertTrue(turn['images'][0][0].startswith('Frame 1, camera top, time 2.400 s'))  # Latest at or before 2.5 s.
        self.assertGreater(detector.feed.origin, 0.)  # Replay time paused while the VLM answered.
        self.assertEqual(len({f['frame_sec'] for f in check['frames']}), len(check['frames']))

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
