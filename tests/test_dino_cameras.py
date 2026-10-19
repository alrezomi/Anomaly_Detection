import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image

from rynnbrain_vlm.dino_cameras import DinoCameraReplay
from rynnbrain_vlm.online_replay import ReplayApplication, SequentialReplayFeed


class DinoCameraTests(unittest.TestCase):
    topics = ['/top/image', '/flange/image']

    def detector(self, topic, threshold, extractor):
        def append(image, timestamp, verify):
            return dict(timestamp_sec=timestamp, anomaly_score=.15, threshold=threshold,
                above_threshold=.15 > threshold, cached_feature_sec=.001, verification=None,
                visual_matching={'status': 'unavailable'}, score_kind='dino_distance'), np.ones(2), None
        return SimpleNamespace(topic=topic, extractor=extractor, path=Path('cache') / topic.split('/')[1],
            signature={'topic': topic}, sample_fps=2., start_sec=0., excluded={'normal'},
            score_source='dino', note='independent', reset=Mock(), append=Mock(side_effect=append),
            preview_image=Image.new('RGB', (8, 8)), matched_image=None)

    def group(self):
        extractor = object()
        detectors = [self.detector(topic, threshold, extractor) for topic, threshold in zip(self.topics, [.1, .2])]
        with patch('rynnbrain_vlm.dino_cameras.DinoReplayDetector', side_effect=detectors) as factory:
            group = DinoCameraReplay({'camera_topics': self.topics, 'rynnbrain': {'camera_topics': ['/vlm']}}, Mock())
        self.assertIsNone(factory.call_args_list[0].kwargs['extractor'])
        self.assertIs(factory.call_args_list[1].kwargs['extractor'], extractor)
        return group

    def frames(self, source, **kwargs):
        times = [0., .002] if source.topic == self.topics[0] else [0., .001, .003]
        for index, timestamp in enumerate(times):
            yield index, timestamp, Image.new('RGB', (8, 8))

    def test_shared_model_independent_topics_and_missing_camera_failure(self):
        group = self.group()
        with patch('rosbag_io.list_bag_topics', return_value=pd.DataFrame({'topic': self.topics[:1]})):
            with self.assertRaisesRegex(ValueError, 'flange'):
                group.reset('bag')
        self.assertEqual(group.excluded, {'normal'})
        self.assertEqual([c['topic'] for c in group.catalogue()], self.topics)

    def test_time_merge_includes_equal_timestamps_and_unequal_frame_counts(self):
        group = self.group()
        group.bag = 'bag'
        with patch('rosbag_io.iter_rosbag_image_frames', side_effect=self.frames):
            feed = SequentialReplayFeed(group.iter_frames(), Mock(), threading.Event(),
                                        clock=lambda: 100., allow_equal_timestamps=True)
            # Avoid wall-clock pacing in this deterministic ordering test.
            feed.origin = 0.
            packets = []
            while (packet := feed.next_frame(0.)) is not None:
                packets.append(packet)
        self.assertEqual([x[1] for x in packets], [0., 0., .001, .002, .003])
        self.assertEqual([x[0][0] for x in packets].count(self.topics[0]), 2)
        self.assertEqual(feed.publish.call_count, 5)

    def test_replay_scores_and_exports_each_camera_without_fusion(self):
        group = self.group()
        with tempfile.TemporaryDirectory() as directory:
            app = ReplayApplication({'output_dir': directory, 'rynnbrain': {'camera_topics': ['/vlm']}},
                                    [Path('test')], 10., False)
            app.detector = group
            app.update(detector_type='dino', note='test', dino_cameras=group.catalogue())
            with patch('rosbag_io.list_bag_topics', return_value=pd.DataFrame({'topic': self.topics})), \
                 patch('rosbag_io.iter_rosbag_image_frames', side_effect=self.frames), \
                 patch('rynnbrain_vlm.visual_matching.read_corrected_failure_annotation', return_value={}), \
                 patch('rynnbrain_vlm.online_replay.save_plot') as plot:
                app.run_bag(Path('test'))
            state = app.snapshot()
            self.assertEqual(state['status'], 'completed', state.get('error'))
            self.assertEqual(len(state['rows']), 5)
            for topic, expected_alarm, count in zip(self.topics, [True, False], [2, 3]):
                rows = [row for row in state['rows'] if row['camera_topic'] == topic]
                self.assertEqual(len(rows), count)
                self.assertTrue(all(row['above_threshold'] == expected_alarm for row in rows))
                self.assertIn('observed', app.camera_jpegs[topic])
            self.assertEqual(plot.call_count, 2)
            self.assertTrue(all(len({r['camera_topic'] for r in call.args[0]}) == 1 for call in plot.call_args_list))
            saved = json.loads(next(Path(directory).glob('online_replay/*/test/online_results.json')).read_text())
            self.assertEqual(set(saved['comparison_signature']['cameras']), set(self.topics))
            self.assertEqual(len({row['vector_file'] for row in saved['rows']}), 5)
