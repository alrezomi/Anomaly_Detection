import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from rynnbrain_vlm.visual_matching import (
    VisualMemory, RecordedStages, build_visual_memory, nominal_matching_bags, pool_visual_tokens,
    final_recorded_stages,
)


class VisualMatchingTests(unittest.TestCase):
    def test_last_three_numbered_markers_with_separate_failure_annotations(self):
        import pandas as pd
        events = pd.DataFrame({'time': [0., 1., 2., 10., 20., 30.],
                               'stage': ['1', '2', '3', '1', '2', '3']})
        stages = final_recorded_stages(events)
        self.assertEqual(stages.starts, (10., 20., 30.))
        self.assertIsNone(stages.at(9.))
        self.assertEqual(stages.at(20.), 2)
        self.assertEqual(stages.at(99.), 3)
        for error_time in (15., 25.):
            annotated = pd.concat([events, pd.DataFrame({'time': [5., error_time], 'stage': ['Error', 'Fail']})])
            result = final_recorded_stages(annotated)
            self.assertEqual(result.metadata(), stages.metadata())
            self.assertEqual(result.at(error_time), 1 if error_time < 20. else 2)
            self.assertEqual(result.failure_annotation()['time_sec'], error_time)
            self.assertEqual(result.failure_annotation()['marker_count'], 1)
        events.loc[4, 'stage'] = 'Error'
        with self.assertRaisesRegex(ValueError, 'missing stage starts'):
            final_recorded_stages(events)

    def test_pooling_normalizes_and_rejects_invalid_features(self):
        import torch
        result = pool_visual_tokens(torch.tensor([[3., 0.], [3., 8.]]))
        np.testing.assert_allclose(result, [.6, .8])
        for values in (torch.zeros(2, 2), torch.empty(0, 2), torch.tensor([[float('nan'), 1.]])):
            with self.assertRaises(ValueError):
                pool_visual_tokens(values)

    def test_nearest_frame_search_is_restricted_to_stage(self):
        entries = [dict(stage=1, frame_id=1), dict(stage=2, frame_id=2), dict(stage=2, frame_id=3)]
        memory = VisualMemory(entries, [[1., 0.], [.6, .8], [0., 1.]], {})
        # The exact match belongs to another stage and must not be selected.
        match = memory.match([1., 0.], 2)
        self.assertEqual(match['best']['frame_id'], 2)
        self.assertAlmostEqual(match['best']['cosine_distance'], .4)
        self.assertEqual(match['candidate_count'], 2)
        self.assertEqual(memory.match([1., 0.], None)['status'], 'unavailable')
        self.assertEqual(memory.match([1., 0.], 3)['status'], 'unavailable')

    def test_sources_use_nominal_training_list_only_or_references(self):
        config = {'rynnbrain': {'reference_bags': ['reference'], 'cop_classifier': {
            'training': {'normal_bags': ['normal', 'normal'], 'failure_bags': ['failure']}}}}
        self.assertEqual(nominal_matching_bags(config), ['normal'])
        config['rynnbrain']['cop_classifier']['training']['normal_bags'] = []
        self.assertEqual(nominal_matching_bags(config), ['reference'])

    def test_memory_is_cached_and_invalidated_when_recording_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bag = root / 'nominal'
            bag.mkdir()
            recording = bag / 'metadata.yaml'
            recording.write_text('original')
            config = {'output_dir': str(root / 'out'), 'rynnbrain': {
                'camera_topics': ['top'], 'reference_bags': [str(bag)]}}
            detector = SimpleNamespace(config=config, vlm=config['rynnbrain'], signature={'model': 'test'})
            def sample(source, count, start_sec, end_sec):
                self.assertEqual(count, 8)
                self.assertLess(start_sec, end_sec)
                return [(int(start_sec), start_sec, Image.new('RGB', (8, 8)))]
            with patch('rynnbrain_vlm.visual_matching.read_recorded_stages', return_value=RecordedStages((1., 2., 3.))), \
                 patch('rosbag_io.bag_duration_seconds', return_value=4.), \
                 patch('rosbag_io.sample_rosbag_image_frames_uniform', side_effect=sample), \
                 patch('rynnbrain_vlm.visual_matching._visual_descriptor', return_value=np.array([1., 0.])) as encode:
                memory = build_visual_memory(detector, Mock(), threading.Event())
                self.assertEqual(len(memory.entries), 3)
                self.assertEqual(encode.call_count, 3)
                again = build_visual_memory(detector, Mock(), threading.Event())
                np.testing.assert_array_equal(again.vectors, memory.vectors)
                self.assertEqual(encode.call_count, 3)
                recording.write_text('changed recording')
                build_visual_memory(detector, Mock(), threading.Event())
                self.assertEqual(encode.call_count, 6)
                stop = threading.Event()
                stop.set()
                with self.assertRaises(InterruptedError):
                    build_visual_memory(detector, Mock(), stop)
            summary = json.loads((root / 'out/online_replay/visual_memory/summary.json').read_text())
            self.assertEqual(summary['image_count'], 3)
            self.assertEqual(summary['skipped_bags'], [])


if __name__ == '__main__':
    unittest.main()
