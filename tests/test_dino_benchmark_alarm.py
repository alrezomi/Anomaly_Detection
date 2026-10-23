"""Alarm persistence and stage cutoffs must agree across decisions and scale sweeps."""
import unittest
from unittest.mock import patch

import pandas as pd

from tests.test_dino_benchmark import manifest, timeline
from vision_dinov2.benchmark_report import decisions, scale_sweep
from vision_dinov2.benchmark_alarm import ensure_stage_windows


class AlarmPolicyTests(unittest.TestCase):
    def decision(self, values, cutoff=100., other=None):
        item = dict(manifest('bag', 'fail', 1.), stage3_start_sec=cutoff)
        rows = timeline('bag', 'euclidean', values)
        if other:
            rows += timeline('bag', 'euclidean', other, camera='/other')
        return decisions([item], pd.DataFrame(rows), 1., 'configured').iloc[0]

    def test_isolated_spikes_and_alternating_cameras_do_not_confirm(self):
        self.assertEqual(self.decision([2., .2, 2.]).predicted_decision, 'success')
        self.assertEqual(self.decision([2., .2], other=[.2, 2.]).predicted_decision, 'success')

    def test_confirmation_time_is_second_crossing_and_equality_is_not_alarm(self):
        self.assertEqual(self.decision([.2, 2., 2.]).first_alarm_time_sec, 3.)
        self.assertEqual(self.decision([2., 1.]).predicted_decision, 'success')

    def test_crossing_at_stage_three_or_after_does_not_confirm(self):
        self.assertEqual(self.decision([2., 2.], cutoff=2.).predicted_decision, 'success')
        self.assertEqual(self.decision([.2, 2., 2.], cutoff=2.).predicted_decision, 'success')
        self.assertEqual(self.decision([2., 2., .2], cutoff=2.1).first_alarm_time_sec, 2.)

    def test_sweep_matches_actual_confirmation_and_ignores_late_large_scores(self):
        items = [dict(manifest('normal', 'normal'), stage3_start_sec=3.),
                 dict(manifest('failure', 'fail', 1.), stage3_start_sec=3.)]
        frames = pd.DataFrame(timeline('normal', 'euclidean', [.1, .2, 100.])+
                              timeline('failure', 'euclidean', [2., 3., 1000.]))
        result = decisions(items, frames, 1., 'configured')
        chosen = result[result.metric.eq('euclidean')]
        self.assertEqual(chosen.set_index('bag_path').loc['failure', 'max_score_ratio'], 2.)
        report, sweep, _ = scale_sweep(chosen, 1., frames)
        self.assertEqual(report['auc'], 1.)
        for point in sweep.itertuples():
            actual = decisions(items, frames, point.threshold_scale, 'test')
            actual = actual[actual.metric.eq('euclidean')].set_index('bag_path')
            self.assertEqual(point.recall, float(actual.loc['failure', 'predicted_decision']=='failure'))
            self.assertEqual(point.false_positive_rate, float(actual.loc['normal', 'predicted_decision']=='failure'))

    def test_old_run_stage_metadata_is_loaded_only_when_missing(self):
        from rynnbrain_vlm.visual_matching import RecordedStages
        item = manifest('bag', 'normal')
        del item['stage3_start_sec']
        with patch('rynnbrain_vlm.visual_matching.read_recorded_stages', return_value=RecordedStages((1.,2.,4.))) as read:
            loaded = ensure_stage_windows([item], '/stage')
            self.assertEqual(loaded[0]['stage3_start_sec'], 4.)
            ensure_stage_windows(loaded, '/stage')
            read.assert_called_once_with('bag', '/stage')
