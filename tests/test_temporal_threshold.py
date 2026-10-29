from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from rynnbrain_vlm.temporal_threshold import first_alarm_cost, write_temporal_cost_report


def item(label, times, scores, onset=None, name='demo'):
    return dict(bag_name=name, ground_truth_label=label, times=np.asarray(times, dtype=float),
                scores=np.asarray(scores, dtype=float), onset=onset)


def scan(name, label, times, scores, complete=True):
    return (dict(bag_name=name, label=label), Path(name+'.npz'),
            dict(complete=complete, rows=[dict(timestamp_sec=t, anomaly_score=s) for t, s in zip(times, scores)]))


class TemporalThresholdTests(unittest.TestCase):
    def test_annotation_read_error_is_explicit_not_a_fabricated_onset(self):
        from rynnbrain_vlm.benchmark_replay import _read_timing_annotation
        with patch('rynnbrain_vlm.visual_matching.read_corrected_failure_annotation', side_effect=ValueError('missing stages')):
            annotation = _read_timing_annotation(Path('bad_bag'), {})
        self.assertEqual(annotation['status'], 'unavailable')
        self.assertIsNone(annotation['time_sec'])
        self.assertIn('missing stages', annotation['error'])

    def test_first_alarm_cannot_be_rescued_and_exact_onset_and_misses_are_handled(self):
        premature = first_alarm_cost(item('fail', [0, 10, 20], [.7, .8, .9], 10), .7)
        self.assertEqual(premature['status'], 'premature_alarm')
        self.assertEqual(premature['cost'], 1.)
        self.assertEqual(premature['delay_sec'], -10.)
        onset = first_alarm_cost(item('fail', [0, 10, 20], [.1, .8, .9], 10), .8)
        self.assertEqual(onset['cost'], 0.)
        self.assertEqual(onset['first_alarm_sec'], 10.)
        late = first_alarm_cost(item('fail', [0, 15, 30], [.1, .8, .9], 10), .8)
        self.assertEqual(late['cost'], .25)
        self.assertEqual(late['delay_sec'], 5.)
        missed = first_alarm_cost(item('fail', [0, 10, 20], [.1, .2, .3], 10), .8)
        self.assertEqual(missed['cost'], 1.)
        self.assertIsNone(missed['delay_sec'])
        self.assertEqual(first_alarm_cost(item('normal', [0, 10], [.1, .8]), .8)['cost'], 1.)
        self.assertEqual(first_alarm_cost(item('normal', [0, 10], [.1, .8]), .9)['cost'], 0.)

    def test_delay_cost_prefers_prompt_detection_to_perfect_eventual_accuracy(self):
        scans = [scan('normal1', 'normal', [0, 60], [.1, .8]),
                 scan('normal2', 'normal', [0, 60], [.1, .2]),
                 scan('failure', 'fail', [0, 10, 20, 60], [.1, .6, .8, .99])]
        annotations = dict(failure=dict(status='available', time_sec=10.))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = write_temporal_cost_report(scans, annotations, root)
            self.assertEqual(report['selected_threshold'], .21)
            self.assertEqual(report['minimum_cost'], .25)
            data = pd.read_csv(report['metrics_csv']).set_index('threshold_percent')
            self.assertEqual(data.loc[21, 'mean_detected_delay_sec'], 0.)
            self.assertEqual(data.loc[81, 'mean_detected_delay_sec'], 50.)
            self.assertGreater(data.loc[81, 'balanced_cost'], data.loc[21, 'balanced_cost'])
            self.assertEqual(data.loc[100, 'balanced_cost'], .5)  # No-alarm cannot win with zero cost.
            self.assertEqual(set(data.index), set(range(101)))
            self.assertTrue(Path(report['plot_file']).is_file())
            doubled = write_temporal_cost_report(scans[:2]*3 + scans[2:], annotations, root/'duplicated_normals')
            self.assertEqual(doubled['minimum_cost'], report['minimum_cost'])
            self.assertEqual(doubled['selected_threshold'], report['selected_threshold'])

    def test_incomplete_missing_onsets_and_outside_window_are_excluded_explicitly(self):
        scans = [scan('normal', 'normal', [0, 20], [.1, .2]),
                 scan('failure', 'fail', [0, 20], [.1, .9]),
                 scan('no_onset', 'fail', [0, 20], [.1, .9]),
                 scan('after_window', 'fail', [0, 20], [.1, .9]),
                 scan('partial', 'fail', [0, 20], [.1, .9], False)]
        annotations = dict(failure=dict(status='available', time_sec=10.),
                           after_window=dict(status='available', time_sec=20.))
        with tempfile.TemporaryDirectory() as directory:
            report = write_temporal_cost_report(scans, annotations, Path(directory))
            self.assertEqual((report['normal_count'], report['failure_count']), (1, 1))
            self.assertEqual({row['reason'] for row in report['excluded_bags']},
                             {'missing_failure_onset', 'failure_onset_outside_scored_window', 'incomplete_scan'})
            unavailable = write_temporal_cost_report(scans[:1], annotations, Path(directory)/'normal_only')
            self.assertEqual(unavailable['status'], 'skipped')
            self.assertIsNone(unavailable['selected_threshold'])
