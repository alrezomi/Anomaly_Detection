from pathlib import Path
import tempfile
import unittest

import pandas as pd

from rynnbrain_vlm.benchmark_protocol import configure_decision_mode, timeline_benchmark_row
from rynnbrain_vlm.benchmark_metrics import folder_metrics
from rynnbrain_vlm.benchmark_roc import write_probability_roc_report, write_anomaly_roc_report


class TimelineBenchmarkTests(unittest.TestCase):
    def row(self, directory, scores, thresholds=None, *, label='normal', method='logistic', status='created'):
        path = Path(directory)/f'{method}_{label}.csv'
        pd.DataFrame(dict(timestamp_sec=list(range(1, len(scores)+1)), anomaly_score=scores,
                          threshold=thresholds or [.5]*len(scores),
                          threshold_source=['prefix_model']*len(scores),
                          classifier_model_path=['onset.npz']*len(scores))).to_csv(path, index=False)
        return dict(bag_name=f'bag_{label}', bag_path=f'/data/{label}/bag', folder='selected',
                    ground_truth_label=label, input_mode='raw', decision='not_generated',
                    classifier_decision='success', classifier_decision_correct=label == 'normal',
                    classifier_failure_probability=.01 if method == 'logistic' else None,
                    classifier_anomaly_score=.1 if method == 'knn' else None,
                    classifier_score_kind='failure_probability' if method == 'logistic' else 'knn_distance',
                    classifier_model_path='whole.npz', classifier_threshold=.5,
                    **{f'{method}_timeline_csv': str(path), f'{method}_timeline_status': status})

    def test_opt_in_requires_classifier_and_enables_prefix_scoring(self):
        vlm = {'cop_classifier': {'enabled': True, 'plot_timeline': False, 'method': 'logistic'}}
        self.assertEqual(configure_decision_mode({}, vlm), 'whole_demo')
        self.assertFalse(vlm['cop_classifier']['plot_timeline'])
        self.assertEqual(configure_decision_mode({'decision_mode': 'timeline'}, vlm), 'timeline')
        self.assertTrue(vlm['cop_classifier']['plot_timeline'])
        for selection, settings, skip in [({'decision_mode': 'wrong'}, vlm, False),
                                           ({'decision_mode': 'timeline'}, {}, False),
                                           ({'decision_mode': 'timeline'}, vlm, True)]:
            with self.assertRaises(ValueError):
                configure_decision_mode(selection, settings, skip_vlm=skip)

    def test_transient_alarm_changes_metrics_even_when_final_outcome_is_success(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.row(directory, [.1, .7, .01])
            row = timeline_benchmark_row(original, 'logistic')
            self.assertEqual(original['classifier_decision'], 'success')
            self.assertEqual(row['whole_demo_classifier_decision'], 'success')
            self.assertEqual(row['whole_demo_classifier_failure_probability'], .01)
            self.assertEqual(row['classifier_decision'], 'failure')
            self.assertEqual(row['classifier_failure_probability'], .7)
            self.assertEqual(row['timeline_first_alarm_time_sec'], 2.)
            self.assertEqual(row['classifier_model_path'], 'onset.npz')
            metrics = folder_metrics([row], ['selected']).iloc[0]
            self.assertEqual(metrics.decision_source, 'classifier_timeline')
            self.assertEqual(metrics.false_positive_rate, 1.)
            self.assertEqual(metrics.accuracy, 0.)

    def test_complete_no_alert_and_threshold_equality(self):
        with tempfile.TemporaryDirectory() as directory:
            normal = timeline_benchmark_row(self.row(directory, [.1, .2]), 'logistic')
            self.assertEqual(normal['classifier_decision'], 'success')
            self.assertIsNone(normal['timeline_first_alarm_time_sec'])
            equality = timeline_benchmark_row(self.row(directory, [.1, .5]), 'logistic')
            self.assertEqual(equality['classifier_decision'], 'failure')

    def test_incomplete_negative_is_unavailable_and_positive_is_counted_without_roc_score(self):
        with tempfile.TemporaryDirectory() as directory:
            for scores, decision in [([.1, None, .2], None), ([.7, None, .2], 'failure')]:
                row = timeline_benchmark_row(self.row(directory, scores, status='partial'), 'logistic')
                self.assertEqual(row['classifier_decision'], decision)
                self.assertIsNone(row['classifier_failure_probability'])
                self.assertTrue(row['timeline_error'])
                metric = folder_metrics([row], []).iloc[0]
                if decision:
                    self.assertEqual(metric.false_positive, 1)
                else:
                    self.assertEqual(metric.evaluated_bags, 0)

    def test_missing_or_invalid_timeline_never_falls_back_to_final_success(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.row(directory, [.1, .2])
            Path(row['logistic_timeline_csv']).unlink()
            result = timeline_benchmark_row(row, 'logistic')
            self.assertIsNone(result['classifier_decision'])
            self.assertTrue(result['classifier_error'])
            self.assertEqual(result['whole_demo_classifier_decision'], 'success')
            row = self.row(directory, [.1, .2])
            data = pd.read_csv(row['logistic_timeline_csv'])
            data['timestamp_sec'] = [2., 1.]
            data.to_csv(row['logistic_timeline_csv'], index=False)
            self.assertIsNone(timeline_benchmark_row(row, 'logistic')['classifier_decision'])

    def test_knn_uses_per_prefix_threshold_ratio_not_maximum_raw_distance(self):
        with tempfile.TemporaryDirectory() as directory:
            row = timeline_benchmark_row(self.row(directory, [.9, .6, .1], [1., .5, .5], method='knn'), 'knn')
            self.assertEqual(row['classifier_decision'], 'failure')
            self.assertEqual(row['timeline_first_alarm_time_sec'], 2.)
            self.assertAlmostEqual(row['classifier_anomaly_score'], 1.2)
            self.assertEqual(row['classifier_threshold'], 1.)
            self.assertEqual(row['timeline_peak_score'], .6)

    def test_roc_and_folder_metrics_agree_on_prefix_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [timeline_benchmark_row(self.row(directory, scores, label=label), 'logistic')
                    for scores, label in [([.1, .2, .01], 'normal'), ([.1, .8, .01], 'fail')]]
            report = write_probability_roc_report(rows, Path(directory))
            self.assertEqual(report['decision_mode'], 'timeline')
            overall = report['metrics'][0]
            self.assertEqual(overall['auroc'], 1.)
            metric = folder_metrics(rows, []).iloc[0]
            self.assertEqual(overall['recall_at_threshold'], metric.recall)
            self.assertEqual(overall['false_positive_rate_at_threshold'], metric.false_positive_rate)
            knn = [timeline_benchmark_row(self.row(directory, scores, thresholds, label=label, method='knn'), 'knn')
                   for scores, thresholds, label in [([.8, .2], [1., .5], 'normal'), ([.9, .6], [1., .5], 'fail')]]
            self.assertEqual(write_anomaly_roc_report(knn, Path(directory))['metrics'][0]['auroc'], 1.)


if __name__ == '__main__':
    unittest.main()
