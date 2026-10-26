from pathlib import Path
import tempfile
import unittest

import pandas as pd

from rynnbrain_vlm.benchmark_metrics import folder_metrics, write_metrics_report


class VlmFolderMetricsTests(unittest.TestCase):
    def rows(self):
        def row(label, decision, classifier, **kwargs):
            return dict(ground_truth_label=label, decision=decision, classifier_decision=classifier,
                        input_mode='raw', folder='chosen', **kwargs)
        return [row('normal', 'failure', 'success'), row('normal', 'success', 'success'),
                row('fail', 'failure', 'failure', failure_timing_status='compared', failure_time_error_sec=2.),
                row('fail', 'success', 'failure'), row('fail', 'uncertain', 'uncertain'),
                row('fail', 'failed', None, evaluation_error='read failed'), row('unknown', 'failure', 'failure')]

    def test_confusion_rates_abstentions_errors_and_classifier_timing_are_separate(self):
        table = folder_metrics(self.rows(), ['chosen', 'empty'])
        raw = table[table.folder.eq('ALL')].set_index('decision_source')
        vlm, classifier = raw.loc['vlm'], raw.loc['classifier']
        self.assertEqual((vlm.true_positive, vlm.false_positive, vlm.true_negative, vlm.false_negative), (1,1,1,1))
        self.assertEqual(vlm.recall, .5)
        self.assertEqual(vlm.false_positive_rate, .5)
        self.assertEqual(vlm.failure_recall_all, .25)
        self.assertEqual(vlm.decision_coverage, 4/6)
        self.assertEqual(vlm.accuracy, 2/6)
        self.assertEqual(vlm.error_bags, 1)
        self.assertEqual(vlm.abstained_or_unavailable_bags, 1)
        self.assertEqual(vlm.unknown_label_count, 1)
        self.assertEqual(vlm.timing_compared, 0)
        self.assertEqual(classifier.recall, 1.)
        self.assertEqual(classifier.false_positive_rate, 0.)
        self.assertEqual(classifier.timing_mae_sec, 2.)
        self.assertTrue(table[table.folder.eq('empty')].recall.isna().all())

    def test_input_modes_are_not_pooled_and_absent_class_rates_stay_unavailable(self):
        rows = self.rows()+[dict(ground_truth_label='normal', decision='success', input_mode='heatmap', folder='other')]
        table = folder_metrics(rows, ['chosen', 'other'])
        heatmap = table[table.input_mode.eq('heatmap') & table.folder.eq('ALL') & table.decision_source.eq('vlm')]
        self.assertTrue(heatmap.recall.isna().all())
        self.assertTrue(heatmap.false_positive_rate.eq(0.).all())

    def test_report_writes_readable_graph_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = write_metrics_report(self.rows(), output, ['chosen'])
            self.assertTrue(Path(report['plot']).is_file())
            table = pd.read_csv(report['csv'])
            self.assertEqual(set(table.folder), {'ALL', 'chosen'})
            self.assertTrue((output/'benchmark_metrics_notes.txt').is_file())

    def test_failed_bags_without_a_mode_remain_in_each_requested_mode(self):
        rows = [dict(ground_truth_label='fail', decision='failed', input_mode=None,
                     evaluation_error='missing images', folder='chosen')]
        table = folder_metrics(rows, ['chosen'], input_modes=['raw', 'heatmap'])
        overall = table[table.folder.eq('ALL')]
        self.assertEqual(set(overall.input_mode), {'raw', 'heatmap'})
        self.assertTrue(overall.error_bags.eq(1).all())
        self.assertTrue(overall.evaluated_bags.eq(0).all())
        self.assertTrue(overall.decision_coverage.eq(0.).all())
        self.assertTrue(overall.recall.isna().all())
