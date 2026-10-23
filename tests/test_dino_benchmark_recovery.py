"""Regression for an overnight run whose recorded-stage labels contained no failures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from vision_dinov2.benchmark_labels import apply_csv_labels, validate_ground_truth
from vision_dinov2.benchmark_recovery import regenerate_reports
from vision_dinov2.benchmark_report import write_reports
from tests.test_dino_benchmark import manifest, timeline


class LabelTests(unittest.TestCase):
    def test_csv_paths_override_stage_labels_and_do_not_invent_missing_onsets(self):
        rows = [manifest('/data/Nominal/n', 'normal'), manifest('/data/Failure_8/f', 'normal')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'labels.csv'
            pd.DataFrame([dict(bag_path='Nominal/n',label='normal'),
                          dict(bag_path='/data/Failure_8/f',label='fail')]).to_csv(path,index=False)
            result = apply_csv_labels(rows, path, '/data')
            self.assertEqual(validate_ground_truth(result), dict(normal=1,fail=1,excluded=0))
            self.assertEqual(result[1]['ground_truth_label'], 'fail')
            self.assertEqual(result[1]['stage_ground_truth_label'], 'normal')
            self.assertTrue(result[1]['label_disagrees_with_stage'])
            self.assertIsNone(result[1]['recorded_failure_time_sec'])
            pd.DataFrame([dict(bag_path='Nominal/n',label='normal'),
                          dict(bag_path='Failure_8/f',label='fail',failure_time_sec=0.)]).to_csv(path,index=False)
            timed = apply_csv_labels(rows, path, '/data')
            self.assertEqual(timed[1]['recorded_failure_time_sec'], 0.)
            self.assertEqual(timed[1]['annotation_status'], 'available')

    def test_missing_duplicate_ambiguous_and_unknown_labels_stop_instead_of_falling_back(self):
        rows = [manifest('/data/a/same', 'normal'), manifest('/data/b/same', 'normal')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'labels.csv'
            for records, message in [([dict(bag_path='a/same',label='normal')], 'no entry'),
                    ([dict(bag_path='a/same',label='normal')]*2,'Duplicate'),
                    ([dict(bag_name='same',label='fail')],'Ambiguous')]:
                pd.DataFrame(records).to_csv(path,index=False)
                with self.assertRaisesRegex(ValueError,message):
                    apply_csv_labels(rows,path,'/data')
        with self.assertRaisesRegex(ValueError, '0 failure'):
            validate_ground_truth(rows)
        with self.assertRaisesRegex(ValueError, 'missing/unknown'):
            validate_ground_truth([manifest('unknown','unknown')])

    def test_preflight_stops_before_memory_and_model_loading(self):
        from vision_dinov2.benchmark import run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['train/a','train/b','train/c','test/n']:
                (root/name).mkdir(parents=True)
                (root/name/'metadata.yaml').write_text('')
            config=dict(output_dir=str(root/'output'),camera_topics=['/camera'],
                        nominal_bags=[str(root/f'train/{name}') for name in ['a','b','c']])
            def all_normal(rows,*args):
                return [dict(row,ground_truth_label='normal',recorded_failure_time_sec=None) for row in rows]
            with patch('vision_dinov2.benchmark.annotate_bags',side_effect=all_normal), \
                 patch('vision_dinov2.benchmark.prepare_camera') as prepare, \
                 patch('vision_dinov2.dinov2_features.DINOv2FeatureExtractor') as model:
                with self.assertRaisesRegex(ValueError,'ROC preflight failed'):
                    run(config,['test'],root)
                prepare.assert_not_called()
                model.assert_not_called()


class RecoveryTests(unittest.TestCase):
    def source_run(self, root):
        source=root/'saved_run'
        source.mkdir()
        rows=[manifest('/data/Nominal/n','normal',folder='Nominal'),
              manifest('/data/Failure_8/f','normal',folder='Failure_8')]
        frames=pd.DataFrame([row for metric in ['euclidean','mahalanobis'] for row in
                            timeline(rows[0]['bag_path'],metric,[.1,.2])+
                            timeline(rows[1]['bag_path'],metric,[.1,2.])])
        pd.DataFrame(rows).to_csv(source/'selection.csv',index=False)
        write_reports(rows,frames,source,1.1,['Nominal','Failure_8'])
        (source/'run.json').write_text(json.dumps(dict(score_variant='pooled',data_root='/data',
            camera_topics=['/camera'],selected_folders=['Nominal','Failure_8'],configured_scale=1.1,
            label_policy='existing final-three eligible stage messages',status='completed')))
        labels=root/'labels.csv'
        pd.DataFrame([dict(bag_path='Nominal/n',label='normal'),
                      dict(bag_path='Failure_8/f',label='fail')]).to_csv(labels,index=False)
        return source,labels

    def test_recovery_uses_saved_frames_and_preserves_original_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source,labels=self.source_run(Path(directory))
            original=(source/'frame_results.csv').read_bytes(),(source/'bag_results.csv').read_bytes()
            with patch('rosbag_io.read_stage_events',side_effect=AssertionError('no bag reads')), \
                 patch('vision_dinov2.benchmark.prepare_camera',side_effect=AssertionError('no calibration')), \
                 patch('vision_dinov2.benchmark.score_bag',side_effect=AssertionError('no inference')):
                output=regenerate_reports(source,labels)
            self.assertEqual(original,((source/'frame_results.csv').read_bytes(),(source/'bag_results.csv').read_bytes()))
            state=json.loads((output/'run.json').read_text())
            self.assertFalse(state['inference_rerun'])
            self.assertEqual(state['ground_truth_counts']['fail'],1)
            for metric in ['euclidean','mahalanobis']:
                self.assertEqual(state['roc'][metric]['status'],'created')
                self.assertEqual(state['roc'][metric]['auc'],1.)
            decisions=pd.read_csv(output/'bag_results.csv')
            configured=decisions[decisions.operating_point.eq('configured')]
            failures=configured[configured.ground_truth_label.eq('fail')]
            self.assertTrue(failures.first_alarm_time_sec.eq(2.).all())
            self.assertTrue(failures.failure_time_error_sec.isna().all())
            self.assertTrue(failures.timing_status.eq('missing_failure_marker').all())
            self.assertTrue((output/'alarm_diagnostics.csv').is_file())

    def test_recovery_rejects_truncated_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            source,labels=self.source_run(Path(directory))
            path=source/'frame_results.csv'
            frames=pd.read_csv(path)
            frames.iloc[:-1].to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'frame count differs'):
                regenerate_reports(source,labels)

    def test_stage_only_one_class_cannot_be_reported_as_a_valid_roc(self):
        with tempfile.TemporaryDirectory() as directory:
            source,_=self.source_run(Path(directory))
            with self.assertRaisesRegex(ValueError,'0 failure'):
                regenerate_reports(source)
