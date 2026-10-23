"""Folder isolation, strict bag decisions, scale tuning and benchmark orchestration."""
import json
from pathlib import Path
import tempfile
import unittest
import shutil
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image

from vision_dinov2.benchmark import select_bags, annotate_bags, prepare_camera, score_bag, run
from vision_dinov2.benchmark_report import decisions, metrics_table, scale_sweep, wilson, write_reports


def manifest(path, label, onset=None, folder='test', status='ok'):
    return dict(bag_path=path, bag_name=Path(path).name, folder=folder, ground_truth_label=label,
                status=status, recorded_failure_time_sec=onset)


def timeline(path, metric, values, bounds=None, camera='/camera'):
    return [dict(bag_path=path, metric=metric, camera_topic=camera, timestamp_sec=float(index+1),
                 pooled_score=value, base_threshold=(bounds[index] if bounds else 1.), feature_sec=.1,
                 comparison_sec=.02, metric_score_sec=.01, read_decode_sec=.03, processing_sec=.13)
            for index, value in enumerate(values)]


class BenchmarkSelectionTests(unittest.TestCase):
    def test_scans_only_selected_folders_and_excludes_by_full_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['chosen/same', 'chosen/train', 'other/same']:
                path = root / name
                path.mkdir(parents=True)
                (path / 'metadata.yaml').write_text('')
            rows, folders = select_bags(root, ['chosen'], [root/'chosen/train', root/'other/same'])
            self.assertEqual(folders, ['chosen'])
            self.assertEqual({row['bag_name'] for row in rows}, {'same', 'train'})
            self.assertEqual(next(x for x in rows if x['bag_name']=='same')['status'], 'ok')
            self.assertEqual(next(x for x in rows if x['bag_name']=='train')['status'], 'excluded_training')
            (root/'chosen/train/data.mcap').write_text('')
            mcap_rows, _ = select_bags(root, ['chosen'], [root/'chosen/train/data.mcap'])
            self.assertEqual(next(x for x in mcap_rows if x['bag_name']=='train')['status'], 'excluded_training')
            single, _ = select_bags(root, ['chosen/same'], [])
            self.assertEqual(len(single), 1)
            for invalid in [[], ['.'], ['chosen', 'chosen/same'], ['../'], ['chosen', 'chosen']]:
                with self.assertRaises(ValueError):
                    select_bags(root, invalid, [])

    def test_labels_and_onsets_use_existing_policies_without_inferring_from_folder_name(self):
        rows = [manifest('Failure_folder/actually_normal', 'unknown'),
                manifest('normal_folder/actually_failure', 'unknown')]
        with patch('build_dataset_manifest.infer_bag_record', side_effect=[
                dict(label='normal', read_status='ok', used_stage_markers='1,2,3'),
                dict(label='fail', read_status='ok', used_stage_markers='2,Error,3')]), \
             patch('rynnbrain_vlm.failure_timing.read_failure_annotation', side_effect=[
                dict(status='missing_marker', time_sec=None), dict(status='available', time_sec=1.5, marker='Error')]):
            annotated = annotate_bags(rows, '/stage', .1)
        self.assertEqual(annotated[0]['ground_truth_label'], 'normal')
        self.assertEqual(annotated[1]['recorded_failure_time_sec'], 1.5)


class BenchmarkStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.manifest = [manifest('normal', 'normal'), manifest('failure', 'fail', 2.),
                         manifest('miss', 'fail', 2.), manifest('broken', 'fail', 2., status='error')]
        self.frames = pd.DataFrame([row for metric in ['euclidean', 'mahalanobis'] for row in
            timeline('normal', metric, [.1, .2, 1.]) + timeline('failure', metric, [.1, 1.1, 2.]) +
            timeline('miss', metric, [.1, .2, .3]) + timeline('broken', metric, [10.])])

    def test_any_camera_alarm_strict_ties_and_first_alarm_use_observation_time(self):
        rows = decisions(self.manifest, self.frames, 1., 'configured')
        group = rows[rows.metric.eq('euclidean')].set_index('bag_path')
        self.assertEqual(group.loc['normal', 'predicted_decision'], 'success')  # equality is no alarm
        self.assertEqual(group.loc['failure', 'first_alarm_time_sec'], 2.)
        self.assertEqual(group.loc['failure', 'failure_time_error_sec'], 0.)
        self.assertEqual(group.loc['miss', 'timing_status'], 'missed_failure')
        self.assertTrue(pd.isna(group.loc['broken', 'predicted_decision']))
        second = pd.DataFrame(timeline('normal', 'euclidean', [2.], camera='/other'))
        extra = decisions(self.manifest, pd.concat([self.frames, second]), 1., 'configured')
        self.assertEqual(extra[(extra.metric=='euclidean') & (extra.bag_path=='normal')].iloc[0].first_alarm_time_sec, 1.)

    def test_confusion_counts_errors_not_successes_and_single_class_rates_are_undefined(self):
        results = decisions(self.manifest, self.frames, 1., 'configured')
        metrics = metrics_table(results, ['test', 'empty'])
        overall = metrics[(metrics.metric=='euclidean') & (metrics.folder=='ALL')].iloc[0]
        self.assertEqual(overall.true_positive, 1)
        self.assertEqual(overall.false_negative, 1)
        self.assertEqual(overall.true_negative, 1)
        self.assertEqual(overall.false_positive, 0)
        self.assertEqual(overall.excluded_or_failed_bags, 1)
        self.assertEqual(overall.recall, .5)
        self.assertEqual(overall.false_positive_rate, 0.)
        self.assertTrue(metrics[metrics.folder.eq('empty')].recall.isna().all())
        failures = results[results.ground_truth_label.eq('fail')]
        self.assertTrue(metrics_table(failures, ['test']).false_positive_rate.isna().all())
        low, high = wilson(1, 2)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)

    def test_scale_sweep_best_reproduces_runtime_decisions_and_ties(self):
        results = decisions(self.manifest, self.frames, 1., 'configured')
        results = results[results.metric.eq('euclidean')]
        frames = self.frames[self.frames.metric.eq('euclidean')]
        report, sweep, curve = scale_sweep(results, 1., frames)
        self.assertEqual(report['status'], 'created')
        self.assertEqual(len(curve[0]), len(curve[1]))
        for row in sweep.itertuples():
            predicted = decisions(self.manifest, self.frames, row.threshold_scale, 'sweep')
            stats = metrics_table(predicted, [])
            actual = stats[stats.metric.eq('euclidean')].iloc[0]
            self.assertEqual(actual.recall, row.recall)
            self.assertEqual(actual.false_positive_rate, row.false_positive_rate)
        unavailable = scale_sweep(results[results.ground_truth_label.eq('fail')], 1.)[0]
        self.assertEqual(unavailable['status'], 'unavailable')

    def test_zero_threshold_and_zero_score_follow_actual_decision_rule(self):
        rows = [manifest('zero', 'normal'), manifest('positive', 'fail', 1.)]
        frames = pd.DataFrame(timeline('zero', 'euclidean', [0.], [0.]) +
                              timeline('positive', 'euclidean', [1.], [0.]))
        result = decisions(rows, frames, 1., 'configured')
        report, _, _ = scale_sweep(result[result.metric.eq('euclidean')], 1., frames)
        self.assertEqual(report['best']['threshold_scale'], 1.)
        self.assertEqual(report['best']['recall'], 1.)
        self.assertEqual(report['best']['false_positive_rate'], 0.)
        self.assertEqual(report['zero_threshold_always_alarm_bags'], 1)

    def test_report_exports_roc_best_scale_and_missing_failure_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, roc = write_reports(self.manifest, self.frames, directory, 1., ['test'])
            root = Path(directory)
            self.assertTrue((root/'roc.png').is_file())
            self.assertTrue((root/'euclidean_failure_timing_01.png').is_file())
            results = pd.read_csv(root/'bag_results.csv')
            self.assertEqual(set(results.operating_point), {'configured','exploratory_best'})
            self.assertTrue(results[(results.operating_point=='configured') & (results.bag_path=='miss')].failure_time_error_sec.isna().all())
            for metric in ['euclidean', 'mahalanobis']:
                self.assertTrue(np.isfinite(roc[metric]['best']['threshold_scale']))
                self.assertTrue((root/f'{metric}_scale_sweep.csv').is_file())


class BenchmarkEngineTests(unittest.TestCase):
    def test_both_calibrations_share_features_and_second_build_reuses_calibration(self):
        # Import before patching cache/extraction functions: the legacy launcher
        # binds these imports at module load, and must not retain test mocks.
        from vision_dinov2 import run_vision_test
        from tests.test_dino_metrics import memory
        data = memory()
        with tempfile.TemporaryDirectory() as directory:
            config = dict(output_dir=directory, nominal_cache_dir=directory+'/cache', vision_memory_name='test',
                          nominal_bags=['nominal_1', 'nominal_2', 'nominal_3', 'nominal_4'], camera_topics=['/camera'])
            get_extractor = Mock()
            with patch('vision_dinov2.nominal_cache.load_feature_cache', return_value=(data, 4)), \
                 patch('vision_dinov2.anomaly_detection.build_nominal_video_memory') as extract:
                memory1, tables, signature, timings, _ = prepare_camera(config, '/camera', get_extractor, Path(directory))
                self.assertIs(memory1, data)
                for metric in ['euclidean', 'mahalanobis']:
                    self.assertIn('threshold_pooled', tables[metric])
                with patch('vision_dinov2.threshold.calibrate_attention_leave_one_video_out', side_effect=AssertionError('should reuse')):
                    _, reused, _, phases, _ = prepare_camera(config, '/camera', get_extractor, Path(directory))
                self.assertTrue(all(x['status']=='reused' for x in phases))
                camera_root = Path(directory)/'cache/test/camera'
                # A removed constant margin must not trigger expensive fitting.
                for metric in ['euclidean', 'mahalanobis']:
                    cached = camera_root/'benchmark_calibration'/metric
                    marker = cached/'signature.json'
                    legacy_signature = json.loads(marker.read_text())
                    legacy_signature['threshold_margin'] = 1.1
                    marker.write_text(json.dumps(legacy_signature))
                with patch('vision_dinov2.threshold.calibrate_attention_leave_one_video_out', side_effect=AssertionError('reuse legacy scores')):
                    _, converted, _, phases, _ = prepare_camera(config, '/camera', get_extractor, Path(directory))
                for metric in ['euclidean', 'mahalanobis']:
                    np.testing.assert_allclose(converted[metric].threshold_pooled, tables[metric].threshold_pooled)
                self.assertTrue(all(x['status']=='reused' for x in phases))
                euclidean_cache = camera_root/'benchmark_calibration/euclidean'
                for filename in ['signature.json','patch_normality.npz','adaptive_threshold.csv','calibration.csv']:
                    shutil.copy2(euclidean_cache/filename, camera_root/filename)
                (euclidean_cache/'signature.json').rename(euclidean_cache/'signature.saved.json')
                with patch('vision_dinov2.threshold.calibrate_attention_leave_one_video_out', side_effect=AssertionError('reuse online calibration')):
                    _, _, _, reused_online, _ = prepare_camera(config, '/camera', get_extractor, Path(directory))
                self.assertEqual(reused_online[1]['status'], 'reused_existing_memory_calibration')
                get_extractor.assert_not_called()
                extract.assert_not_called()
            # One extractor call and one representative match per sampled frame,
            # while BOTH actual metric scorers use the same extracted features.
            image = Image.new('RGB', (56, 56))
            query = data[0]
            extractor = Mock(device='cpu')
            extractor.extract.return_value = query['features'], query['attention'], query['cls'], 4
            generator = (packet for packet in [(0, 1., image), (1, 2., image)])
            with patch('vision_dinov2.dinov2_features.iter_source_frames', return_value=generator):
                rows = score_bag(config, '/camera', 'test_bag', data, tables, signature, extractor)
            self.assertEqual(extractor.extract.call_count, 2)
            self.assertEqual(len(rows), 4)
            self.assertEqual({x['metric'] for x in rows}, {'euclidean', 'mahalanobis'})

    def test_separate_calibration_features_are_extracted_once_for_both_metrics(self):
        from vision_dinov2 import run_vision_test
        from vision_dinov2.benchmark import _signature
        from vision_dinov2.nominal_cache import save_feature_cache
        from tests.test_dino_metrics import memory
        data = memory()
        with tempfile.TemporaryDirectory() as directory:
            config = dict(output_dir=directory, nominal_cache_dir=directory+'/cache', vision_memory_name='test',
                          nominal_bags=['n1','n2','n3','n4'], dino_calibration_bags=['calibration'])
            signature, _, _ = _signature(config, '/camera', 'euclidean')
            save_feature_cache(Path(directory)/'cache/test/camera', data, 4, signature)
            with patch('vision_dinov2.anomaly_detection.build_nominal_video_memory', return_value=([data[0]],4)) as extract:
                prepare_camera(config, '/camera', Mock(), Path(directory))
            self.assertEqual(extract.call_count, 1)

    def test_stage_constrained_scoring_uses_numbered_stages_and_starts_at_stage_one(self):
        from tests.test_dino_metrics import memory
        from rynnbrain_vlm.visual_matching import RecordedStages
        from vision_dinov2.threshold import calibrate_attention_leave_one_video_out
        data = memory()
        table, _ = calibrate_attention_leave_one_video_out(data, n_bins=2, stage_constrained_matching=True)
        signature = dict(stage_constrained_matching=True, sample_fps=2., start_sec=None, end_sec=None,
                         top_k_cls=5, attention_power=1.5, min_weight=.1)
        image = Image.new('RGB', (56,56))
        extractor = Mock(device='cpu')
        extractor.extract.return_value = data[0]['features'], data[0]['attention'], data[0]['cls'], 4
        stages = RecordedStages((1.,2.,3.), ((1.2,'Error'),))
        with patch('rynnbrain_vlm.visual_matching.read_recorded_stages', return_value=stages), \
             patch('vision_dinov2.dinov2_features.iter_source_frames', return_value=(x for x in [(0,1.5,image)])) as source:
            rows = score_bag({}, '/camera', 'bag', data, {'euclidean':table}, signature, extractor)
        self.assertEqual(source.call_args.kwargs['start_sec'], 1.)
        self.assertEqual(rows[0]['matched_frame_id'], 1)

    def test_full_run_folder_reports_and_partial_camera_error_exclusion(self):
        from tests.test_dino_metrics import memory
        data = memory()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['train/one', 'train/two', 'train/three', 'selected/normal', 'selected/fail', 'outside/unselected']:
                path = root/name
                path.mkdir(parents=True)
                (path/'metadata.yaml').write_text('')
            config = dict(output_dir=str(root/'outputs'), nominal_bags=[str(root/f'train/{x}') for x in ['one','two','three']],
                          camera_topics=['/one', '/two'], dino_threshold_scale=1., stage_constrained_matching=False)
            def annotated(rows, *args):
                return [dict(item, ground_truth_label='normal' if item['bag_name']=='normal' else 'fail',
                             recorded_failure_time_sec=None if item['bag_name']=='normal' else 1.) for item in rows]
            def score(config, topic, bag, *args):
                if topic == '/two' and Path(bag).name == 'fail':
                    raise ValueError('missing camera')
                return [row for metric in ['euclidean','mahalanobis'] for row in
                        timeline(bag, metric, [.1, 2.] if Path(bag).name == 'fail' else [.1, .2], camera=topic)]
            with patch('vision_dinov2.benchmark.annotate_bags', side_effect=annotated), \
                 patch('vision_dinov2.benchmark.prepare_camera', return_value=(data, {}, {}, [], {})), \
                 patch('vision_dinov2.benchmark.score_bag', side_effect=score), \
                 patch('vision_dinov2.dinov2_features.DINOv2FeatureExtractor') as model:
                output = run(config, ['selected'], root)
            model.assert_called_once()
            results = pd.read_csv(output/'bag_results.csv')
            self.assertEqual(set(results.bag_name), {'normal','fail'})
            self.assertTrue(results[results.bag_name.eq('fail')].predicted_decision.isna().all())
            self.assertEqual(json.loads((output/'run.json').read_text())['status'], 'completed_with_errors')
            self.assertTrue((output/'roc.png').is_file())
            self.assertTrue((output/'runtime_summary.csv').is_file())
            self.assertFalse('dino_metric' in config)  # user's config wasn't mutated
