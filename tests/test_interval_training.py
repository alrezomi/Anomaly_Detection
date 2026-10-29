from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rynnbrain_vlm.execution_sampling import (execution_sampling, interval_execution_inputs,
    validate_interval_replay)
from rynnbrain_vlm.temporal_lora import training_profile, prefix_samples, validate_profile
from rynnbrain_vlm.cop_timeline import (prepare_nominal_prefixes, _load_prefix_cache,
    signature_sampling, timeline_sampling)
from rynnbrain_vlm.prompt_comparison import (comparison_variants, comparison_prompts,
    run_prompt_comparison)
from tests.test_benchmark_replay import FakeDetector, packets, _scan_bag


class IntervalTrainingTests(unittest.TestCase):
    def config(self):
        return dict(camera_topics=['top'], rynnbrain=dict(source='rosbag', input_modes=['raw'],
            num_frames=8, camera_topics=['top'], memory_camera_topics=['top'], reference_bags=['reference'],
            model={'model_id': 'test', 'lora_adapter_path': None}, execution_interval_sec=3.,
            sampling_start_sec=1., sampling_end_sec=None,
            online_replay={'interval_mode': 'constant', 'max_observations': None},
            lora={'training': {'temporal_supervision': True, 'input_mode': 'raw'}},
            cop_classifier={'method': 'logistic'}))

    def test_recording_interval_preserves_real_times_and_matches_replay_over_eight_frames(self):
        config = self.config()
        source = packets([(i / 10, 26) for i in range(380)])
        policy = execution_sampling(config['rynnbrain'])
        images, metadata = interval_execution_inputs('bag', ['top'], policy, frames=iter(source))
        self.assertEqual(len(images), 13)
        self.assertEqual([row['timestamp_sec'] for row in metadata], [float(i) for i in range(1, 38, 3)])
        self.assertIn('Frame 4, camera top, time 10.000 s', images[3][0])
        with tempfile.TemporaryDirectory() as directory:
            detector = FakeDetector()
            result = _scan_bag(detector, config, Path('bag'), Path(directory) / 'scan.npz', iter(source))
            self.assertTrue(result['complete'])
            self.assertEqual([row['timestamp_sec'] for row in result['rows']],
                             [row['timestamp_sec'] for row in metadata])

    def test_jitter_uses_actual_selected_time_for_next_deadline_and_closes_source(self):
        policy = execution_sampling(self.config()['rynnbrain'])
        closed = []
        def source():
            try:
                yield from [(0, .5, None), (1, 1.04, None), (2, 4.0, None), (3, 4.08, None),
                            (4, 7.02, None), (5, 7.12, None), (6, 10.2, None)]
            finally:
                closed.append(True)
        _, frames = interval_execution_inputs('bag', ['top'], {**policy, 'end_sec': 8.}, frames=source())
        self.assertEqual([row['timestamp_sec'] for row in frames], [1.04, 4.08, 7.12])
        self.assertEqual(closed, [True])

    def test_profile_tracks_execution_interval_and_reference_stays_eight(self):
        config = self.config()
        profile = training_profile(config, config['rynnbrain']['lora']['training'])
        self.assertEqual(profile['num_frames'], 8)
        self.assertEqual(profile['execution_sampling']['interval_sec'], 3.)
        validate_profile(profile, config, config['rynnbrain'], 8, ['raw'])
        validate_interval_replay(config['rynnbrain'])
        for options in ({'interval_sec': 9.}, {'max_observations': 8}, {'interval_mode': 'adaptive'}):
            bad = deepcopy(config['rynnbrain'])
            bad['online_replay'].update(options)
            with self.assertRaisesRegex(ValueError, 'match training'):
                validate_interval_replay(bad)
        with self.assertRaisesRegex(ValueError, 'differ'):
            validate_profile(profile, config, {**config['rynnbrain'], 'execution_interval_sec': 9.}, 8, ['raw'])
        for value in (True, 0, -1, float('nan'), '3'):
            with self.assertRaises(ValueError):
                execution_sampling({**config['rynnbrain'], 'execution_interval_sec': value})
        self.assertIsNone(execution_sampling({}))

    def test_interval_training_creates_all_prefixes_with_correct_onset_labels(self):
        config = self.config()
        profile = training_profile(config, config['rynnbrain']['lora']['training'])
        images, frames = interval_execution_inputs('bag', ['top'], execution_sampling(config['rynnbrain']),
            frames=packets([(i, 26) for i in range(35)]))
        row = dict(bag_name='bag', label='fail', split='train')
        samples = prefix_samples(row, images, frames, profile, {'time_sec': 8.})
        self.assertEqual(len(samples), 12)
        self.assertEqual([len(item['images']) for item in samples], list(range(1, 13)))
        self.assertEqual([item['label'] for item in samples[:4]], ['normal', 'normal', 'normal', 'fail'])
        self.assertEqual(samples[3]['cutoff_sec'], 10.)
        self.assertEqual(samples[3]['target_text'], 'Decision: failure\nFailure onset (s): 8.000')
        self.assertEqual(len(samples[-1]['images']), len(images))
        self.assertTrue(all(frame['timestamp_sec'] <= sample['cutoff_sec']
                            for sample in samples for frame in frames[:len(sample['images'])]))

    def test_execution_loader_uses_interval_but_reference_sampler_is_not_replaced(self):
        from rynnbrain_vlm import run
        config = self.config()
        profile = training_profile(config, config['rynnbrain']['lora']['training'])
        vlm = {**config['rynnbrain'], '_temporal_profile': profile}
        images, frames = interval_execution_inputs('bag', ['top'], execution_sampling(vlm),
            frames=packets([(i, 26) for i in range(35)]))
        with patch('rynnbrain_vlm.execution_sampling.interval_execution_inputs', return_value=(images, frames)), \
             patch.object(run, '_raw_inputs', side_effect=AssertionError('Execution must use interval inputs')):
            selected, metadata = run._execution_inputs({**config, 'test_bag': 'bag', 'output_dir': 'output'}, vlm, 8, ['raw'])
        self.assertEqual(len(selected['raw'][0]), 12)
        self.assertEqual(metadata, frames)

    def test_classifier_prefix_cache_supports_different_recording_lengths(self):
        from rynnbrain_vlm import run
        config = self.config()
        vlm = config['rynnbrain']
        profile = training_profile(config, vlm['lora']['training'])
        signature = dict(num_frames=8, temporal_profile=profile, turn1_prompt='reference', turn2_prompt='evaluation')
        sampling = timeline_sampling(vlm, 8)
        self.assertEqual(signature_sampling(signature), sampling)
        with tempfile.TemporaryDirectory() as directory:
            records, counts = [], {'short': 4, 'long': 12}
            for name, count in counts.items():
                records.append(SimpleNamespace(metadata=dict(bag_name=name, input_mode='raw',
                    test_bag=name, evaluation_id=name, comparison_signature=signature),
                    metadata_path=Path(directory)/name/'rynnbrain_multiturn/cop_vectors/raw.json',
                    vector_path=Path(directory)/name/'raw.npy', vector=np.array([count, 1., 1.], dtype=np.float32)))
            def inputs(cfg, *_):
                count = counts[cfg['test_bag']]
                frames = [dict(sample_index=i, timestamp_sec=1. + i * 3, topic='top') for i in range(count)]
                return {'raw': ([('frame', object()) for _ in frames], frames)}, frames
            def extract(turns, *_args, **_kwargs):
                return np.array([len(turns[1]['images']), 1., 1.], dtype=np.float32)
            model = SimpleNamespace(extract_multiturn_cop_vector=Mock(side_effect=extract))
            cached_calls = []
            class PrefixCache:
                def __init__(self, *args, **kwargs):
                    self.count = 0
                def append(self, label, image):
                    self.count += 1
                    cached_calls.append(self.count)
                    return np.array([self.count, 1., 1.], dtype=np.float32)
            with patch.object(run, '_execution_inputs', side_effect=inputs), \
                 patch('rynnbrain_vlm.online_cache.CachedExecution', PrefixCache):
                prepare_nominal_prefixes(model, config, vlm, records, [], 'reference', {}, require_timestamps=True)
            for record in records:
                vectors = _load_prefix_cache(record, require_timestamps=True)
                self.assertEqual(vectors.shape, (counts[record.metadata['bag_name']], 3))
                np.testing.assert_array_equal(vectors[:, 0], np.arange(1, len(vectors) + 1))
            model.extract_multiturn_cop_vector.assert_not_called()
            self.assertEqual(len(cached_calls), 14)

    def test_removed_browser_preview_defaults_off_and_dino_matching_remains(self):
        from rynnbrain_vlm.online_replay import ReplayApplication
        app = ReplayApplication({}, [Path('bag')], 3., False)
        self.assertFalse(app.snapshot()['match_preview'])
        html = (Path(__file__).resolve().parents[1]/'rynnbrain_vlm/online_replay.html').read_text(encoding='utf-8')
        self.assertNotIn('id="matchPreview"', html)
        self.assertIn('DINO selects one reference', html)

    def test_real_onset_fit_pools_all_variable_length_prefixes_and_loads_for_replay(self):
        from rynnbrain_vlm.cop_classifier import CoPLogisticClassifier, save_classifier
        from rynnbrain_vlm.cop_onset import train_onset_classifier, load_onset_classifier
        from rynnbrain_vlm.cop_timeline import _prefix_cache_paths
        from rynnbrain_vlm import run
        config = self.config()
        vlm = config['rynnbrain']
        profile = training_profile(config, vlm['lora']['training'])
        signature = dict(source='rosbag', num_frames=8, temporal_profile=profile,
                         turn1_prompt='reference', turn2_prompt='evaluation')
        targets = [0, 0, 1, 1]
        counts = [4, 5, 10, 12]
        records = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete_path = root/'raw_logistic.npz'
            complete_metadata = dict(input_mode='raw', representation_id='test', comparison_signature=signature,
                fit_diagnostics={'regularization_c': 1.}, class_weight='balanced', threshold=.7)
            complete = CoPLogisticClassifier(np.ones(3), 0., np.zeros(3), .7, 'raw', 'test', signature, complete_metadata)
            save_classifier(complete_path, complete)
            for index, count in enumerate(counts):
                name = f'bag_{index}'
                metadata = dict(bag_name=name, input_mode='raw', test_bag=name,
                                evaluation_id=name, comparison_signature=signature)
                record = SimpleNamespace(metadata=metadata,
                    metadata_path=root/name/'rynnbrain_multiturn/cop_vectors/raw.json',
                    vector_path=root/name/'raw.npy', vector=np.array([targets[index], count, 1.], dtype=np.float32))
                cache, sidecar = _prefix_cache_paths(record)
                cache.parent.mkdir(parents=True)
                times = [1. + step*3 for step in range(count)]
                vectors = np.array([[int(targets[index] and time >= 10.), step+1, 1.]
                                    for step, time in enumerate(times)], dtype=np.float32)
                np.savez_compressed(cache, vectors=vectors)
                sidecar.write_text(json.dumps(dict(protocol=profile['execution_sampling']['protocol'],
                    sampling=signature_sampling(signature), full_evaluation_id=name,
                    comparison_signature=signature,
                    snapshots=[{'timestamp_sec': time, 'sample_index': step} for step, time in enumerate(times)])))
                records.append(record)
            def annotation(bag, **kwargs):
                failed = int(Path(bag).name.split('_')[-1]) >= 2
                return dict(status='available' if failed else 'missing_marker', time_sec=10. if failed else None)
            with patch('rynnbrain_vlm.cop_onset.read_failure_annotation', side_effect=annotation):
                metadata = train_onset_classifier(records, targets, complete_path, complete_metadata,
                    {'topic': '/recording_stage', 'startup_ignore_sec': .1})
            self.assertEqual(metadata['training_prefix_count'], sum(counts))
            self.assertEqual(metadata['sampling']['interval_sec'], 3.)
            loaded, _ = load_onset_classifier(complete_path, complete, bag_name='held_out')
            self.assertEqual(loaded.weights.shape, (3,))

    def test_fixed_threshold_benchmark_reuses_gui_scan_without_selecting_another_threshold(self):
        from rynnbrain_vlm.benchmark_replay import run_threshold_search
        from tests.test_benchmark_replay import BenchmarkReplayTests
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = BenchmarkReplayTests().config(root)
            config['vlm_benchmark']['threshold_search'] = False
            config['rynnbrain']['execution_interval_sec'] = 3.
            config['rynnbrain']['online_replay'] = dict(interval_mode='constant', interval_sec=3.,
                                                       max_observations=None, stop_on_alarm=True)
            records = [dict(bag_name=name, bag_path=str(root/name), label=label, folder='held_out')
                       for name, label in [('normal', 'normal'), ('failure', 'fail')]]
            detector = FakeDetector()
            def source(bag, **kwargs):
                failed = bag.bag_path.name == 'failure'
                return iter(packets([(1+i*3, 230 if failed and i >= 2 else 26) for i in range(12)]))
            with patch('rosbag_io.iter_rosbag_image_frames', side_effect=source), \
                 patch('rynnbrain_vlm.benchmark_replay._read_timing_annotation',
                       return_value={'status': 'missing_marker', 'time_sec': None}), \
                 patch('rynnbrain_vlm.temporal_threshold.write_temporal_cost_report',
                       side_effect=AssertionError('Manual threshold must stay manual')):
                rows, output, effective = run_threshold_search(config, records, root, detector,
                                                               Mock(), .1, select_threshold=False)
            self.assertEqual(output, root)
            self.assertEqual(effective['rynnbrain']['cop_classifier']['training']['threshold'], .7)
            self.assertEqual(len(detector.calls), 24)
            self.assertEqual([row['runtime_frame_count'] for row in rows], [12, 3])
            self.assertEqual([row['classifier_decision'] for row in rows], ['success', 'failure'])
            self.assertFalse((root/'best_threshold').exists())
            self.assertTrue((root/'replay_observations/normal.csv').is_file())


class PromptComparisonTests(unittest.TestCase):
    def setUp(self):
        from tests.test_vlm_benchmark_training import BenchmarkTrainingTests
        fixture = BenchmarkTrainingTests('test_flag_is_opt_in_and_rejects_strings')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        from pipeline_settings import resolve_pipeline_config
        raw = json.loads(fixture.original)
        raw['rynnbrain']['prompt_variant'] = 'A'
        self.config = resolve_pipeline_config(raw)
        self.config['vlm_benchmark']['prompt_variants'] = ['A', 'B']
        self.config['rynnbrain']['execution_interval_sec'] = 3.
        self.prompts = self.enterContext(patch('rynnbrain_vlm.prompts.task_context_prompt_b',
            side_effect=lambda task, **kwargs: f'User prompt B reference: {task}'))

    def test_variants_require_distinct_actual_temporal_prompts_and_fresh_training(self):
        self.assertEqual(comparison_variants(self.config), ['A', 'B'])
        snapshots = comparison_prompts(self.config, ['A', 'B'])
        self.assertNotEqual(snapshots['A']['sha256'], snapshots['B']['sha256'])
        with patch('rynnbrain_vlm.prompts.task_context_prompt', return_value='same'), \
             patch('rynnbrain_vlm.prompts.evaluation_prompt_multiturn', return_value='same'):
            with self.assertRaisesRegex(ValueError, 'identical temporal'):
                comparison_prompts(self.config, ['A', 'B'])
        for values in (['A'], ['A', 'A'], ['A', 'C'], 'A,B'):
            with self.assertRaises(ValueError):
                comparison_variants({**self.config, 'vlm_benchmark': {'prompt_variants': values, 'train_lora': True}})

    def test_two_independent_pipelines_keep_original_config_and_capture_prompts(self):
        from pipeline_settings import load_pipeline_config
        from rynnbrain_vlm.benchmark_training import prepare_pipeline_run, save_reusable_config, resolve_trained_artifacts
        original = deepcopy(self.config)
        calls = []
        def child(command, **kwargs):
            path = Path(command[command.index('--config') + 1])
            config = load_pipeline_config(path)
            self.assertIsNone(config['rynnbrain']['model']['lora_adapter_path'])
            self.assertNotIn('prompt_variants', config['vlm_benchmark'])
            variant = config['rynnbrain']['prompt_variant']
            self.assertEqual(config['rynnbrain']['online_replay']['interval_sec'], 3.)
            self.assertIsNone(config['rynnbrain']['online_replay']['max_observations'])
            args = self.fixture.arguments()
            args.config = path
            args.benchmark_dir = Path(command[command.index('--benchmark-dir') + 1])
            prepared_args, saved, reports = prepare_pipeline_run(args, config)
            adapter = Path(saved['rynnbrain']['model']['lora_adapter_path'])
            adapter.mkdir()
            for name in ('adapter_config.json', 'training_manifest.json'):
                (adapter/name).write_text('{}')
            (adapter/'adapter_model.safetensors').write_bytes(b'fixture weights')
            classifier = Path(saved['rynnbrain']['cop_classifier']['model_paths']['raw'])
            classifier.parent.mkdir()
            classifier.write_bytes(b'fixture classifier')
            classifier.with_suffix('.json').write_text('{}')
            save_reusable_config(prepared_args.config, saved)
            reports.mkdir()
            (reports/'benchmark_runtime.json').write_text(json.dumps(dict(status='completed', benchmark_output_dir=str(reports))))
            (reports/'benchmark_summary.csv').write_text('bag_path,ground_truth_label\ntest_normal,normal\ntest_fail,fail\n')
            (reports/'benchmark_statistics.json').write_text(json.dumps({'accuracy': .5}))
            (reports/'benchmark_frame_runtime.json').write_text(json.dumps({'average_processing_sec_per_frame': .2}))
            calls.append((variant, saved))
            process = Mock(args=command, stdout=iter([f'Completed {variant}\n']))
            process.wait.return_value = 0
            process.__enter__ = Mock(return_value=process)
            process.__exit__ = Mock(return_value=False)
            return process
        with patch('rynnbrain_vlm.prompt_comparison.subprocess.Popen', side_effect=child):
            root = run_prompt_comparison(self.fixture.arguments(), self.config)
        self.assertEqual([variant for variant, saved in calls], ['A', 'B'])
        self.assertNotEqual(calls[0][1]['rynnbrain']['model']['lora_adapter_path'],
                            calls[1][1]['rynnbrain']['model']['lora_adapter_path'])
        self.assertEqual(self.config, original)
        self.assertEqual(self.fixture.config_path.read_bytes(), self.fixture.original)
        report = json.loads((root/'prompt_comparison.json').read_text())
        self.assertEqual(len(report['runs']), 2)
        self.assertTrue((root/'prompt_comparison.csv').is_file())
        saved_prompts = json.loads((root/'prompt_texts.json').read_text())
        self.assertIn('User prompt B reference', saved_prompts['B']['texts']['turn1_prompt'])
        self.assertEqual(report['runs'][0]['average_processing_sec_per_frame'], .2)
        for variant in ('A', 'B'):
            from pipeline_settings import resolve_pipeline_config
            selection = json.loads(self.fixture.original)
            selection['rynnbrain']['prompt_variant'] = variant
            selection['rynnbrain']['execution_interval_sec'] = 3.
            selection = resolve_pipeline_config(selection)
            resolved = resolve_trained_artifacts(selection, progress=Mock())
            self.assertIn(f'prompt_{variant.lower()}', resolved['rynnbrain']['model']['lora_adapter_path'])
            self.assertEqual(resolved['rynnbrain']['model']['lora_adapter_path'],
                next(item['lora_adapter_path'] for item in report['runs'] if item['prompt_variant'] == variant))

    def test_failed_first_training_does_not_start_second_prompt(self):
        process = Mock(args=['benchmark'], stdout=iter(['training failed\n']))
        process.wait.return_value = 1
        process.__enter__ = Mock(return_value=process)
        process.__exit__ = Mock(return_value=False)
        import subprocess
        with patch('rynnbrain_vlm.prompt_comparison.subprocess.Popen', return_value=process) as child:
            with self.assertRaises(subprocess.CalledProcessError):
                run_prompt_comparison(self.fixture.arguments(), self.config)
        self.assertEqual(child.call_count, 1)


if __name__ == '__main__':
    unittest.main()
