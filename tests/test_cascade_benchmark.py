import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image

from rynnbrain_vlm import cascade_benchmark as benchmark

CONFIG = {'camera_topics': ['top'], 'rynnbrain': {
    'model': {'model_id': 'base-vlm'}, 'reference_bags': ['/data/Nominal/a/ref'], 'num_frames': 8,
    'task_description': 'Pick the scissor and place it in the basket.'}}
CROSSINGS = {'/data/Nominal/a/n1': {2., 2.5, 3., 6., 6.5, 7.}, '/data/Fail/b/f1': {5.5, 6., 6.5},
             '/data/Nominal/a/ref': {1., 1.5, 2.}}


def saved_run(directory):
    """A saved DINO benchmark: one normal bag, one failure, the VLM reference and a memory bag."""
    (directory / 'run.json').write_text(json.dumps(dict(
        score_variant='pooled', configured_scale=1., stage_topic='/stage', selected_folders=['Nominal/a', 'Fail/b'],
        bag_rule='two_consecutive_same_camera_before_stage3_v1')))
    bags, frames = [], []
    for metric in ('euclidean', 'mahalanobis'):
        for path, label, status, saved_alarm in [('/data/Nominal/a/n1', 'normal', 'ok', 2.5),
                                                 ('/data/Fail/b/f1', 'fail', 'ok', 6.),
                                                 ('/data/Nominal/a/ref', 'normal', 'ok', 1.5),
                                                 ('/data/Nominal/a/mem', 'normal', 'excluded_training', None)]:
            folder, name = path.rsplit('/', 1)
            bags.append(dict(bag_path=path, bag_name=name, folder=folder[len('/data/'):], status=status,
                             ground_truth_label=label, recorded_failure_time_sec=5. if label == 'fail' else None,
                             metric=metric, operating_point='configured', threshold_scale=1.,
                             predicted_decision='failure' if saved_alarm else None, correct=None,
                             first_alarm_time_sec=saved_alarm, timing_status='not_scored',
                             failure_time_error_sec=None, processing_sec_mean=.2))
            if status == 'ok':
                frames += [dict(bag_path=path, camera_topic='top', metric=metric, timestamp_sec=t / 2,
                                pooled_score=2. if t / 2 in CROSSINGS[path] else .5, base_threshold=1.)
                           for t in range(0, 27)]  # 0 to 13 s; stage 3 at 8 s, window ends at 13 s.
    pd.DataFrame(bags).to_csv(directory / 'bag_results.csv', index=False)
    pd.DataFrame(frames).to_csv(directory / 'frame_results.csv', index=False)


class FakeChecker:
    calls = []

    def __init__(self, config, progress):
        self.topic, self.reference_response = 'top', 'The robot places the scissor in the basket.'
        self.model = type('Model', (), {'max_image_size': 64})

    def ask(self, frames, boundaries, alarm_sec):
        FakeChecker.calls.append(alarm_sec)
        verdict = 'failure' if alarm_sec == 6.5 and frames[-1][0] <= alarm_sec else 'safe'
        images = [(f'Frame {k + 1}', Image.new('RGB', (8, 8))) for k in range(2)]
        return dict(stage=benchmark.current_stage(boundaries, alarm_sec), frames=[], prompt='p',
                    answer=f'Decision: {verdict}', verdict=verdict, vlm_sec=1.5), images


class CascadeBenchmarkTests(unittest.TestCase):
    def test_saved_dino_run_is_rescored_with_cached_vlm_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            saved_run(run_dir)
            frames = [(t / 10, ('image', Image.new('RGB', (8, 8)))) for t in range(0, 140)]
            FakeChecker.calls = []
            with patch.object(benchmark, 'stage_boundaries', return_value={1: 0., 2: 4., 3: 8.}), \
                 patch.object(benchmark, 'camera_frames', return_value=frames), \
                 patch.object(benchmark, 'VlmChecker', FakeChecker):
                output = benchmark.run(CONFIG, run_dir, progress=lambda _m: None)
            # Normal bag: two episodes, both safe. Failure bag: one episode, failure. Mahalanobis reuses them all.
            self.assertEqual(FakeChecker.calls, [3., 7., 6.5])
            results = pd.read_csv(output / 'bag_results.csv')
            row = lambda setup, bag: results[(results.metric == 'euclidean') & (results.operating_point == setup)
                                             & (results.bag_path == bag)].iloc[0]
            self.assertEqual(row(benchmark.TRIGGER, '/data/Nominal/a/n1').first_alarm_time_sec, 3.)
            self.assertEqual(row(benchmark.CHECKED, '/data/Nominal/a/n1').predicted_decision, 'success')
            failure = row(benchmark.CHECKED, '/data/Fail/b/f1')
            self.assertEqual((failure.predicted_decision, failure.timing_status, failure.failure_time_error_sec),
                             ('failure', 'compared', 1.5))
            # The VLM's reference execution is left out of every setup, including the saved one.
            self.assertTrue((results[results.bag_path == '/data/Nominal/a/ref'].status == 'excluded_vlm_reference').all())
            metrics = pd.read_csv(output / 'metrics.csv')
            overall = metrics[(metrics.metric == 'euclidean') & (metrics.folder == 'ALL')].set_index('operating_point')
            self.assertEqual(overall.loc[benchmark.SAVED, 'false_positive_rate'], 1.)
            self.assertEqual(overall.loc[benchmark.CHECKED, 'false_positive_rate'], 0.)
            self.assertEqual(overall.loc[benchmark.CHECKED, 'recall'], 1.)
            self.assertEqual(int(overall.loc[benchmark.CHECKED, 'evaluated_bags']), 2)
            for name in ('metrics_euclidean.png', 'metrics_mahalanobis.png', 'SUMMARY.md', 'run.json'):
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual(len((output / 'checks.jsonl').read_text().splitlines()), 3)
            self.assertEqual(len(list((output / 'frames').rglob('*.jpg'))), 6)
            # A rerun answers everything from the cache, without loading the VLM.
            with patch.object(benchmark, 'stage_boundaries', return_value={1: 0., 2: 4., 3: 8.}), \
                 patch.object(benchmark, 'VlmChecker', side_effect=AssertionError('VLM must not be loaded')):
                benchmark.run(CONFIG, run_dir, progress=lambda _m: None)
            self.assertEqual(json.loads((output / 'run.json').read_text())['vlm_checks_new'], 0)

    def test_episodes_stop_with_the_execution_window(self):
        timeline = pd.DataFrame(dict(camera_topic='top', timestamp_sec=[1., 1.5, 2., 13.5, 14., 14.5],
                                     pooled_score=[2.] * 6, base_threshold=[1.] * 6))
        episodes = benchmark.alarm_episodes(timeline, {1: 0., 2: 4., 3: 8.}, scale=1., post_sec=5.)
        self.assertEqual([e['time_sec'] for e in episodes], [2.])  # 13.5 s onward is after stage 3 + 5 s.
        self.assertEqual(episodes[0]['stage'], 1)


if __name__ == '__main__':
    unittest.main()
