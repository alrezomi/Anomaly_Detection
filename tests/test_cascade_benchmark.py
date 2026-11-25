import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image

from rynnbrain_vlm import cascade_benchmark as benchmark

CONFIG = {'camera_topics': ['top'], 'rynnbrain': {
    'model': {'model_id': 'base-vlm'}, 'task_description': 'Pick the scissor and place it in the basket.',
    'vlm_check': {'camera_topic': 'gripper', 'item': 'a metal surgical instrument', 'destination': 'a blue surface'}}}
CROSSINGS = {'/data/Nominal/a/n1': {2., 2.5, 3., 6., 6.5, 7.}, '/data/Fail/b/f1': {5.5, 6., 6.5}}


def saved_run(directory):
    """A saved DINO benchmark: one normal bag, one failure and a DINO memory bag."""
    (directory / 'run.json').write_text(json.dumps(dict(
        score_variant='pooled', configured_scale=1., stage_topic='/stage', selected_folders=['Nominal/a', 'Fail/b'],
        bag_rule='two_consecutive_same_camera_before_stage3_v1')))
    bags, frames = [], []
    for metric in ('euclidean', 'mahalanobis'):
        for path, label, status, saved_alarm in [('/data/Nominal/a/n1', 'normal', 'ok', 2.5),
                                                 ('/data/Fail/b/f1', 'fail', 'ok', 6.),
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
        self.topic = 'gripper'
        self.model = type('Model', (), {'max_image_size': 64})

    def ask(self, frames, boundaries, alarm_sec):
        FakeChecker.calls.append(alarm_sec)
        verdict = 'failure' if alarm_sec == 6.5 and frames[-1][0] <= alarm_sec else 'safe'
        images = [(f'Frame {k + 1}', Image.new('RGB', (8, 8))) for k in range(2)]
        return dict(stage=benchmark.current_stage(boundaries, alarm_sec), frames=[], questions=['q'],
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
                output = benchmark.run(CONFIG, run_dir, progress=lambda _m: None, output=run_dir / 'v2')
            self.assertEqual(output, run_dir / 'v2')
            # Normal bag: two episodes, both safe. Failure bag: one episode, failure. Mahalanobis reuses them all.
            self.assertEqual(FakeChecker.calls, [3., 7., 6.5])
            results = pd.read_csv(output / 'bag_results.csv')
            row = lambda setup, bag: results[(results.metric == 'euclidean') & (results.operating_point == setup)
                                             & (results.bag_path == bag)].iloc[0]
            normal = row(benchmark.CHECKED, '/data/Nominal/a/n1')
            self.assertEqual((normal.predicted_decision, json.loads(normal.alarm_episodes)), ('success', [3., 7.]))
            self.assertEqual(set(results.operating_point), {benchmark.SAVED, benchmark.CHECKED})
            failure = row(benchmark.CHECKED, '/data/Fail/b/f1')
            self.assertEqual((failure.predicted_decision, failure.timing_status, failure.failure_time_error_sec),
                             ('failure', 'compared', 1.5))
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
                benchmark.run(CONFIG, run_dir, progress=lambda _m: None, output=output)
            self.assertEqual(json.loads((output / 'run.json').read_text())['vlm_checks_new'], 0)

    def test_episodes_stop_with_the_execution_window(self):
        timeline = pd.DataFrame(dict(camera_topic='top', timestamp_sec=[1., 1.5, 2., 13.5, 14., 14.5],
                                     pooled_score=[2.] * 6, base_threshold=[1.] * 6))
        episodes = benchmark.alarm_episodes(timeline, {1: 0., 2: 4., 3: 8.}, scale=1., post_sec=5.)
        self.assertEqual([e['time_sec'] for e in episodes], [2.])  # 13.5 s onward is after stage 3 + 5 s.
        self.assertEqual(episodes[0]['stage'], 1)

    def test_an_alarm_persisting_into_a_new_stage_is_checked_again(self):
        times = [t / 2 for t in range(4, 21)]  # Above threshold from 2 s to 10 s.
        timeline = pd.DataFrame(dict(camera_topic='top', timestamp_sec=times, pooled_score=[2.] * len(times),
                                     base_threshold=[1.] * len(times)))
        episodes = benchmark.alarm_episodes(timeline, {1: 0., 2: 4., 3: 8.}, scale=1., post_sec=5.)
        self.assertEqual([(e['time_sec'], e['stage']) for e in episodes], [(3., 1), (5., 2), (9., 3)])


    def test_every_alarm_is_checked_and_judged_with_all_checks(self):
        class StopFirst(FakeChecker):
            def ask(self, frames, boundaries, alarm_sec):
                FakeChecker.calls.append(alarm_sec)
                verdict = 'failure' if alarm_sec in (3., 6.5) else 'safe'
                return dict(stage=benchmark.current_stage(boundaries, alarm_sec), frames=[], questions=['q'],
                            answer='', verdict=verdict, vlm_sec=1., subgoal_probability=.2 if verdict == 'failure'
                            else .8), [('Frame 1', Image.new('RGB', (8, 8)))]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            saved_run(run_dir)
            frames = [(t / 10, ('image', Image.new('RGB', (8, 8)))) for t in range(0, 140)]
            for all_checks, asked in ((False, [3., 6.5]), (True, [3., 7., 6.5])):
                FakeChecker.calls = []
                with patch.object(benchmark, 'stage_boundaries', return_value={1: 0., 2: 4., 3: 8.}), \
                     patch.object(benchmark, 'camera_frames', return_value=frames), \
                     patch.object(benchmark, 'VlmChecker', StopFirst):
                    output = benchmark.run(CONFIG, run_dir, metrics=['euclidean'], progress=lambda _m: None,
                                           output=run_dir / f'all_{all_checks}', all_checks=all_checks)
                self.assertEqual(FakeChecker.calls, asked)
            results = pd.read_csv(output / 'bag_results.csv')
            normal = results[(results.operating_point == benchmark.CHECKED) & (results.bag_path == '/data/Nominal/a/n1')]
            self.assertEqual(normal.first_alarm_time_sec.iloc[0], 3.)  # The bag decision stays the live rule's.
            checks = pd.read_csv(output / 'alarm_checks.csv')
            self.assertEqual(list(zip(checks.time_sec, checks.alarm_truth, checks.verdict, checks.after_stop)),
                             [(3., 'normal', 'failure', False), (7., 'normal', 'safe', True),
                              (6.5, 'failure', 'failure', False)])
            overall = pd.read_csv(output / 'alarm_metrics.csv').set_index('alarm').loc['All alarms']
            self.assertEqual((overall['true: stopped (right)'], overall['false: ignored (right)'],
                              overall['false: stopped (wrong)']), (1, 1, 1))
            self.assertIn('Per alarm', (output / 'SUMMARY.md').read_text())

    def test_alarm_truth_follows_the_failure_marker(self):
        failed = dict(ground_truth_label='fail', recorded_failure_time_sec=10.)
        self.assertEqual([benchmark.alarm_truth(failed, t) for t in (5., 9.5, 10., 30.)],
                         ['unclear', 'unclear', 'failure', 'failure'])
        self.assertEqual(benchmark.alarm_truth(dict(ground_truth_label='normal'), 5.), 'normal')
        self.assertEqual(benchmark.alarm_truth(dict(ground_truth_label='fail', recorded_failure_time_sec=float('nan')), 5.),
                         'unclear')


if __name__ == '__main__':
    unittest.main()
