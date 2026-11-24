from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from botfails_benchmark.phases import Phases, gripper_events
from botfails_benchmark.prompts import holds_probability, task_questions
from botfails_benchmark.report import episode_outcome, first_alarm, rates

FPS = 30


def signal(*parts):
    """Concatenate (seconds, value) pieces into a 30 Hz signal."""
    return np.concatenate([np.full(int(seconds * FPS), value, float) for seconds, value in parts])


class GripperPhaseTests(unittest.TestCase):
    def test_a_closed_gripper_stopped_by_an_object_holds_until_it_opens(self):
        command = signal((1, 20), (2, 0), (1, 20))
        position = signal((1, 20), (2, 3), (1, 20))  # The object keeps the fingers at 3.
        events = gripper_events(position, command)
        self.assertEqual([kind for _, kind in events], ['hold', 'release'])
        self.assertAlmostEqual(events[0][0], 1.3, delta=.05)  # Closed for DWELL_SEC first.
        self.assertAlmostEqual(events[1][0], 3.3, delta=.05)

    def test_closing_an_empty_gripper_is_not_a_hold(self):
        command = signal((1, 20), (2, 0))
        position = signal((1, 20), (2, .3))  # Parked: fingers fully closed.
        self.assertEqual(gripper_events(position, command), [])

    def test_an_object_slipping_out_of_the_closed_gripper_is_a_release(self):
        command = signal((1, 20), (3, 0))
        position = signal((1, 20), (1, 3), (2, .1))
        self.assertEqual([kind for _, kind in gripper_events(position, command)], ['hold', 'release'])

    def test_phase_index_and_kind(self):
        phases = Phases([(2., 'hold'), (5., 'release')])
        self.assertEqual([phases.at(t) for t in (1., 2., 4.9, 5., 9.)],
                         [(0, 'start'), (1, 'holding'), (1, 'holding'), (2, 'released'), (2, 'released')])


TASK = {'checks': [
    {'name': 'probe', 'kind': 'choice', 'phases': ['released'], 'lead': 'Where is the probe?',
     'A': 'In the black socket', 'B': 'In a red socket', 'C': 'On its stand', 'holds': ['A', 'C']},
    {'name': 'spill', 'kind': 'yesno', 'question': 'Is coffee on the table?', 'holds': ['no']}]}


class PromptTests(unittest.TestCase):
    def test_instruction_covers_empty_tasks_and_uncovered_phases(self):
        for phase in ('start', 'holding', 'released'):
            for checks in ([], TASK['checks']):
                questions = task_questions({'instruction': 'Pick the iron, solder, return it to its holder.', 'checks': checks}, phase)
                self.assertTrue(questions)
                if phase == 'start' or not checks:
                    self.assertEqual(questions[0]['name'], f'task_context_{phase}')
                    self.assertIn('work in progress', questions[0]['text'])
                    self.assertEqual(questions[0]['holds'], ('no',))

    def test_no_question_before_the_first_grasp(self):
        self.assertEqual(task_questions(TASK, 'start'), [])

    def test_phase_selects_the_checks_and_the_template_is_shared(self):
        self.assertEqual([q['name'] for q in task_questions(TASK, 'holding')], ['spill'])
        released = task_questions(TASK, 'released')
        self.assertEqual([q['name'] for q in released], ['probe', 'spill'])
        self.assertTrue(released[0]['text'].endswith('C) On its stand.\nAnswer with one letter: A, B or C.'))
        self.assertEqual(released[1]['options'], ('yes', 'no'))

    def test_holds_probability_sums_the_accepted_answers(self):
        self.assertAlmostEqual(holds_probability([.2, .5, .3], ('A', 'B', 'C'), ('A', 'C')), .5)

    def test_holds_must_be_a_proper_subset(self):
        with self.assertRaises(ValueError):
            task_questions({'checks': [dict(TASK['checks'][1], holds=['yes', 'no'])]}, 'holding')


class CheckerTests(unittest.TestCase):
    def test_alarm_frames_and_the_weakest_sub_goal_decide(self):
        from botfails_benchmark.cascade import Checker
        answers = {'probe': [.1, .7, .2], 'spill': [.2, .8]}  # P(probe holds)=.3 < .5, P(spill holds)=.8

        class Model:
            def answer_distribution(self, turn, options):
                return answers['probe' if options == ('A', 'B', 'C') else 'spill']

        with tempfile.TemporaryDirectory() as directory:
            config = {'rynnbrain': {'model': {'model_id': 'x'}}}
            checker = Checker(config, {'task': TASK}, 'cam', Path(directory) / 'cache.jsonl', print)
            checker.model = Model()
            frames = [(i * 15, i * .5, Path(directory) / f'{i}.jpg') for i in range(20)]
            from PIL import Image
            for _, _, path in frames:
                Image.new('RGB', (8, 8)).save(path)
            check = checker.ask('task', 'released', frames, 5.)
            self.assertEqual([f['time_sec'] for f in check['frames']], [5., 4.5, 4.])  # Never a later frame.
            self.assertEqual(check['verdict'], 'failure')
            self.assertAlmostEqual(check['subgoal_probability'], .3)
            self.assertEqual(checker.asked, 6)
            # Cached answers are reused by a new checker.
            again = Checker(config, {'task': TASK}, 'cam', Path(directory) / 'cache.jsonl', print)
            again.model = None
            self.assertEqual(again.ask('task', 'released', frames, 5.)['verdict'], 'failure')
            self.assertEqual(again.asked, 0)


class FailureLabelTests(unittest.TestCase):
    def test_latent_recording_metrics_separate_early_stops_and_misses(self):
        import json
        from botfails_benchmark.latent import summarize
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / 'dataset/BotFails', root / 'report'
            task = 'industrial_soldering'
            labels = root / 'failure_labels' / (task + '_anomaly')
            records = output / 'episodes' / task
            labels.mkdir(parents=True)
            records.mkdir(parents=True)
            for index, stop in enumerate([None, 15., 5., None]):
                episode = f'episode_{index:06d}'
                y = np.zeros(900, int)
                if index:
                    y[300:] = 1
                np.savetxt(labels / (episode + '_labels.csv'), y, fmt='%d')
                (records / (episode + '.json')).write_text(json.dumps(dict(task=task, episode=episode,
                    split='test', first_alarm_sec=stop, observations=[dict(score=.2)])))
            summarize(output, dataset)
            result = json.loads((output / 'summary.json').read_text())[0]
            self.assertEqual(result['recording_accuracy'], 50.)
            self.assertEqual((result['detected_after_onset'], result['early_stops'], result['missed']), (1, 1, 1))
            self.assertEqual(result['false_stops'], 0)

    def test_calibration_preserves_boundary_and_handles_too_few_samples(self):
        import json
        import subprocess
        import sys
        script = Path(__file__).resolve().parents[1] / 'botfails_benchmark/experiments/cal_thresholds.py'
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'scores.jsonl', Path(directory) / 'threshold.json'
            values = [.9, .91, .92, .93, .9526912458240986] + [.97] * 45
            def write(values):
                source.write_text('\n'.join(json.dumps(dict(tag=dict(task='task', question='q', holds=['no']),
                    options=['yes', 'no'], distributions=[[1-v, v]])) for v in values))
            write(values)
            subprocess.run([sys.executable, str(script), str(source), str(output), '.1'], check=True, capture_output=True)
            threshold = json.loads(output.read_text())['task']['q']
            self.assertEqual(threshold, values[4])
            self.assertEqual(sum(v < threshold for v in values), 4)
            write([.95])
            subprocess.run([sys.executable, str(script), str(source), str(output), '.05'], check=True, capture_output=True)
            self.assertEqual(json.loads(output.read_text())['task']['q'], 0.)

    def test_task_failures_and_user_adjudicated_interventions_retain_timing(self):
        from botfails_benchmark.failure_labels import derive
        raw = np.array([0, 0, 1, 1])
        labels, status, _ = derive('domotic_pouringCoffee', 'episode_000000', raw)
        np.testing.assert_array_equal(labels, raw)
        self.assertEqual(status, 'failure')
        self.assertFalse(np.shares_memory(raw, labels))
        labels, status, _ = derive('industrial_soldering', 'episode_000010', raw)
        np.testing.assert_array_equal(labels, raw)
        self.assertEqual(status, 'failure')

    def test_normal_examples_remain_negative(self):
        from botfails_benchmark.failure_labels import derive
        labels, status, _ = derive('domotic_makingCoffee', 'episode_000005', np.zeros(4, int))
        self.assertEqual(status, 'normal')
        self.assertFalse(labels.any())

    def test_frame_fpr_is_complement_of_specificity(self):
        result = rates([0, 0, 0, 1], [0, 1, 0, 1])
        self.assertAlmostEqual(result['FPR'], 100 / 3)
        self.assertAlmostEqual(result['FPR'] + result['TNR'], 100)

    def test_report_uses_failure_frame_indices_and_excludes_unknown_episodes(self):
        import json
        from unittest.mock import patch
        from botfails_benchmark import report
        task = 'domotic_makingCoffee'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores_dir, labels_dir, cascade_dir = root / 'scores' / task, root / 'failure_labels' / (task + '_anomaly'), root / 'cascade'
            for path in (scores_dir, labels_dir, cascade_dir):
                path.mkdir(parents=True)
            rows, runs = [], []
            for index in range(3):
                episode = f'episode_{index:06d}'
                rows.extend(dict(episode=episode, frame_index=f, timestamp_sec=f/30,
                                 pooled_score=2., base_threshold=1., label=int(f == 30)) for f in (0, 15, 30))
                labels = np.zeros(31, int)
                if index == 1:
                    labels[15:] = 1
                if index == 2:
                    labels[:] = -1
                np.savetxt(labels_dir / f'{episode}_labels.csv', labels, fmt='%d')
                runs.append(dict(episode=episode, stop_sec=[None, .5, 0.][index], alarms=[], phases=[]))
            pd.DataFrame(rows).to_csv(scores_dir / 'test_scores.csv', index=False)
            (cascade_dir / (task + '.json')).write_text(json.dumps(runs))
            args = SimpleNamespace(output=root / 'report', dataset=root / 'dataset', threshold_scale=1.1,
                                   label_type='failure', labels_dir=labels_dir.parent, dino_run=root / 'scores',
                                   cascade_run=cascade_dir, split_file=None, subset=None, no_timelines=True)
            with patch.object(report, 'TASKS', (task,)), \
                 patch.object(report, 'threshold_free', return_value={'AUROC': 50., 'F1_opt': 50.}), \
                 patch.object(report, 'first_alarm', return_value=None), \
                 patch.object(report, 'write_markdown'), patch.object(report, 'plot_comparison'):
                result = report.run(args, lambda _: None)
            self.assertEqual((result['included_episodes'], result['excluded_episodes']), (2, 1))
            cascade = next(row for row in result['end_to_end'] if row['task'] == task and row['system'].endswith('(cascade)'))
            self.assertEqual((cascade['TPR'], cascade['TNR'], cascade['FPR']), (100., 100., 0.))


class ImprovedCascadeTests(unittest.TestCase):
    def fake(self, scores, events=(), decision=None, checks=None, verdicts=None):
        """An episode replay with a fake checker that answers from `verdicts` by trigger."""
        from unittest.mock import patch
        from botfails_benchmark import cascade
        calls = []

        class FakeChecker:
            tasks = {'task': {'checks': checks or [], **({'decision': decision} if decision else {})}}

            def ask(self, task, kind, frames, t, trigger='dino'):
                calls.append((t, kind, trigger))
                return dict(verdict=(verdicts or {}).get(trigger, 'safe'), subgoal_probability=.5, questions=[], frames=[])

        episode = SimpleNamespace(name='episode_000000', split='test', frame_table=lambda camera: [])
        frame = pd.DataFrame(dict(timestamp_sec=np.arange(len(scores)) * .5, frame_index=np.arange(len(scores)) * 15,
                                  base_threshold=1., pooled_score=scores))
        return frame, calls, FakeChecker(), episode, patch.object(cascade, 'episode_phases', return_value=Phases(list(events)))

    def test_legacy_dino_routing_cannot_bypass_a_safe_vlm_answer(self):
        from botfails_benchmark.cascade import replay_episode
        frame, calls, checker, episode, phases = self.fake([2, 2, 2, 2], decision='dino')
        with phases:
            result = replay_episode('task', episode, frame, checker, 'cam', 1.1)
        self.assertIsNone(result['stop_sec'])
        self.assertEqual(calls, [(1., 'start', 'dino')])

    def test_three_consecutive_alarms_require_a_failing_vlm_answer_to_stop(self):
        from botfails_benchmark.cascade import replay_episode
        frame, calls, checker, episode, phases = self.fake([2, 2, .1, 2, 2, 2], verdicts={'dino': 'failure'})
        with phases:
            result = replay_episode('task', episode, frame, checker, 'cam', 1.1)
        self.assertEqual(result['stop_sec'], 2.5)
        self.assertEqual(calls, [(2.5, 'start', 'dino')])

    def test_a_persisting_alarm_is_checked_again(self):
        from botfails_benchmark.cascade import replay_episode
        frame, calls, checker, episode, phases = self.fake([2] * 12)
        with phases:
            replay_episode('task', episode, frame, checker, 'cam', 1.1, recheck_sec=2.)
        self.assertEqual([(t, trigger) for t, _, trigger in calls], [(1., 'dino'), (3., 'recheck'), (5., 'recheck')])

    def test_legacy_release_only_checks_are_rejected(self):
        from botfails_benchmark.cascade import replay_episode
        checks = [dict(name='box', kind='yesno', question='q?', holds=['no'], triggers=['release'])]
        frame, calls, checker, episode, phases = self.fake([.1] * 12, [(1., 'hold'), (2., 'release')], checks=checks, verdicts={'release': 'failure'})
        with phases, self.assertRaisesRegex(ValueError, 'Release-only checks were removed'):
            replay_episode('task', episode, frame, checker, 'cam', 1.1, release_delay=1.5)
        self.assertEqual(calls, [])

    def test_release_does_not_trigger_a_check_without_a_dino_alarm(self):
        from botfails_benchmark.cascade import replay_episode
        frame, calls, checker, episode, phases = self.fake([.1] * 12, [(1., 'hold'), (2., 'release')])
        with phases:
            replay_episode('task', episode, frame, checker, 'cam', 1.1)
        self.assertEqual(calls, [])

    def test_triggers_select_the_questions(self):
        config = {'checks': [dict(name='a', kind='yesno', question='q?', holds=['no']),
                             dict(name='b', kind='yesno', question='q?', holds=['no'], triggers=['dino', 'release'])]}
        self.assertEqual([q['name'] for q in task_questions(config, 'released')], ['a', 'b'])
        self.assertEqual([q['name'] for q in task_questions(config, 'released', 'release')], ['b'])

    def test_a_normal_data_threshold_replaces_the_fixed_half(self):
        from botfails_benchmark.cascade import Checker

        class Model:
            def answer_distribution(self, turn, options):
                return [.2, .8]  # P(no) = 0.8

        task = {'checks': [dict(name='spill', kind='yesno', question='q?', holds=['no'])]}
        with tempfile.TemporaryDirectory() as directory:
            from PIL import Image
            path = Path(directory) / 'a.jpg'
            Image.new('RGB', (8, 8)).save(path)
            frames = [(0, 0., path)]
            config = {'rynnbrain': {'model': {'model_id': 'x'}}}
            default = Checker(config, {'task': task}, 'cam', Path(directory) / 'c1.jsonl', print)
            default.model = Model()
            self.assertEqual(default.ask('task', 'released', frames, 0.)['verdict'], 'safe')
            strict = Checker(config, {'task': task}, 'cam', Path(directory) / 'c2.jsonl', print, {'task': {'spill': .9}})
            strict.model = Model()
            self.assertEqual(strict.ask('task', 'released', frames, 0.)['verdict'], 'failure')

    def test_a_crop_selects_the_region_and_is_enlarged(self):
        from PIL import Image
        from botfails_benchmark.data import load_image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'a.png'
            image = Image.new('RGB', (100, 100), 'white')
            image.paste('black', (50, 0, 100, 100))
            image.save(path)
            crop = load_image(path, (.5, 0, 1, 1))
        self.assertEqual(max(crop.size), 640)
        self.assertEqual(crop.getpixel((160, 320))[0], 0)  # Only the black half is left.


class MetricTests(unittest.TestCase):
    def test_fidel_rates(self):
        labels = [0, 0, 0, 1, 1, 1, 1, 1]
        predictions = [0, 1, 0, 1, 1, 0, 1, 1]
        result = rates(labels, predictions)
        self.assertAlmostEqual(result['TPR'], 80.)
        self.assertAlmostEqual(result['TNR'], 100 * 2 / 3)
        self.assertAlmostEqual(result['Acc'], (80 + 100 * 2 / 3) / 2)
        beta = 3 / 8  # Normal frames / all frames.
        self.assertAlmostEqual(result['WAcc'], beta * 100 * 2 / 3 + (1 - beta) * 80)

    def test_first_alarm_needs_three_frames_in_a_row(self):
        frame = pd.DataFrame(dict(timestamp_sec=np.arange(8) * .5, base_threshold=1.,
                                  pooled_score=[2, 2, 0, 2, 2, 2, 2, 0]))
        self.assertEqual(first_alarm(frame, 1.1), 2.5)

    def test_episode_outcomes(self):
        self.assertEqual(episode_outcome(False, None, None), 'correct_continue')
        self.assertEqual(episode_outcome(False, None, 3.), 'false_stop')
        self.assertEqual(episode_outcome(True, 10., None), 'missed')
        self.assertEqual(episode_outcome(True, 10., 9.5), 'stopped_before_onset')
        self.assertEqual(episode_outcome(True, 10., 12.), 'detected')


if __name__ == '__main__':
    unittest.main()
