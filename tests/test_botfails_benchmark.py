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

    def test_a_trusted_dino_alarm_stops_without_asking(self):
        from botfails_benchmark.cascade import replay_episode
        frame, calls, checker, episode, phases = self.fake([2, 2, 2, 2], decision='dino')
        with phases:
            result = replay_episode('task', episode, frame, checker, 'cam', 1.1)
        self.assertEqual(result['stop_sec'], 1.)
        self.assertEqual(calls, [])

    def test_a_persisting_alarm_is_checked_again(self):
        from botfails_benchmark.cascade import replay_episode
        frame, calls, checker, episode, phases = self.fake([2] * 12)
        with phases:
            replay_episode('task', episode, frame, checker, 'cam', 1.1, recheck_sec=2.)
        self.assertEqual([(t, trigger) for t, _, trigger in calls], [(1., 'dino'), (3., 'recheck'), (5., 'recheck')])

    def test_a_release_triggers_a_check_without_a_dino_alarm(self):
        from botfails_benchmark.cascade import replay_episode
        checks = [dict(name='box', kind='yesno', question='q?', holds=['no'], triggers=['release'])]
        frame, calls, checker, episode, phases = self.fake([.1] * 12, [(1., 'hold'), (2., 'release')], checks=checks, verdicts={'release': 'failure'})
        with phases:
            result = replay_episode('task', episode, frame, checker, 'cam', 1.1, release_delay=1.5)
        self.assertEqual(calls, [(3.5, 'released', 'release')])
        self.assertEqual(result['stop_sec'], 3.5)

    def test_no_release_check_without_a_release_question(self):
        from botfails_benchmark.cascade import replay_episode
        frame, calls, checker, episode, phases = self.fake([.1] * 12, [(1., 'hold'), (2., 'release')])
        with phases:
            replay_episode('task', episode, frame, checker, 'cam', 1.1, release_delay=1.5)
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
