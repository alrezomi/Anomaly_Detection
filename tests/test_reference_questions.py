import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from rynnbrain_vlm import reference_questions as reference
from rynnbrain_vlm.cascade import VlmChecker, question_text
from rynnbrain_vlm.prompts import QuestionSet

RED, BLUE = (200, 30, 30), (30, 30, 200)


def config(directory, item=None, destination=None):
    return {'nominal_bags': ['/data/n1', '/data/n2', '/data/n3'], 'nominal_cache_dir': str(directory),
            'vision_memory_name': 'memory', 'stage_topic': '/stage',
            'rynnbrain': {'model': {'model_id': 'base-vlm'},
                          'vlm_check': {'camera_topic': 'gripper', 'questions': 'reference',
                                        **({'item': item, 'destination': destination} if item else {})}}}


def frames(*_args):
    """Every reference run: the sub-goal holds in red frames and does not hold yet in blue ones."""
    red, blue = Image.new('RGB', (8, 8), RED), Image.new('RGB', (8, 8), BLUE)
    return {goal: dict(pos=[[red] * 3] * 3, neg=[blue] * 3) for goal in (2, 3)}


class FakeModel:
    """Names a red cube; only questions about the red cube (or its place) tell red from blue frames."""
    max_image_size = 64

    def __init__(self, separating=True):
        self.separating = separating

    def generate_nominal(self, turn, generation):
        text = turn['text']
        if 'noun phrase' in text:
            return 'A red cube.'
        if 'preposition' in text:
            return 'The red cube is on the [shelf] (12, 40).'
        return 'The gripper holds a red cube.'

    def answer_probability(self, turn, options):
        red = turn['images'][0][1].getpixel((0, 0)) == RED
        specific = self.separating and ('a red cube' in turn['text'] or 'on the shelf' in turn['text'])
        return (.9 if red else .2) if specific else .6


class TaskModel(FakeModel):
    """FakeModel that records the describing prompts and writes yes/no questions when asked to."""
    def __init__(self):
        super().__init__()
        self.prompts = []

    def generate_nominal(self, turn, generation):
        self.prompts.append(turn['text'])
        if 'Write only the question' in turn['text']:
            return 'Is a red cube between the fingers? Yes.' if 'grasp worked' in turn['text'] else 'Is a red cube on the shelf?'
        if 'noun phrase' in turn['text'] and "The robot's task" not in turn['text']:
            return 'A red block.'  # Not told the task, it names the part differently.
        return super().generate_nominal(turn, generation)


class ReferenceQuestionTests(unittest.TestCase):
    def test_descriptions_are_cleaned_into_noun_and_place_phrases(self):
        self.assertEqual(reference.clean('The object is a [black cable] (0, 582), below. More text.'),
                         'The object is a black cable, below')
        self.assertEqual(reference.noun_phrase('<usb plug>'), 'the usb plug')  # A name marked in its pointing format.
        self.assertEqual(reference.yes_no_question('Is the grasped object the [USB plug] <object> <frame> (818, 383), '
                                                   '(940, 560) </object> ?'), 'Is the grasped object the USB plug?')
        self.assertEqual(reference.noun_phrase('Black plastic part.'), 'the black plastic part')
        self.assertEqual(reference.noun_phrase('A pair of scissors'), 'a pair of scissors')
        self.assertIsNone(reference.noun_phrase('It is a long thin metallic object with two round handles'))
        self.assertIsNone(reference.noun_phrase('<grasp pose> (383, 430), (673, 519)'))
        self.assertEqual(reference.place_phrase('The black plug is now in the socket.'), 'in the socket')
        self.assertEqual(reference.place_phrase('on the blue surface'), 'on the blue surface')
        self.assertEqual(reference.place_phrase('Into the [white box].'), 'in the white box')
        self.assertIsNone(reference.place_phrase('The robot moved away'))

    def test_the_question_that_tells_done_from_not_yet_done_is_chosen_with_a_calibrated_threshold(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(reference, 'reference_frames', side_effect=frames):
            settings = config(directory)
            saved = reference.derive(settings, FakeModel(), progress=lambda _m: None)
            questions = reference.load_questions(settings)
        held, placed = saved['stages']['2']['chosen'], saved['stages']['3']['chosen']
        self.assertEqual(question_text(held['question']), 'Is a red cube held between the gripper fingers?')
        self.assertEqual(question_text(placed['question']), 'Is a red cube on the shelf?')
        self.assertTrue(held['separates'] and placed['separates'])
        self.assertEqual((held['specificity'], held['kept'], held['threshold']), (1., 1., .5))  # Youden 0.9, capped
        self.assertEqual(questions.subgoal(2), [(held['question'], ('yes', 'no'))])
        self.assertEqual(questions.for_alarm('release', 2), (3, [(placed['question'], ('yes', 'no'))]))
        self.assertEqual(saved['references'], ['/data/n1', '/data/n2', '/data/n3'])

    def test_a_question_that_hardly_separates_is_flagged_and_the_fixed_one_only_compared(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(reference, 'reference_frames', side_effect=frames):
            settings = config(directory, 'a metal surgical instrument', 'a blue surface')
            saved = reference.derive(settings, FakeModel(separating=False), progress=lambda _m: None)
            questions = reference.load_questions(settings)
        chosen = saved['stages']['2']['chosen']
        self.assertFalse(chosen['separates'])
        self.assertTrue(chosen['guarded'])  # Its threshold keeps 95% of the normal checks.
        self.assertEqual(questions.subgoal(2)[0][0], reference.held_question('a red cube'))
        self.assertEqual(questions.threshold(2), .5)  # No cut separates; capped at the yes/no point.
        fixed = saved['stages']['2']['fixed']
        self.assertIn('held between the two white fingers', fixed['question'])
        self.assertEqual((fixed['auroc'], fixed['kept'], fixed['specificity']), (.5, 1., 0.))

    def test_saved_questions_are_reused_only_for_the_same_reference_runs(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(reference, 'reference_frames', side_effect=frames):
            settings = config(directory)
            reference.derive(settings, FakeModel(), progress=lambda _m: None)
            self.assertIsNotNone(reference.load_questions(settings))
            settings['nominal_bags'] = ['/data/n1', '/data/n2']
            self.assertIsNone(reference.load_questions(settings))
            self.assertTrue(json.loads(Path(reference.questions_path(settings)).read_text())['stages'])

    def test_youden_threshold_balances_kept_normal_checks_and_stopped_not_done_frames(self):
        result = reference.separation([.3, .35, .4, .45], [.1, .2, .32, .5], cap=1.)
        self.assertEqual((result['threshold'], result['kept'], result['specificity']), (.3, 1., .5))
        self.assertEqual(reference.separation([.6, .7, .8], [.2, .3, .65])['threshold'], .5)  # never above 0.5

    def test_reselect_applies_the_current_rule_to_saved_scores_without_the_vlm(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(reference, 'reference_frames', side_effect=frames):
            settings = config(directory)
            saved = reference.derive(settings, FakeModel(), progress=lambda _m: None)
            path = Path(reference.questions_path(settings))
            old = json.loads(path.read_text())
            old['signature']['version'] = 'reference_questions_v3'
            old['stages']['2']['chosen'] = dict(old['stages']['2']['candidates'][1], threshold=.01)
            path.write_text(json.dumps(old))
            self.assertIsNone(reference.load_questions(settings))
            again = reference.reselect(settings, progress=lambda _m: None)
            self.assertEqual(again['stages']['2']['chosen']['question'], saved['stages']['2']['chosen']['question'])
            self.assertEqual(again['reselected_from'], 'reference_questions_v3')
            self.assertEqual(reference.load_questions(settings).threshold(2), .5)

    def test_the_task_sentence_guides_the_descriptions_and_the_vlm_writes_candidate_questions(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(reference, 'reference_frames', side_effect=frames):
            settings = config(directory)
            settings['rynnbrain']['vlm_check']['task_instruction'] = 'Pick up the red cube and put it on the shelf'
            model = TaskModel()
            saved = reference.derive(settings, model, progress=lambda _m: None)
            told = [p for p in model.prompts if "The robot's task: Pick up the red cube and put it on the shelf. " in p]
            self.assertEqual(len(told) * 2, len(model.prompts))  # Described once told the task, once not.
            written = [c['question'] for c in saved['stages']['2']['candidates'] if c['kind'].startswith('written')]
            self.assertEqual([question_text(q) for q in written], ['Is a red cube between the fingers?'])
            self.assertEqual({c['kind'].split()[-1] for c in saved['stages']['3']['candidates']}, {'(task)', '(plain)'})
            self.assertEqual((saved['signature']['instruction'], saved['object']),
                             ('Pick up the red cube and put it on the shelf.', 'a red cube'))
            # The check questions themselves do not state the task (a stated expectation gets repeated back).
            self.assertFalse(any('task' in c['question'] for st in saved['stages'].values() for c in st['candidates']))
            self.assertIsNotNone(reference.load_questions(settings))
            settings['rynnbrain']['vlm_check']['task_instruction'] = 'Something else.'
            self.assertIsNone(reference.load_questions(settings))  # A new task sentence derives again.

    def test_the_task_description_is_the_default_task_sentence(self):
        settings = config('/tmp')
        settings['rynnbrain']['task_description'] = 'Pick the scissor and place it in the basket'
        self.assertEqual(reference.task_instruction(settings), 'Pick the scissor and place it in the basket.')
        settings['rynnbrain']['vlm_check']['task_instruction'] = 'Insert the plug.'
        self.assertEqual(reference.task_instruction(settings), 'Insert the plug.')

    def test_written_questions_must_be_short_yes_no_questions(self):
        self.assertEqual(reference.yes_no_question('is the ethernet plug in the hole? yes'),
                         'Is the ethernet plug in the hole?')
        self.assertIsNone(reference.yes_no_question('What is held between the fingers?'))
        self.assertIsNone(reference.yes_no_question('Is the [Ethernet plug] <object> <area> (451, 683) in the [hole]'))

    def test_scores_are_cached_so_deriving_again_asks_nothing_new(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(reference, 'reference_frames', side_effect=frames):
            settings = config(directory)
            model = FakeModel()
            model.answer_probability = Mock(side_effect=FakeModel().answer_probability)
            first = reference.derive(settings, model, progress=lambda _m: None)
            asked = model.answer_probability.call_count
            again = reference.derive(settings, model, progress=lambda _m: None)
            self.assertGreater(asked, 0)
            self.assertEqual(model.answer_probability.call_count, asked)
            self.assertEqual(first['stages']['2']['chosen']['question'], again['stages']['2']['chosen']['question'])

    def test_a_derived_question_decides_at_its_own_threshold(self):
        checker = VlmChecker.__new__(VlmChecker)
        checker.topic, checker.item, checker.destination = 'gripper', 'a metal surgical instrument', 'a blue surface'
        checker.model = Mock(answer_probability=Mock(return_value=.3), max_image_size=64)
        frames_ = [(t / 10, ('image', Image.new('RGB', (8, 8)))) for t in range(0, 300, 2)]
        checker.questions = QuestionSet(checker.item, checker.destination)
        fixed, _ = checker.ask(frames_, {1: 0., 2: 10.}, 20.)
        checker.questions = QuestionSet(checker.item, checker.destination, source='reference', derived={
            2: dict(question=reference.held_question('the plug'), options=['yes', 'no'], threshold=.25)})
        derived, _ = checker.ask(frames_, {1: 0., 2: 10.}, 20.)
        self.assertEqual((fixed['verdict'], fixed['threshold'], fixed['question_source']), ('failure', .5, 'fixed'))
        self.assertEqual((derived['verdict'], derived['threshold'], derived['question_source']),
                         ('safe', .25, 'reference'))
        self.assertIn('Question (derived from the reference runs): Is the plug held between the gripper fingers?',
                      derived['answer'])


if __name__ == '__main__':
    unittest.main()
