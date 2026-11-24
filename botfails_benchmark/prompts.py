"""One question template for every BotFails task; tasks.json only fills in the task's objects.

Same rules as our gripper-camera check (rynnbrain_vlm/prompts.py): ask what is
visible, never state what should be true, answer with one word or letter, score
the first answer token. A third-person camera cannot tell which object was just
released, so the scene checks ask about the task's goal regions (the boxes, the
rack, the cup) and accept every state a correct execution passes through.

Check kinds in tasks.json:
  {"kind": "choice", "lead": ..., "A": ..., "B": ..., "C": ..., "holds": ["A", "C"]}
  {"kind": "yesno", "question": ..., "holds": ["no"]}
holds lists the answers that mean the sub-goal holds; the check's probability is
their total probability.
"""
from __future__ import annotations

CAMERA = 'This image comes from a camera that watches a red robot arm at work.'


def hold_question(held: str) -> dict:
    """Our stage-2 question for a third-person camera: is the grasped object held?"""
    return dict(name='held', text=f'{CAMERA} Is {held} held between the fingers of the red gripper?\n'
                                  'Answer with one word: yes or no.',
                options=('yes', 'no'), holds=('yes',))


def check_question(check: dict) -> dict:
    intro = check.get('intro', CAMERA)
    if check['kind'] == 'choice':
        text = (f"{intro} {check['lead']}\nA) {check['A']}.\nB) {check['B']}.\nC) {check['C']}.\n"
                'Answer with one letter: A, B or C.')
        options = ('A', 'B', 'C')
    elif check['kind'] == 'yesno':
        text = f"{intro} {check['question']}\nAnswer with one word: yes or no."
        options = ('yes', 'no')
    else:
        raise ValueError(f"Unknown check kind {check['kind']!r}.")
    holds = tuple(check['holds'])
    if not holds or not set(holds) < set(options):
        raise ValueError(f"holds must be a nonempty proper subset of {options}.")
    return dict(name=check['name'], text=text, options=options, holds=holds, crop=check.get('crop'))


def task_questions(task_config: dict, phase: str, trigger: str = 'dino') -> list[dict]:
    """Questions for a check in `phase` ('start', 'holding' or 'released'), caused by `trigger`.

    A check is asked in the phases it lists (default: after the first grasp, like our stage 1 rule)
    and for the triggers it lists: 'dino' (a DINO alarm, the default) and/or 'release' (the gripper
    has just let go of something, whether DINO alarms or not).
    """
    questions = []
    if trigger == 'dino' and phase == 'holding' and task_config.get('held'):
        questions.append(hold_question(task_config['held']))
    questions += [check_question(check) for check in task_config.get('checks', [])
                  if phase in check.get('phases', ('holding', 'released'))
                  and trigger in check.get('triggers', ('dino',))]
    if not questions and trigger == 'dino' and task_config.get('instruction'):
        # A missing phase-specific question must not silently bypass the VLM.
        # The new name deliberately avoids thresholds calibrated for other questions.
        questions.append(check_question(dict(
            name=f'task_context_{phase}', kind='yesno', holds=['no'],
            question=f'The task is: "{task_config["instruction"]}". '
                     'Allow normal work in progress; the task need not be finished yet. '
                     'Is a task-threatening error or an obstruction of the robot or its tool visible in this image?')))
    return questions


def holds_probability(distribution, options, holds) -> float:
    return float(sum(p for option, p in zip(options, distribution) if option in holds))
