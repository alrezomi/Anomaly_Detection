# MULTI-TURN MODE PROMPTS
# The model sees nominal reference-bag frames first, then evaluates test frames
# against them in a second turn (visual memory - no saved description text).

def visual_evidence_prompt(task_description: str, input_mode: str) -> str:
    return f"""Compare these current observation frames with the nominal demonstration above.
            Task: {task_description}
            
            Describe the visible object motion, grasp, and final position relative to the target.
            Mention concrete matches or mismatches and say when the frames are insufficient to tell.
            Do not infer unseen events or give a success/failure decision.
            Return: Visual evidence: [one or two sentences describing what is visible]"""

def task_context_prompt_a(task_description: str, *, temporal: bool = False) -> str:
    """First turn: Keep the nominal reference clear and faithful to the user's intent."""
    return f"""You are a Failure Detector for Robotic Manipulation Tasks.

        Nominal reference task that robot is trying to perform: {task_description}

        This nominal demonstration represents the expected behavior.  You have the top view of the workspace, and you can see the progress across several frames.

        Your task is to decide whether the observed behavior matches the expected behavior or deviates from it.
        Focus on the object appearance, the motion sequence, the initial and final positions of the object the robot is manipulating.

        Observe the nominal demonstration closely. It is one example of the expected behavior:"""


def evaluation_prompt_multiturn_a(task_description: str, input_mode: str, *, temporal: bool = False) -> str:
    """Second turn: Compare the current observation against the nominal demonstration."""

    return f"""You previously saw the nominal demonstration for: {task_description}

        Now look at the current observation carefully and compare it with the nominal behavior you saw earlier

        Check whether the current scene follows the same pattern as the nominal reference:
        - Does the object start and finish in the locations as what the nominal reference expects exactly?
        - Does the object move through the same general sequence and direction as the nominal behavior?
        - Are there visible differences in the scene that suggest the current observation deviates from the nominal pattern?
        - The background and surrounding objects may differ, but the robot's manipulation of the target object should match the nominal reference.
        
        Use these cues to decide whether the until current observation, any failure happened during the performing of the task or not.

        Return exactly this format:
        Decision: success / failure / uncertain
        Visual evidence: [describe the observed match or mismatch]"""


def task_context_prompt_b(
    task_description: str, *, temporal: bool = False
) -> str:
    time_info = (
        "Each image includes its actual timestamp in seconds from "
        "this reference recording's start."
        if temporal else
        "The images are ordered observations; frame numbers are not seconds."
    )

    return f"""You are a Failure Detector for Robotic Manipulation Tasks.

        Nominal reference task that robot is trying to perform: {task_description}

        This nominal demonstration represents the expected behavior.  You have the top view of the workspace, and you can see the progress across several frames.

        Your task is to decide whether the observed behavior matches the expected behavior or deviates from it.
        Focus on the object appearance, the motion sequence, the initial and final positions of the object the robot is manipulating.

        Observe the nominal demonstration closely. It is one example of the expected behavior:

            """


def evaluation_prompt_multiturn_b(
    task_description: str, input_mode: str, *, temporal: bool = False
) -> str:
    time_info = (
        "Use the supplied timestamps to understand observation timing. "
        "Reference and test timestamps belong to separate recordings. "
    
        if temporal else
        "Use observation order to understand progress. "
        "Do not interpret frame numbers as elapsed seconds."
    )

    return f"""You previously saw the nominal demonstration for: {task_description}

        Now look at the current observation carefully and compare it with the nominal behavior you saw earlier.
        {time_info}

        Check whether the current scene follows the same pattern as the nominal reference:
        - Does the object start and finish in the locations as what the nominal reference expects exactly?
        - Does the object move through the same general sequence and direction as the nominal behavior?
        - Are there visible differences in the scene that suggest the current observation deviates from the nominal pattern?
        - The background and surrounding objects may differ, but the robot's manipulation of the target object should match the nominal reference.
        - You are visiting task on time, then the last frame is the current moment and may the task didn't finish yet, because the robot may still be in progress of performing the task.

        Use these cues to decide whether until the current observation any failure happened during performing desired task or not.
        Focus on camera frames that provided as an inputs and decide carefully.

        Return exactly this format:
        Decision: success / failure / uncertain
        Visual evidence:"""


def _prompt_pair(variant):
    if variant == 'A':
        return task_context_prompt_a, evaluation_prompt_multiturn_a
    if variant == 'B':
        return task_context_prompt_b, evaluation_prompt_multiturn_b
    raise ValueError('rynnbrain.prompt_variant must be A or B.')


def task_context_prompt(task_description: str, *, temporal: bool = False, variant='A') -> str:
    return _prompt_pair(variant)[0](task_description, temporal=temporal)


def evaluation_prompt_multiturn(task_description: str, input_mode: str, *, temporal: bool = False, variant='A') -> str:
    return _prompt_pair(variant)[1](task_description, input_mode, temporal=temporal)


# DINO -> VLM check. Recording protocol: stage 1 starts the task, stage 2 is
# marked once the robot has grasped the object, stage 3 once it has released it
# at the destination. The check asks the gripper camera whether the sub-goal of
# the current stage really holds. The questions only ask what is visible: a
# prompt stating what should be true was answered by repeating it back.
GRIPPER_CAMERA = ('This image comes from a camera mounted on a robot gripper. It looks down past the two white '
                  'gripper fingers at what is below the gripper.')


def subgoal_questions(stage: int, item: str, destination: str) -> list[tuple[str, tuple[str, ...]]] | None:
    """(question, answer options) for the sub-goal of `stage`; the first option means it holds.

    None before the grasp. item: the object as the gripper camera sees it, e.g.
    'a metal surgical instrument'. destination: the surface the object lies on
    at the destination, e.g. 'a blue surface'. After the release the object
    may lie next to the destination, with both surfaces in view, so that
    question asks which surface it lies on. A two-option form of the grasp
    question ("held" or "nothing held") leaned towards "nothing" and flagged
    nominal grasps, so the grasp is asked as yes/no.
    """
    item_start = item[:1].upper() + item[1:]
    noun = item.split(' ', 1)[1] if item.split(' ', 1)[0].lower() in ('a', 'an', 'the') else item
    if stage == 2:
        return [(f'{GRIPPER_CAMERA} Is {item} held between the two white fingers?\n'
                 'Answer with one word: yes or no.', ('yes', 'no'))]
    if stage == 3:
        return [(f'{GRIPPER_CAMERA} Which describes the image?\nA) {item_start} lies on {destination}.\n'
                 f'B) {item_start} lies on a surface of another color.\nC) No {noun} is visible.\n'
                 'Answer with one letter: A, B or C.', ('A', 'B', 'C'))]
    return None
