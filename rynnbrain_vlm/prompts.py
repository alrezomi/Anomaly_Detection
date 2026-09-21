# MULTI-TURN MODE PROMPTS
# The model sees nominal reference-bag frames first, then evaluates test frames
# against them in a second turn (visual memory - no saved description text).

def visual_evidence_prompt(task_description: str, input_mode: str) -> str:
    return f"""Compare these current observation frames with the nominal demonstration above.
Task: {task_description}
Input mode: {input_mode}. Heatmaps, if present, indicate differences, not proof of failure.
Describe the visible object motion, grasp, and final position relative to the target.
Mention concrete matches or mismatches and say when the frames are insufficient to tell.
Do not infer unseen events or give a success/failure decision.
Return: Visual evidence: [one or two sentences describing what is visible]"""

def task_context_prompt(task_description: str) -> str:
    """First turn: Keep the nominal reference clear and faithful to the user's intent."""
    return f"""You are a Failure Detector for Robotic Manipulation Tasks.

        Nominal reference task that robot is trying to perform: {task_description}

        This nominal demonstration represents the expected behavior.  You have the top view of the workspace, and you can see the progress across several frames.

        Your task is to decide whether the observed behavior matches the expected behavior or deviates from it.
        Focus on the object appearance, the motion sequence, the initial and final positions of the object the robot is manipulating.

        Observe the nominal demonstration closely. It is one example of the expected behavior:"""


def evaluation_prompt_multiturn(task_description: str, input_mode: str) -> str:
    """Second turn: Compare the current observation against the nominal demonstration."""
    if input_mode == "raw":
        heatmap_info = ""
        visual_focus = "raw camera frames"
    elif input_mode == "heatmap":
        heatmap_info = "\nNote: You are reviewing anomaly heatmaps that highlight visual differences from the nominal demonstration which we we are not sure if is it a failure in manipulation task or not. So don't judge only based on the heatmap alarms and differences; it just highlights the differences as a suggestion for where you can be suspicious more, and it wasn't clear to the model whether it's a failure or not. So you should decide according to what you see."
        visual_focus = "anomaly heatmap patterns"
    else:  # raw_heatmap
        heatmap_info = "\nNote: You are reviewing both raw frames and anomaly heatmaps which highlight visual differences from the nominal demonstration which we we are not sure if is it a failure in manipulation task or not. So don't judge only based on the heatmap alarms and differences; it just highlights the differences as a suggestion for where you can be suspicious more, and it wasn't clear to the model whether it's a failure or not. So you should decide according to what you see."
        visual_focus = "raw camera frames and anomaly heatmap patterns"

    return f"""You previously saw the nominal demonstration for: {task_description}

        Now look at the current observation carefully and compare it with the nominal behavior you saw earlier.{heatmap_info}

        Check whether the current scene follows the same pattern as the nominal reference:
        - Does the object start and finish in the locations as what the nominal reference expects exactly?
        - Does the object move through the same general sequence and direction as the nominal behavior?
        - Are there visible differences in the scene that suggest the current observation deviates from the nominal pattern?

        Use these cues to decide whether the current observation is there any failure during performing desired task or not.
        Focus on {visual_focus} that provided as an inputs and decide carefully.

        Return exactly this format:
        Decision: success / failure / uncertain
        Visual evidence: [describe the observed match or mismatch in {visual_focus}]"""
