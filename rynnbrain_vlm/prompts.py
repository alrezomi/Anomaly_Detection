# MULTI-TURN MODE PROMPTS
# The model sees nominal reference-bag frames first, then evaluates test frames
# against them in a second turn (visual memory - no saved description text).

def task_context_prompt(task_description: str) -> str:
    """First turn: Establish task context and show nominal demonstration."""
    return f"""You are evaluating whether there is any anomaly in a robot manipulation attempt.

        Nominal reference task that robot is trying to perform: {task_description}

        You will see a nominal (correct/successful) demonstration first.
        Study the object state, gripper motion, and action sequence carefully.
        You will then evaluate if a test case matches this demonstration.

        Observe this nominal demonstration closely:"""


def evaluation_prompt_multiturn(task_description: str, input_mode: str) -> str:
    """Second turn: Evaluate test case against observed nominal demonstration.

    The model has already seen the nominal frames, so we just ask it to compare.
    """
    if input_mode == "raw":
        heatmap_info = ""
        visual_focus = "raw camera frames"
    elif input_mode == "heatmap":
        heatmap_info = "\nNote: You will see anomaly detection heatmaps (based on approximate visual differences between the nominal demonstration and the current observation, which is our test case)."
        visual_focus = "anomaly heatmap patterns"
    else:  # raw_heatmap
        heatmap_info = "\nNote: You will see both raw frames and anomaly heatmaps for cross-reference."
        visual_focus = "raw frames and anomaly patterns"

    return f"""You observed the nominal demonstration for: {task_description}

        Now evaluate this test case against the nominal demonstration.{heatmap_info}

        Compare:
        - Object identity and state (same object? same position/orientation?)
        - Gripper action (same motion and trajectory?)
        - Object motion and displacement (same movement?)
        - Final state (same end result?)

        According to the nominal demonstration, evaluate if the attempt is successful or if there is an anomaly. Focus on {visual_focus} for your decision.

        Return exactly:
        Decision: success / failure / uncertain
        Failure reason: [explain what is different, if anything]
        Visual evidence: [describe what you observed in {visual_focus}]
        Confidence: high / medium / low"""
