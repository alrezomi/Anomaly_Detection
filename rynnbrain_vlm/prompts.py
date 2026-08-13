DESCRIPTION_PROMPT = """You are observing ordered multi-camera frames from a nominal robot manipulation attempt.
Describe only visible evidence. Identify the manipulated object carefully and compare the first, middle, and last frames. Focus on the initial state, robot action, object motion or state change, and final state. Do not decide success or failure.

Write a complete answer. Never return only a heading such as "Answer:".
Return these fields on separate lines:
Scene description: ...
Manipulated object: ...
Initial state: ...
Robot action: ...
Object motion or state change: ...
Final state: ...
Uncertain points: ..."""


def evaluation_prompt(nominal_description: str, input_mode: str) -> str:
    heatmap_note = (
        "Colored heatmap regions indicate locations highlighted by a separate visual anomaly detector. "
        "They are not thermal measurements and do not prove contact, grasping, or failure; use them only as supporting evidence."
        if "heatmap" in input_mode else ""
    )
    return f"""You are evaluating a robot manipulation attempt against a nominal reference.

Nominal reference:
{nominal_description}

{heatmap_note}
Compare object identity and appearance, robot action, object motion/state change, and final state. Gripper proximity alone is not proof of grasping: claim a pickup only when object displacement or lifting is visibly supported across frames. Base the decision only on visible evidence. If evidence is insufficient, choose uncertain.

Return exactly:
Decision: success / failure / uncertain
Failure reason: ...
Visual evidence: ...
Confidence: high / medium / low"""
