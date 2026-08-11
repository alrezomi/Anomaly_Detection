DESCRIPTION_PROMPT = """You are observing ordered frames from a robot manipulation attempt.
Describe only visible evidence. Identify the manipulated object carefully and compare the first, middle, and last frames. Focus on the initial state, robot action, object motion or state change, and final state. Do not decide success or failure.

Return these fields: Scene description; Manipulated object; Initial state; Robot action; Object motion or state change; Final state; Uncertain points."""


def evaluation_prompt(nominal_description: str, input_mode: str) -> str:
    heatmap_note = (
        "Colored heatmap regions indicate locations highlighted by a separate visual anomaly detector; use them as supporting evidence, not as proof of failure."
        if "heatmap" in input_mode else ""
    )
    return f"""You are evaluating a robot manipulation attempt against a nominal reference.

Nominal reference:
{nominal_description}

{heatmap_note}
Compare object identity and appearance, robot action, object motion/state change, and final state. Base the decision only on visible evidence. If evidence is insufficient, choose uncertain.

Return exactly:
Decision: success / failure / uncertain
Failure reason: ...
Visual evidence: ...
Confidence: high / medium / low"""
