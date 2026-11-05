"""TT-OPSD turn-local privileged information (arXiv:2610.05303, Appendix B.2).

z^TT_{i,t} = (x_i, w_{i,t}, o_{i,t+1}). Unlike ASCENT, there is no
full-trajectory success record and no invalid-action filtering. The text wrapper
below is an implementation choice; the paper specifies the tuple, not a template.
"""
from .ascent import episode_goal


def turn_records(episode):
    """One teacher-only record per turn, including invalid and unparsed turns."""
    if not episode.outcome.get("success"):
        raise ValueError("TT-OPSD distillation requires a verified successful episode")
    if not episode.steps:
        return []
    goal = episode_goal(episode)
    records = []
    for step in episode.steps:
        # feedback is what the agent actually received. On parse errors the
        # environment is not called and next_observation still holds the old state.
        text = (f"Task: {goal}\n"
                f"Current response:\n{step['raw_output']}\n"
                f"Returned observation:\n{step['feedback']}\n"
                "Condition on this response and its feedback when choosing the next action.\n\n")
        records.append({"turn": step["step"], "teacher_record": text})
    return records
