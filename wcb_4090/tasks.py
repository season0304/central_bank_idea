"""Explicit label mappings shared by preparation, training and prediction."""

LABELS_BY_TASK = {
    "time_label": {"not forward looking": 0, "forward looking": 1},
    "certain_label": {"certain": 0, "uncertain": 1},
    "stance_label": {"dovish": 0, "hawkish": 1, "neutral": 2, "irrelevant": 3},
}


def labels_for(task):
    if task not in LABELS_BY_TASK:
        raise ValueError(f"Unknown task: {task}")
    return LABELS_BY_TASK[task]
