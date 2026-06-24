from __future__ import annotations

from pathlib import Path

import pandas as pd


REVIEW_QUEUE_COLUMNS = [
    "review_rank",
    "case_id",
    "route_label",
    "priority_score",
    "predicted_exploitation_type",
    "P(Sex)",
    "P(Labor)",
    "P(Both)",
    "confidence",
    "target_probability",
    "key_indicators",
    "selected_for_review",
    "reviewer_action",
    "reviewer_notes",
]


def build_review_queue_table(triage_output: pd.DataFrame) -> pd.DataFrame:
    queue = triage_output.copy()
    queue["reviewer_action"] = ""
    queue["reviewer_notes"] = ""
    columns = [col for col in REVIEW_QUEUE_COLUMNS if col in queue.columns]
    return queue[columns]


def export_review_queue(review_queue: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    review_queue.to_csv(path, index=False)
    return path

