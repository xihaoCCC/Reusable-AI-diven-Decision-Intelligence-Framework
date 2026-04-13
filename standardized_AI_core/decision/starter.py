from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from standardized_AI_core.decision.base import BaseDecisionModule, rank_recommendations
from standardized_AI_core.explainability.reason_codes import build_reason_summary


@dataclass
class WeightedSignalPrioritizer(BaseDecisionModule):
    """
    Reusable starter prioritizer for early deployments.

    Each deployment can provide a column-to-weight mapping and threshold bands
    without rewriting the ranking pipeline itself.
    """

    id_col: str = "record_id"
    signal_weights: Dict[str, float] = field(default_factory=dict)
    high_priority_threshold: float = 75.0
    medium_priority_threshold: float = 50.0
    review_capacity: int = 50

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()

        if self.id_col not in working.columns:
            raise ValueError(f"Missing required id column: {self.id_col}")

        if not self.signal_weights:
            raise ValueError("signal_weights must contain at least one signal column")

        for column in self.signal_weights:
            if column not in working.columns:
                working[column] = 0.0
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)

        working["raw_score"] = 0.0
        for column, weight in self.signal_weights.items():
            working["raw_score"] += working[column] * weight

        working["score"] = _minmax_to_100(working["raw_score"])
        working["priority_band"] = working["score"].apply(self._priority_band)
        working["recommended_action"] = working["priority_band"].map(
            {
                "high": "Immediate review",
                "medium": "Queue for analyst review",
                "low": "Monitor or batch review",
            }
        )
        signal_columns: List[str] = list(self.signal_weights.keys())
        working["reason_summary"] = working.apply(
            lambda row: build_reason_summary(row, signal_columns=signal_columns),
            axis=1,
        )

        preferred_columns = [
            self.id_col,
            "score",
            "priority_band",
            "recommended_action",
            "reason_summary",
        ] + signal_columns

        ranked = rank_recommendations(
            working[preferred_columns + ["raw_score"]],
            score_col="score",
            review_capacity=self.review_capacity,
        )
        return ranked.drop(columns=["raw_score"])

    def _priority_band(self, score: float) -> str:
        if score >= self.high_priority_threshold:
            return "high"
        if score >= self.medium_priority_threshold:
            return "medium"
        return "low"


def _minmax_to_100(series: pd.Series) -> pd.Series:
    min_value = float(series.min())
    max_value = float(series.max())
    if min_value == max_value:
        return pd.Series(50.0, index=series.index)
    return ((series - min_value) / (max_value - min_value) * 100).round(2)
