from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class DecisionResult:
    record_id: str
    score: float
    priority_band: str
    recommended_action: str
    reason_codes: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


class BaseDecisionModule(ABC):
    """
    Shared interface for turning model outputs into ranked recommendations.
    """

    @abstractmethod
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


def rank_recommendations(
    df: pd.DataFrame,
    score_col: str = "score",
    review_capacity: int | None = None,
) -> pd.DataFrame:
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True).copy()

    if review_capacity is not None:
        ranked["within_review_capacity"] = False
        ranked.loc[: review_capacity - 1, "within_review_capacity"] = True

    return ranked
