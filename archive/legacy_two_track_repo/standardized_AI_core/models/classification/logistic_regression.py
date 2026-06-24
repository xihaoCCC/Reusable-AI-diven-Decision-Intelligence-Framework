from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sklearn.linear_model import LogisticRegression

from standardized_AI_core.models.classification.base import (
    BaseVictimClassifier,
    VictimClassifierConfig,
)


@dataclass
class LogisticRegressionConfig(VictimClassifierConfig):
    max_iter: int = 1000
    random_state: int = 42


class LogisticRegressionVictimClassifier(BaseVictimClassifier):
    """
    Interpretable baseline classifier for victim-identification support.
    """

    def __init__(self, config: Optional[LogisticRegressionConfig] = None) -> None:
        super().__init__(config=config or LogisticRegressionConfig())

    def _build_estimator(self) -> LogisticRegression:
        config = self.config
        return LogisticRegression(
            max_iter=config.max_iter,
            random_state=config.random_state,
        )
