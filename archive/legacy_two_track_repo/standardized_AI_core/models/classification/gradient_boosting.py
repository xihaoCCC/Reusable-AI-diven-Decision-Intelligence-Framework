from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sklearn.ensemble import GradientBoostingClassifier

from standardized_AI_core.models.classification.base import (
    BaseVictimClassifier,
    VictimClassifierConfig,
)


@dataclass
class GradientBoostingConfig(VictimClassifierConfig):
    n_estimators: int = 150
    learning_rate: float = 0.05
    max_depth: int = 3
    random_state: int = 42


class GradientBoostingVictimClassifier(BaseVictimClassifier):
    """
    Strong structured-data classifier for early Track A experimentation.
    """

    def __init__(self, config: Optional[GradientBoostingConfig] = None) -> None:
        super().__init__(config=config or GradientBoostingConfig())

    def _build_estimator(self) -> GradientBoostingClassifier:
        config = self.config
        return GradientBoostingClassifier(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            random_state=config.random_state,
        )
