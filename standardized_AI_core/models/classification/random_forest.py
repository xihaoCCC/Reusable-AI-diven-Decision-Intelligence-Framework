from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sklearn.ensemble import RandomForestClassifier

from standardized_AI_core.models.classification.base import (
    BaseVictimClassifier,
    VictimClassifierConfig,
)


@dataclass
class RandomForestConfig(VictimClassifierConfig):
    n_estimators: int = 200
    max_depth: int | None = 8
    random_state: int = 42


class RandomForestVictimClassifier(BaseVictimClassifier):
    """
    Nonlinear classifier for capturing feature interactions in structured
    demographic and case-context data.
    """

    def __init__(self, config: Optional[RandomForestConfig] = None) -> None:
        super().__init__(config=config or RandomForestConfig())

    def _build_estimator(self) -> RandomForestClassifier:
        config = self.config
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
        )
