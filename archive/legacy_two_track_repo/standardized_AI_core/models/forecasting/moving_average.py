from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from standardized_AI_core.models.forecasting.base import (
    BaseForecastingModel,
    ForecastingConfig,
)


@dataclass
class MovingAverageConfig(ForecastingConfig):
    window: int = 28


class MovingAverageForecaster(BaseForecastingModel):
    """
    Simple baseline demand forecaster using the recent moving average.
    """

    def __init__(self, config: Optional[MovingAverageConfig] = None) -> None:
        super().__init__(config=config or MovingAverageConfig())

    def _fit_state(self, hist_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        state: Dict[str, Dict[str, object]] = {}
        config = self.config
        for series_id, group in hist_df.groupby(config.id_col):
            recent = group[config.target_col].tail(config.window)
            state[series_id] = {
                "moving_average": float(recent.mean()) if not recent.empty else 0.0,
            }
        return state

    def _forecast_one_series(
        self,
        series_id: str,
        future: pd.DataFrame,
    ) -> pd.DataFrame:
        baseline = float(self.series_state_[series_id]["moving_average"])
        output = future[[self.config.id_col, self.config.date_col]].copy()
        output["yhat"] = round(baseline, 4)
        output["model_name"] = "moving_average"
        output["forecast_explanation"] = (
            f"forecast uses the recent {self.config.window}-period moving average"
        )
        return output
