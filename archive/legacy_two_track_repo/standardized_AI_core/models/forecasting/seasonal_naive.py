from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from standardized_AI_core.models.forecasting.base import (
    BaseForecastingModel,
    ForecastingConfig,
)


@dataclass
class SeasonalNaiveConfig(ForecastingConfig):
    seasonal_period: int = 7


class SeasonalNaiveForecaster(BaseForecastingModel):
    """
    Seasonal baseline forecaster that repeats the most recent seasonal cycle.
    """

    def __init__(self, config: Optional[SeasonalNaiveConfig] = None) -> None:
        super().__init__(config=config or SeasonalNaiveConfig())

    def _fit_state(self, hist_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        state: Dict[str, Dict[str, object]] = {}
        config = self.config

        for series_id, group in hist_df.groupby(config.id_col):
            ordered = group.sort_values(config.date_col).reset_index(drop=True)
            cycle = ordered[config.target_col].tail(config.seasonal_period).tolist()
            if not cycle:
                cycle = [0.0]
            state[series_id] = {"recent_cycle": cycle}
        return state

    def _forecast_one_series(
        self,
        series_id: str,
        future: pd.DataFrame,
    ) -> pd.DataFrame:
        cycle = list(self.series_state_[series_id]["recent_cycle"])
        cycle_length = len(cycle)

        output = future[[self.config.id_col, self.config.date_col]].copy()
        output["yhat"] = [
            round(float(cycle[index % cycle_length]), 4)
            for index in range(len(output))
        ]
        output["model_name"] = "seasonal_naive"
        output["forecast_explanation"] = (
            f"forecast repeats the most recent {cycle_length}-period seasonal pattern"
        )
        return output
