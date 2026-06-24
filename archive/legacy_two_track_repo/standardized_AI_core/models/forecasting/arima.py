from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from standardized_AI_core.models.forecasting.base import (
    BaseForecastingModel,
    ForecastingConfig,
)

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:  # pragma: no cover - exercised only when statsmodels is absent
    ARIMA = None


@dataclass
class ARIMAConfig(ForecastingConfig):
    order: tuple[int, int, int] = (1, 1, 1)
    min_training_points: int = 10


class ARIMAForecaster(BaseForecastingModel):
    """
    Per-series ARIMA forecaster for classical time-series baselines.

    `statsmodels` is imported lazily/optionally so the broader model package can
    still import in environments where ARIMA support is not installed.
    """

    def __init__(self, config: Optional[ARIMAConfig] = None) -> None:
        super().__init__(config=config or ARIMAConfig())

    def _fit_state(self, hist_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        _require_statsmodels()
        state: Dict[str, Dict[str, object]] = {}
        config = self.config

        for series_id, group in hist_df.groupby(config.id_col):
            y = (
                group.sort_values(config.date_col)[config.target_col]
                .astype(float)
                .reset_index(drop=True)
            )
            fallback_value = float(y.mean()) if len(y) else 0.0

            if len(y) < config.min_training_points:
                state[series_id] = {
                    "fit_result": None,
                    "fallback_value": fallback_value,
                }
                continue

            try:
                fit_result = ARIMA(y, order=config.order).fit()
            except Exception:
                fit_result = None

            state[series_id] = {
                "fit_result": fit_result,
                "fallback_value": fallback_value,
            }

        return state

    def _forecast_one_series(
        self,
        series_id: str,
        future: pd.DataFrame,
    ) -> pd.DataFrame:
        state = self.series_state_[series_id]
        fit_result = state["fit_result"]
        steps = len(future)

        if fit_result is None:
            values = np.repeat(float(state["fallback_value"]), steps)
        else:
            values = fit_result.forecast(steps=steps)

        output = future[[self.config.id_col, self.config.date_col]].copy()
        output["yhat"] = [round(max(float(value), 0.0), 4) for value in values]
        output["model_name"] = "arima"
        output["forecast_explanation"] = (
            f"forecast uses per-series ARIMA order {self.config.order}"
        )
        return output


def _require_statsmodels() -> None:
    if ARIMA is None:
        raise ImportError(
            "ARIMAForecaster requires statsmodels. Install it with "
            "`pip install statsmodels`."
        )
