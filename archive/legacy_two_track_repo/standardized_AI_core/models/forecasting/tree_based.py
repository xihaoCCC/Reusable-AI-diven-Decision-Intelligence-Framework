from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from standardized_AI_core.models.forecasting.base import (
    BaseForecastingModel,
    ForecastingConfig,
)


@dataclass
class TreeBasedDemandConfig(ForecastingConfig):
    """
    Configuration for a per-series tree-based demand forecaster.
    """

    lags: Sequence[int] = field(default_factory=lambda: (1, 7, 14))
    rolling_windows: Sequence[int] = field(default_factory=lambda: (7, 14))
    n_estimators: int = 200
    max_depth: int | None = 8
    random_state: int = 42
    min_training_rows: int = 12


class TreeBasedDemandForecaster(BaseForecastingModel):
    """
    Per-series Random Forest demand forecaster.

    This model builds simple lag, rolling-average, and calendar features for
    each demand series. It is still lightweight, but gives Track B a stronger
    nonlinear baseline than moving-average or seasonal-naive methods.
    """

    def __init__(self, config: Optional[TreeBasedDemandConfig] = None) -> None:
        super().__init__(config=config or TreeBasedDemandConfig())

    def _fit_state(self, hist_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        state: Dict[str, Dict[str, object]] = {}
        config = self.config

        for series_id, group in hist_df.groupby(config.id_col):
            ordered = group.sort_values(config.date_col).reset_index(drop=True)
            supervised = self._build_supervised_frame(ordered)
            recent_values = ordered[config.target_col].astype(float).tolist()

            if len(supervised) < config.min_training_rows:
                state[series_id] = {
                    "model": None,
                    "recent_values": recent_values,
                    "fallback_value": _safe_mean(recent_values),
                }
                continue

            feature_columns = self._feature_columns()
            model = RandomForestRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                random_state=config.random_state,
            )
            model.fit(supervised[feature_columns], supervised[config.target_col])
            state[series_id] = {
                "model": model,
                "recent_values": recent_values,
                "fallback_value": _safe_mean(recent_values),
            }

        return state

    def _forecast_one_series(
        self,
        series_id: str,
        future: pd.DataFrame,
    ) -> pd.DataFrame:
        state = self.series_state_[series_id]
        model = state["model"]
        history = list(state["recent_values"])
        fallback_value = float(state["fallback_value"])

        yhat: List[float] = []
        for _, row in future.sort_values(self.config.date_col).iterrows():
            if model is None:
                prediction = fallback_value
            else:
                features = self._feature_row(
                    ds=pd.Timestamp(row[self.config.date_col]),
                    history=history,
                )
                prediction = float(
                    model.predict(pd.DataFrame([features], columns=self._feature_columns()))[0]
                )

            prediction = max(prediction, 0.0)
            yhat.append(round(prediction, 4))
            history.append(prediction)

        output = future[[self.config.id_col, self.config.date_col]].copy()
        output["yhat"] = yhat
        output["model_name"] = "tree_based_random_forest"
        output["forecast_explanation"] = (
            "forecast uses lag, rolling-average, and calendar features in a "
            "per-series random forest model"
        )
        return output

    def _build_supervised_frame(self, ordered: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        values = ordered[self.config.target_col].astype(float).tolist()

        for idx, row in ordered.iterrows():
            history = values[:idx]
            if len(history) < max(self.config.lags):
                continue
            features = self._feature_row(
                ds=pd.Timestamp(row[self.config.date_col]),
                history=history,
            )
            features[self.config.target_col] = float(row[self.config.target_col])
            rows.append(features)

        return pd.DataFrame(rows)

    def _feature_row(self, ds: pd.Timestamp, history: Sequence[float]) -> Dict[str, float]:
        features: Dict[str, float] = {
            "day_of_week": float(ds.dayofweek),
            "day_of_month": float(ds.day),
            "month": float(ds.month),
            "is_month_start": float(ds.is_month_start),
            "is_month_end": float(ds.is_month_end),
        }

        for lag in self.config.lags:
            features[f"lag_{lag}"] = float(history[-lag]) if len(history) >= lag else 0.0

        for window in self.config.rolling_windows:
            window_values = history[-window:] if len(history) >= window else history
            features[f"rolling_mean_{window}"] = _safe_mean(window_values)

        return features

    def _feature_columns(self) -> List[str]:
        columns = [
            "day_of_week",
            "day_of_month",
            "month",
            "is_month_start",
            "is_month_end",
        ]
        columns.extend(f"lag_{lag}" for lag in self.config.lags)
        columns.extend(f"rolling_mean_{window}" for window in self.config.rolling_windows)
        return columns


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))
