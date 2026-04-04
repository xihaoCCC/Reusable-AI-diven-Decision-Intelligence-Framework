from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class TrackBConfig:
    """
    Configuration for a lightweight Track B forecasting module.

    This starter version is designed for:
    - explainable baseline forecasting
    - event-aware forecast adjustment
    - public-safe release in an early-stage repo
    """
    id_col: str = "series_id"
    date_col: str = "ds"
    target_col: str = "y"
    horizon: int = 30
    freq: str = "D"

    rolling_window: int = 28
    seasonal_period: int = 7
    min_history: int = 35

    # Event columns can be binary indicators already present in future_df
    event_cols: Sequence[str] = (
        "is_holiday",
        "is_promo",
        "is_month_start",
        "is_month_end",
    )

    event_effect_clip_low: float = 0.70
    event_effect_clip_high: float = 1.30

    default_trend_drift: float = 0.0


class ContextAwareForecaster:
    """
    Initial Track B forecaster with:
    1. rolling baseline
    2. weekday seasonality
    3. learned event multipliers
    4. explainable component outputs

    This is intentionally a clean starter module, not a proprietary production model.
    """

    def __init__(self, config: Optional[TrackBConfig] = None) -> None:
        self.config = config or TrackBConfig()
        self.event_effects_: Dict[str, Dict[str, float]] = {}
        self.fitted_: bool = False

    def fit(self, hist_df: pd.DataFrame) -> "ContextAwareForecaster":
        self._validate_hist_input(hist_df)

        df = hist_df.copy()
        df = self._prepare_history(df)

        self.event_effects_ = {}
        for series_id, group in df.groupby(self.config.id_col):
            self.event_effects_[series_id] = self._estimate_event_effects(group)

        self.fitted_ = True
        return self

    def forecast(self, hist_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            self.fit(hist_df)

        self._validate_hist_input(hist_df)
        self._validate_future_input(future_df)

        hist = self._prepare_history(hist_df.copy())
        future = future_df.copy()
        future[self.config.date_col] = pd.to_datetime(future[self.config.date_col])

        outputs: List[pd.DataFrame] = []

        for series_id, future_group in future.groupby(self.config.id_col):
            series_hist = hist.loc[hist[self.config.id_col] == series_id].copy()
            if series_hist.empty:
                continue

            fcst = self._forecast_one_series(series_hist, future_group.copy())
            outputs.append(fcst)

        if not outputs:
            return pd.DataFrame()

        result = pd.concat(outputs, ignore_index=True)
        return result.sort_values([self.config.id_col, self.config.date_col]).reset_index(drop=True)

    def _forecast_one_series(
        self, hist: pd.DataFrame, future: pd.DataFrame
    ) -> pd.DataFrame:
        hist = hist.sort_values(self.config.date_col).reset_index(drop=True)
        future = future.sort_values(self.config.date_col).reset_index(drop=True)

        base_level = self._recent_level(hist)
        trend = self._recent_trend(hist)
        weekday_profile = self._weekday_profile(hist)
        series_id = hist[self.config.id_col].iloc[0]

        rows: List[Dict[str, object]] = []

        for step, (_, row) in enumerate(future.iterrows(), start=1):
            ds = pd.Timestamp(row[self.config.date_col])
            dow = ds.dayofweek

            seasonal_factor = weekday_profile.get(dow, 1.0)
            trend_factor = 1.0 + trend * step
            trend_factor = max(trend_factor, 0.05)

            baseline = base_level * seasonal_factor * trend_factor
            event_multiplier, event_drivers = self._event_multiplier(series_id, row)
            yhat = baseline * event_multiplier

            rows.append(
                {
                    self.config.id_col: series_id,
                    self.config.date_col: ds,
                    "yhat": round(float(yhat), 4),
                    "baseline_component": round(float(base_level), 4),
                    "seasonal_factor": round(float(seasonal_factor), 4),
                    "trend_factor": round(float(trend_factor), 4),
                    "event_multiplier": round(float(event_multiplier), 4),
                    "event_drivers": ", ".join(event_drivers) if event_drivers else "none",
                    "forecast_explanation": self._build_explanation(
                        seasonal_factor=seasonal_factor,
                        trend_factor=trend_factor,
                        event_multiplier=event_multiplier,
                        event_drivers=event_drivers,
                    ),
                }
            )

        return pd.DataFrame(rows)

    def _prepare_history(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.config.date_col] = pd.to_datetime(df[self.config.date_col])
        df[self.config.target_col] = pd.to_numeric(df[self.config.target_col], errors="coerce")
        df = df.dropna(subset=[self.config.id_col, self.config.date_col, self.config.target_col])
        df = df.sort_values([self.config.id_col, self.config.date_col]).reset_index(drop=True)
        df["dow"] = df[self.config.date_col].dt.dayofweek
        return df

    def _validate_hist_input(self, df: pd.DataFrame) -> None:
        required = [self.config.id_col, self.config.date_col, self.config.target_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required history columns: {missing}")

    def _validate_future_input(self, df: pd.DataFrame) -> None:
        required = [self.config.id_col, self.config.date_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required future columns: {missing}")

    def _recent_level(self, hist: pd.DataFrame) -> float:
        y = hist[self.config.target_col].tail(self.config.rolling_window)
        if y.empty:
            return 0.0
        return float(y.mean())

    def _recent_trend(self, hist: pd.DataFrame) -> float:
        y = hist[self.config.target_col].tail(self.config.rolling_window).reset_index(drop=True)
        if len(y) < 2:
            return self.config.default_trend_drift

        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, deg=1)
        slope = coeffs[0]

        mean_level = max(float(y.mean()), 1e-6)
        normalized_slope = float(slope / mean_level)
        return normalized_slope

    def _weekday_profile(self, hist: pd.DataFrame) -> Dict[int, float]:
        overall = hist[self.config.target_col].mean()
        if overall <= 0:
            return {i: 1.0 for i in range(7)}

        by_dow = hist.groupby("dow")[self.config.target_col].mean() / overall
        profile = {i: float(by_dow.get(i, 1.0)) for i in range(7)}
        return profile

    def _estimate_event_effects(self, hist: pd.DataFrame) -> Dict[str, float]:
        effects: Dict[str, float] = {}

        base_mean = hist[self.config.target_col].mean()
        if base_mean <= 0:
            return {c: 1.0 for c in self.config.event_cols}

        for col in self.config.event_cols:
            if col not in hist.columns:
                effects[col] = 1.0
                continue

            flagged = hist.loc[hist[col] == 1, self.config.target_col]
            unflagged = hist.loc[hist[col] == 0, self.config.target_col]

            if len(flagged) < 3 or len(unflagged) < 3:
                effects[col] = 1.0
                continue

            raw = float(flagged.mean() / max(unflagged.mean(), 1e-6))
            effects[col] = float(
                np.clip(
                    raw,
                    self.config.event_effect_clip_low,
                    self.config.event_effect_clip_high,
                )
            )

        return effects

    def _event_multiplier(self, series_id: str, row: pd.Series) -> tuple[float, List[str]]:
        multipliers = self.event_effects_.get(series_id, {})
        total = 1.0
        drivers: List[str] = []

        for col in self.config.event_cols:
            if col in row.index and int(row.get(col, 0) or 0) == 1:
                effect = multipliers.get(col, 1.0)
                total *= effect
                drivers.append(f"{col}({effect:.2f}x)")

        total = float(
            np.clip(total, self.config.event_effect_clip_low, self.config.event_effect_clip_high)
        )
        return total, drivers

    def _build_explanation(
        self,
        seasonal_factor: float,
        trend_factor: float,
        event_multiplier: float,
        event_drivers: Sequence[str],
    ) -> str:
        pieces: List[str] = []

        if seasonal_factor > 1.03:
            pieces.append("positive weekday seasonal pattern")
        elif seasonal_factor < 0.97:
            pieces.append("weaker weekday seasonal pattern")

        if trend_factor > 1.02:
            pieces.append("recent upward trend")
        elif trend_factor < 0.98:
            pieces.append("recent downward trend")

        if event_drivers:
            pieces.append(f"event effects: {', '.join(event_drivers)}")

        if not pieces:
            return "forecast driven primarily by recent baseline behavior"

        return "; ".join(pieces)


if __name__ == "__main__":
    # Example usage
    hist = pd.DataFrame(
        {
            "series_id": ["dept_a"] * 60,
            "ds": pd.date_range("2025-01-01", periods=60, freq="D"),
            "y": (
                100
                + np.sin(np.arange(60) / 7 * 2 * np.pi) * 10
                + np.linspace(0, 8, 60)
            ).round(2),
        }
    )
    hist["is_holiday"] = 0
    hist["is_promo"] = 0
    hist["is_month_start"] = (pd.to_datetime(hist["ds"]).dt.day <= 2).astype(int)
    hist["is_month_end"] = (pd.to_datetime(hist["ds"]).dt.day >= 29).astype(int)

    future = pd.DataFrame(
        {
            "series_id": ["dept_a"] * 14,
            "ds": pd.date_range("2025-03-02", periods=14, freq="D"),
        }
    )
    future["is_holiday"] = 0
    future["is_promo"] = 0
    future["is_month_start"] = (pd.to_datetime(future["ds"]).dt.day <= 2).astype(int)
    future["is_month_end"] = (pd.to_datetime(future["ds"]).dt.day >= 29).astype(int)

    model = ContextAwareForecaster()
    model.fit(hist)
    fcst = model.forecast(hist, future)
    print(fcst.to_string(index=False))