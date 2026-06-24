from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd

from standardized_AI_core.models import (
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
)
from standardized_AI_core.models.forecasting.moving_average import MovingAverageConfig
from standardized_AI_core.models.forecasting.seasonal_naive import SeasonalNaiveConfig
from Track_B_Supply_Chain_Decision_Intelligence.context_aware_forecaster import (
    ContextAwareForecaster,
    TrackBConfig,
)


@dataclass
class TrackBWorkflowConfig:
    id_col: str = "series_id"
    date_col: str = "ds"
    target_col: str = "y"
    horizon: int = 30
    freq: str = "D"
    event_cols: Iterable[str] = field(
        default_factory=lambda: (
            "is_holiday",
            "is_promo",
            "is_month_start",
            "is_month_end",
        )
    )


def build_future_frame(
    hist_df: pd.DataFrame,
    config: Optional[TrackBWorkflowConfig] = None,
    event_defaults: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Create a future forecasting frame for every series in the history table.
    """

    config = config or TrackBWorkflowConfig()
    event_defaults = event_defaults or {}

    _validate_history(hist_df, config)
    hist = hist_df.copy()
    hist[config.date_col] = pd.to_datetime(hist[config.date_col])

    rows: List[Dict[str, object]] = []
    for series_id, group in hist.groupby(config.id_col):
        last_date = group[config.date_col].max()
        future_dates = pd.date_range(
            last_date + pd.tseries.frequencies.to_offset(config.freq),
            periods=config.horizon,
            freq=config.freq,
        )
        for ds in future_dates:
            row: Dict[str, object] = {
                config.id_col: series_id,
                config.date_col: ds,
            }
            for event_col in config.event_cols:
                row[event_col] = event_defaults.get(event_col, 0)
            rows.append(row)

    return pd.DataFrame(rows)


def run_baseline_forecasts(
    hist_df: pd.DataFrame,
    future_df: pd.DataFrame,
    workflow_config: Optional[TrackBWorkflowConfig] = None,
) -> pd.DataFrame:
    """
    Run shared-core baseline forecasting models for demand planning.
    """

    workflow_config = workflow_config or TrackBWorkflowConfig()
    outputs = []
    models = (
        MovingAverageForecaster(
            MovingAverageConfig(
                id_col=workflow_config.id_col,
                date_col=workflow_config.date_col,
                target_col=workflow_config.target_col,
                horizon=workflow_config.horizon,
                freq=workflow_config.freq,
            )
        ),
        SeasonalNaiveForecaster(
            SeasonalNaiveConfig(
                id_col=workflow_config.id_col,
                date_col=workflow_config.date_col,
                target_col=workflow_config.target_col,
                horizon=workflow_config.horizon,
                freq=workflow_config.freq,
            )
        ),
    )
    for model in models:
        outputs.append(model.fit(hist_df).forecast(future_df))

    return pd.concat(outputs, ignore_index=True).sort_values(
        ["model_name", workflow_config.id_col, workflow_config.date_col]
    ).reset_index(drop=True)


def run_context_aware_forecast(
    hist_df: pd.DataFrame,
    future_df: pd.DataFrame,
    config: Optional[TrackBConfig] = None,
) -> pd.DataFrame:
    """
    Run the Track B context-aware forecaster and label the output consistently
    with shared-core baseline outputs.
    """

    model = ContextAwareForecaster(config=config)
    forecast = model.fit(hist_df).forecast(hist_df, future_df)
    if forecast.empty:
        return forecast
    forecast["model_name"] = "context_aware"
    return forecast


def build_demand_forecast_suite(
    hist_df: pd.DataFrame,
    future_df: Optional[pd.DataFrame] = None,
    workflow_config: Optional[TrackBWorkflowConfig] = None,
) -> pd.DataFrame:
    """
    Build a starter forecast suite containing baseline and context-aware model
    outputs for the same future horizon.
    """

    workflow_config = workflow_config or TrackBWorkflowConfig()
    future = future_df if future_df is not None else build_future_frame(
        hist_df,
        config=workflow_config,
    )

    baseline = run_baseline_forecasts(hist_df, future, workflow_config=workflow_config)
    context = run_context_aware_forecast(
        hist_df,
        future,
        config=TrackBConfig(
            id_col=workflow_config.id_col,
            date_col=workflow_config.date_col,
            target_col=workflow_config.target_col,
            horizon=workflow_config.horizon,
            freq=workflow_config.freq,
            event_cols=tuple(workflow_config.event_cols),
        ),
    )

    return pd.concat([baseline, context], ignore_index=True).sort_values(
        ["model_name", workflow_config.id_col, workflow_config.date_col]
    ).reset_index(drop=True)


def summarize_forecast_suite(
    forecasts: pd.DataFrame,
    id_col: str = "series_id",
    date_col: str = "ds",
) -> pd.DataFrame:
    """
    Summarize forecast volume by model and series for quick planning review.
    """

    required = {id_col, date_col, "model_name", "yhat"}
    missing = [column for column in required if column not in forecasts.columns]
    if missing:
        raise ValueError(f"Missing required forecast columns: {missing}")

    return (
        forecasts.groupby(["model_name", id_col])
        .agg(
            forecast_start=(date_col, "min"),
            forecast_end=(date_col, "max"),
            average_yhat=("yhat", "mean"),
            total_yhat=("yhat", "sum"),
        )
        .round({"average_yhat": 4, "total_yhat": 4})
        .reset_index()
    )


def _validate_history(df: pd.DataFrame, config: TrackBWorkflowConfig) -> None:
    required = [config.id_col, config.date_col, config.target_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required history columns: {missing}")


if __name__ == "__main__":
    hist = pd.DataFrame(
        {
            "series_id": ["dept_a"] * 35 + ["dept_b"] * 35,
            "ds": list(pd.date_range("2026-01-01", periods=35, freq="D")) * 2,
            "y": [100 + (i % 7) * 3 for i in range(35)]
            + [80 + (i % 7) * 2 for i in range(35)],
        }
    )
    hist["is_holiday"] = 0
    hist["is_promo"] = 0
    hist["is_month_start"] = (pd.to_datetime(hist["ds"]).dt.day <= 2).astype(int)
    hist["is_month_end"] = (pd.to_datetime(hist["ds"]).dt.day >= 29).astype(int)

    config = TrackBWorkflowConfig(horizon=7)
    suite = build_demand_forecast_suite(hist, workflow_config=config)
    print(suite.head(12).to_string(index=False))
    print(summarize_forecast_suite(suite).to_string(index=False))
