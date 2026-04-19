from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standardized_AI_core.models.forecasting.base import (
    BaseForecastingModel,
    ForecastingConfig,
)

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only when torch is absent
    torch = None
    nn = None

_TorchModuleBase = nn.Module if nn is not None else object


@dataclass
class SimpleMLPForecastingConfig(ForecastingConfig):
    lookback: int = 14
    hidden_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.01
    min_training_windows: int = 12
    random_state: int = 42


class SimpleMLPForecaster(BaseForecastingModel):
    """
    Simple per-series neural-network forecaster.

    This intentionally lives in its own script because deep-learning models have
    heavier dependencies and training behavior than the other starter models.
    It uses a small MLP over recent lag windows and recursively forecasts future
    demand. It is a baseline, not a production deep-learning architecture.
    """

    def __init__(self, config: Optional[SimpleMLPForecastingConfig] = None) -> None:
        super().__init__(config=config or SimpleMLPForecastingConfig())

    def _fit_state(self, hist_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        _require_torch()
        _set_seed(self.config.random_state)

        state: Dict[str, Dict[str, object]] = {}
        config = self.config

        for series_id, group in hist_df.groupby(config.id_col):
            values = (
                group.sort_values(config.date_col)[config.target_col]
                .astype(float)
                .to_numpy()
            )
            fallback_value = float(values.mean()) if len(values) else 0.0
            recent_values = values.astype(float).tolist()

            X, y = _window_training_data(values, lookback=config.lookback)
            if len(X) < config.min_training_windows:
                state[series_id] = {
                    "model": None,
                    "scale": 1.0,
                    "recent_values": recent_values,
                    "fallback_value": fallback_value,
                }
                continue

            scale = max(float(np.mean(np.abs(values))), 1.0)
            X_tensor = torch.tensor(X / scale, dtype=torch.float32)
            y_tensor = torch.tensor((y / scale).reshape(-1, 1), dtype=torch.float32)

            model = _DemandMLP(
                lookback=config.lookback,
                hidden_size=config.hidden_size,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            loss_fn = nn.MSELoss()

            model.train()
            for _ in range(config.epochs):
                optimizer.zero_grad()
                predictions = model(X_tensor)
                loss = loss_fn(predictions, y_tensor)
                loss.backward()
                optimizer.step()

            state[series_id] = {
                "model": model.eval(),
                "scale": scale,
                "recent_values": recent_values,
                "fallback_value": fallback_value,
            }

        return state

    def _forecast_one_series(
        self,
        series_id: str,
        future: pd.DataFrame,
    ) -> pd.DataFrame:
        state = self.series_state_[series_id]
        model = state["model"]
        scale = float(state["scale"])
        history = list(state["recent_values"])
        fallback_value = float(state["fallback_value"])

        yhat: List[float] = []
        for _ in range(len(future)):
            if model is None or len(history) < self.config.lookback:
                prediction = fallback_value
            else:
                window = np.array(history[-self.config.lookback :], dtype=float) / scale
                with torch.no_grad():
                    tensor = torch.tensor(window.reshape(1, -1), dtype=torch.float32)
                    prediction = float(model(tensor).item() * scale)

            prediction = max(prediction, 0.0)
            yhat.append(round(prediction, 4))
            history.append(prediction)

        output = future[[self.config.id_col, self.config.date_col]].copy()
        output["yhat"] = yhat
        output["model_name"] = "simple_mlp"
        output["forecast_explanation"] = (
            f"forecast uses a simple per-series MLP over the last "
            f"{self.config.lookback} observations"
        )
        return output


class _DemandMLP(_TorchModuleBase):
    def __init__(self, lookback: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(lookback, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.network(x)


def _window_training_data(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X: List[Sequence[float]] = []
    y: List[float] = []
    for idx in range(lookback, len(values)):
        X.append(values[idx - lookback : idx])
        y.append(values[idx])
    return np.array(X, dtype=float), np.array(y, dtype=float)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "SimpleMLPForecaster requires PyTorch. Install it with `pip install torch`."
        )


def _set_seed(seed: int) -> None:
    if torch is not None:
        torch.manual_seed(seed)
