from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class ForecastingConfig:
    id_col: str = "series_id"
    date_col: str = "ds"
    target_col: str = "y"
    horizon: int = 30
    freq: str = "D"


class BaseForecastingModel(ABC):
    def __init__(self, config: Optional[ForecastingConfig] = None) -> None:
        self.config = config or ForecastingConfig()
        self.fitted_: bool = False
        self.series_state_: Dict[str, Dict[str, object]] = {}

    def fit(self, hist_df: pd.DataFrame) -> "BaseForecastingModel":
        self._validate_history_input(hist_df)
        prepared = self._prepare_history(hist_df)
        self.series_state_ = self._fit_state(prepared)
        self.fitted_ = True
        return self

    def forecast(self, future_df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ValueError("Model must be fitted before forecasting")

        self._validate_future_input(future_df)
        future = future_df.copy()
        future[self.config.date_col] = pd.to_datetime(future[self.config.date_col])

        outputs: List[pd.DataFrame] = []
        for series_id, future_group in future.groupby(self.config.id_col):
            if series_id not in self.series_state_:
                continue
            outputs.append(
                self._forecast_one_series(series_id=series_id, future=future_group.copy())
            )

        if not outputs:
            return pd.DataFrame()

        return pd.concat(outputs, ignore_index=True).sort_values(
            [self.config.id_col, self.config.date_col]
        ).reset_index(drop=True)

    def _prepare_history(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        prepared[self.config.date_col] = pd.to_datetime(prepared[self.config.date_col])
        prepared[self.config.target_col] = pd.to_numeric(
            prepared[self.config.target_col], errors="coerce"
        )
        prepared = prepared.dropna(
            subset=[self.config.id_col, self.config.date_col, self.config.target_col]
        )
        return prepared.sort_values(
            [self.config.id_col, self.config.date_col]
        ).reset_index(drop=True)

    def _validate_history_input(self, df: pd.DataFrame) -> None:
        required = [
            self.config.id_col,
            self.config.date_col,
            self.config.target_col,
        ]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required history columns: {missing}")

    def _validate_future_input(self, df: pd.DataFrame) -> None:
        required = [self.config.id_col, self.config.date_col]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required future columns: {missing}")

    @abstractmethod
    def _fit_state(self, hist_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def _forecast_one_series(
        self,
        series_id: str,
        future: pd.DataFrame,
    ) -> pd.DataFrame:
        raise NotImplementedError
