from __future__ import annotations

import unittest
import warnings
from pathlib import Path

import pandas as pd

from standardized_AI_core.models import (
    ARIMAForecaster,
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
    SimpleMLPForecaster,
    TreeBasedDemandForecaster,
)
from standardized_AI_core.models.forecasting.deep_learning_mlp import (
    SimpleMLPForecastingConfig,
)


ROOT = Path(__file__).resolve().parents[1]
TRACK_B_DATA = ROOT / "sample_data" / "track_b_synthetic_demand.csv"


class CoreForecastingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data = pd.read_csv(TRACK_B_DATA)
        cls.history = data.loc[data["series_id"] == "dept_a"].head(60).copy()
        cls.future = pd.DataFrame(
            {
                "series_id": ["dept_a"] * 7,
                "ds": pd.date_range("2026-04-01", periods=7, freq="D"),
            }
        )

    def assert_forecast_output(
        self,
        forecast: pd.DataFrame,
        expected_model_name: str,
        expected_rows: int | None = None,
    ) -> None:
        self.assertEqual(len(forecast), expected_rows or len(self.future))
        self.assertIn("yhat", forecast.columns)
        self.assertIn("model_name", forecast.columns)
        self.assertEqual(set(forecast["model_name"]), {expected_model_name})
        self.assertFalse(forecast["yhat"].isna().any())

    def test_baseline_and_advanced_forecasters_fit_and_forecast(self) -> None:
        model_cases = [
            (MovingAverageForecaster(), "moving_average"),
            (SeasonalNaiveForecaster(), "seasonal_naive"),
            (TreeBasedDemandForecaster(), "tree_based_random_forest"),
            (ARIMAForecaster(), "arima"),
        ]

        for model, expected_name in model_cases:
            with self.subTest(model=expected_name):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    forecast = model.fit(self.history).forecast(self.future)
                self.assert_forecast_output(forecast, expected_name)

    def test_simple_mlp_forecaster_runs_when_torch_is_available(self) -> None:
        model = SimpleMLPForecaster(
            SimpleMLPForecastingConfig(
                lookback=7,
                epochs=3,
                min_training_windows=5,
            )
        )

        try:
            forecast = model.fit(self.history).forecast(self.future.head(3))
        except ImportError as exc:
            self.skipTest(str(exc))

        self.assert_forecast_output(forecast, "simple_mlp", expected_rows=3)


if __name__ == "__main__":
    unittest.main()
