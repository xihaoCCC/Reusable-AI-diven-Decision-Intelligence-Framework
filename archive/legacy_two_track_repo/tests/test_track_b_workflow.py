from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from Track_B_Supply_Chain_Decision_Intelligence.demand_forecasting_workflow import (
    TrackBWorkflowConfig,
    build_demand_forecast_suite,
    build_future_frame,
    summarize_forecast_suite,
)


ROOT = Path(__file__).resolve().parents[1]
TRACK_B_DATA = ROOT / "sample_data" / "track_b_synthetic_demand.csv"


class TrackBWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.history = pd.read_csv(TRACK_B_DATA)

    def test_build_future_frame_creates_horizon_for_each_series(self) -> None:
        config = TrackBWorkflowConfig(horizon=7)

        future = build_future_frame(self.history, config=config)

        self.assertEqual(len(future), 21)
        self.assertEqual(future.groupby("series_id").size().to_dict(), {"dept_a": 7, "dept_b": 7, "dept_c": 7})
        self.assertTrue({"is_holiday", "is_promo", "is_month_start", "is_month_end"}.issubset(future.columns))

    def test_demand_forecast_suite_and_summary_run(self) -> None:
        config = TrackBWorkflowConfig(horizon=7)

        forecasts = build_demand_forecast_suite(self.history, workflow_config=config)
        summary = summarize_forecast_suite(forecasts)

        expected_models = {"context_aware", "moving_average", "seasonal_naive"}
        self.assertEqual(set(forecasts["model_name"].unique()), expected_models)
        self.assertEqual(len(forecasts), 63)
        self.assertTrue({"model_name", "series_id", "average_yhat", "total_yhat"}.issubset(summary.columns))


if __name__ == "__main__":
    unittest.main()
