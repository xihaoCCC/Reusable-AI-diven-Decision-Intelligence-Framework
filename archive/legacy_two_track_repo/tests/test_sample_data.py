from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = ROOT / "sample_data"


class SampleDataTests(unittest.TestCase):
    def test_track_a_sample_data_shape_and_label_balance(self) -> None:
        df = pd.read_csv(SAMPLE_DATA / "track_a_synthetic_cases.csv")

        expected_columns = {
            "case_id",
            "incident_text",
            "notes",
            "age",
            "prior_contacts",
            "has_missing_history",
            "location_risk_score",
            "household_instability_score",
            "gender",
            "race_ethnicity",
            "housing_status",
            "employment_status",
            "school_enrollment_status",
            "is_victim",
        }

        self.assertEqual(len(df), 200)
        self.assertTrue(expected_columns.issubset(df.columns))
        self.assertEqual(df["is_victim"].value_counts().sort_index().to_dict(), {0: 180, 1: 20})

    def test_track_b_sample_data_shape_and_scale(self) -> None:
        df = pd.read_csv(SAMPLE_DATA / "track_b_synthetic_demand.csv", parse_dates=["ds"])

        self.assertEqual(len(df), 270)
        self.assertEqual(df.groupby("series_id").size().to_dict(), {"dept_a": 90, "dept_b": 90, "dept_c": 90})
        self.assertEqual(str(df["ds"].min().date()), "2026-01-01")
        self.assertEqual(str(df["ds"].max().date()), "2026-03-31")

        ranges = df.groupby("series_id")["y"].agg(["min", "max"])
        self.assertGreaterEqual(ranges.loc["dept_a", "min"], 0)
        self.assertLessEqual(ranges.loc["dept_a", "max"], 1000)
        self.assertGreaterEqual(ranges.loc["dept_b", "min"], 5000)
        self.assertLessEqual(ranges.loc["dept_b", "max"], 50000)
        self.assertGreaterEqual(ranges.loc["dept_c", "min"], 750000)


if __name__ == "__main__":
    unittest.main()
