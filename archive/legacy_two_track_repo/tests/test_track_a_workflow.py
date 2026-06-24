from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from Track_A_Public_Safety_Decision_Intelligence.victim_identification_workflow import (
    build_track_a_review_queue,
    summarize_review_queue,
    train_victim_identification_model,
)


ROOT = Path(__file__).resolve().parents[1]
TRACK_A_DATA = ROOT / "sample_data" / "track_a_synthetic_cases.csv"


class TrackAWorkflowTests(unittest.TestCase):
    def test_track_a_review_queue_runs_with_sample_data(self) -> None:
        cases = pd.read_csv(TRACK_A_DATA)
        model = train_victim_identification_model(cases, model_name="logistic_regression")
        queue = build_track_a_review_queue(cases, trained_model=model)

        expected_columns = {
            "case_id",
            "risk_score",
            "victim_probability",
            "combined_priority_score",
            "priority_band",
            "recommended_action",
            "top_signals",
        }

        self.assertEqual(len(queue), len(cases))
        self.assertTrue(expected_columns.issubset(queue.columns))
        self.assertFalse(queue["combined_priority_score"].isna().any())

        summary = summarize_review_queue(queue)
        self.assertEqual(summary["case_count"], len(cases))
        self.assertIn("priority_band_counts", summary)


if __name__ == "__main__":
    unittest.main()
