from __future__ import annotations

import unittest

import pandas as pd

from standardized_AI_core.decision import WeightedSignalPrioritizer


class CoreDecisionTests(unittest.TestCase):
    def test_weighted_signal_prioritizer_ranks_and_marks_capacity(self) -> None:
        df = pd.DataFrame(
            {
                "record_id": ["low", "high", "medium"],
                "risk_signal": [0.1, 0.9, 0.5],
                "context_signal": [0.0, 1.0, 0.2],
            }
        )
        prioritizer = WeightedSignalPrioritizer(
            signal_weights={"risk_signal": 1.0, "context_signal": 0.5},
            review_capacity=2,
        )

        ranked = prioritizer.run(df)

        self.assertEqual(ranked.loc[0, "record_id"], "high")
        self.assertIn("score", ranked.columns)
        self.assertIn("reason_summary", ranked.columns)
        self.assertEqual(int(ranked["within_review_capacity"].sum()), 2)

    def test_weighted_signal_prioritizer_requires_signals(self) -> None:
        prioritizer = WeightedSignalPrioritizer(signal_weights={})

        with self.assertRaises(ValueError):
            prioritizer.run(pd.DataFrame({"record_id": ["A-1"]}))


if __name__ == "__main__":
    unittest.main()
