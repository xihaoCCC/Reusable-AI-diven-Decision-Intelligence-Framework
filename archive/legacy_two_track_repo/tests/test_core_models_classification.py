from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from standardized_AI_core.models import (
    GradientBoostingVictimClassifier,
    LogisticRegressionVictimClassifier,
    RandomForestVictimClassifier,
)


ROOT = Path(__file__).resolve().parents[1]
TRACK_A_DATA = ROOT / "sample_data" / "track_a_synthetic_cases.csv"


class CoreClassificationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = pd.read_csv(TRACK_A_DATA)

    def test_classification_models_fit_and_score_records(self) -> None:
        model_classes = [
            LogisticRegressionVictimClassifier,
            RandomForestVictimClassifier,
            GradientBoostingVictimClassifier,
        ]

        for model_class in model_classes:
            with self.subTest(model=model_class.__name__):
                model = model_class().fit(self.data)
                scored = model.score_records(self.data, id_col="case_id")

                self.assertEqual(len(scored), len(self.data))
                self.assertIn("case_id", scored.columns)
                self.assertIn("predicted_label", scored.columns)
                self.assertIn("victim_probability", scored.columns)
                self.assertTrue(scored["victim_probability"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
