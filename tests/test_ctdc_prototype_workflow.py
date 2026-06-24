from __future__ import annotations

from pathlib import Path
import unittest

from sklearn.model_selection import train_test_split

from src.ai_core import ExploitationTypeClassifier, ExploitationTypeClassifierConfig
from src.data_mapping import CTDCMapper
from src.decision_layer import apply_triage_scenario, load_scenario
from src.human_review import build_review_queue_table
from src.utils import generate_ctdc_style_synthetic_records


ROOT = Path(__file__).resolve().parents[1]


class CTDCPrototypeWorkflowTests(unittest.TestCase):
    def test_mapping_classifier_and_scenarios_run(self) -> None:
        records = generate_ctdc_style_synthetic_records(n_records=240, random_state=7)
        mapped = CTDCMapper().map_records(records)
        train_df, local_df = train_test_split(
            mapped,
            test_size=80,
            random_state=7,
            stratify=mapped["exploitation_type"],
        )

        classifier = ExploitationTypeClassifier(
            ExploitationTypeClassifierConfig(model_type="logistic_regression")
        ).fit(train_df)
        scored = classifier.score_records(local_df)

        expected_probability_columns = {"P(Sex)", "P(Labor)", "P(Both)", "confidence"}
        self.assertTrue(expected_probability_columns.issubset(scored.columns))
        self.assertEqual(len(scored), 80)

        for scenario_file in [
            "scenario_small_ngo_multidisciplinary.yaml",
            "scenario_labor_task_force.yaml",
        ]:
            scenario = load_scenario(ROOT / "configs" / scenario_file)
            triage_output = apply_triage_scenario(scored, scenario)
            review_queue = build_review_queue_table(triage_output)

            self.assertEqual(len(review_queue), 80)
            self.assertIn("priority_score", review_queue.columns)
            self.assertIn("reviewer_action", review_queue.columns)
            self.assertEqual(int(review_queue["selected_for_review"].sum()), 30)


if __name__ == "__main__":
    unittest.main()
