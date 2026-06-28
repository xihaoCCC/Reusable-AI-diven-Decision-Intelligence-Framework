from __future__ import annotations

from pathlib import Path
import unittest

from sklearn.model_selection import train_test_split

from src.ai_core import ExploitationTypeClassifier, ExploitationTypeClassifierConfig
from src.data_mapping import (
    CTDCHTCDSPlusMapper,
    ModelFeatureConfig,
    add_decision_support_features,
    build_ctdc_exploitation_model_frame,
    load_htcds_plus_schema,
    load_htcds_schema,
)
from src.decision_layer import apply_triage_scenario, load_scenario
from src.human_review import build_review_queue_table
from src.utils import generate_ctdc_style_synthetic_records


ROOT = Path(__file__).resolve().parents[1]


class CTDCPrototypeWorkflowTests(unittest.TestCase):
    def test_mapping_classifier_and_scenarios_run(self) -> None:
        records = generate_ctdc_style_synthetic_records(n_records=240, random_state=7)
        base_schema = load_htcds_schema(
            ROOT / "HTCDS_standard" / "HTCDS Field Standards 2.0.xlsx"
        )
        schema = load_htcds_plus_schema(
            base_schema, ROOT / "HTCDS_standard" / "HTCDS+ Extensions.yaml"
        )
        mapped = CTDCHTCDSPlusMapper(schema).map_records(records)
        feature_config = ModelFeatureConfig.from_yaml(
            ROOT / "configs" / "ctdc_exploitation_type_features.yaml"
        )
        model_frame = build_ctdc_exploitation_model_frame(mapped, feature_config)
        model_frame = add_decision_support_features(model_frame)
        train_df, local_df = train_test_split(
            model_frame,
            test_size=80,
            random_state=7,
            stratify=model_frame["exploitation_type"],
        )

        classifier = ExploitationTypeClassifier(
            ExploitationTypeClassifierConfig.from_feature_config(
                feature_config, model_type="logistic_regression"
            )
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
