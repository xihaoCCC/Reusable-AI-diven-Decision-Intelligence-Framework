from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from src.data_mapping import (
    CTDCHTCDSPlusMapper,
    ModelCompatibilityReport,
    ModelFeatureConfig,
    build_ctdc_exploitation_model_frame,
    load_htcds_plus_schema,
    load_htcds_schema,
)


ROOT = Path(__file__).resolve().parents[1]


class CTDCHTCDSMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base_schema = load_htcds_schema(
            ROOT / "HTCDS_standard" / "HTCDS Field Standards 2.0.xlsx"
        )
        schema = load_htcds_plus_schema(
            base_schema, ROOT / "HTCDS_standard" / "HTCDS+ Extensions.yaml"
        )
        cls.mapper = CTDCHTCDSPlusMapper(schema)

    def test_coverage_distinguishes_standard_partial_and_extension_fields(self) -> None:
        report = self.mapper.coverage_report()

        self.assertEqual(
            report["mapping_status"].value_counts().to_dict(),
            {"covered": 15, "partial": 8, "extension": 4},
        )
        extensions = set(
            report.loc[report["mapping_status"] == "extension", "ctdc_field"]
        )
        self.assertEqual(
            extensions,
            {"yearOfRegistration", "ageBroad", "CountryOfExploitation", "traffickMonths"},
        )

    def test_records_map_to_standard_fields_without_expanding_composites(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "case_id": "case-1",
                    "gender": "Woman",
                    "ageBroad": "9--17",
                    "citizenship": "USA",
                    "yearOfRegistration": 2022,
                    "CountryOfExploitation": "USA",
                    "traffickMonths": "13--24 (1-2 yrs)",
                    "meansDebtBondageEarnings": 1,
                    "meansThreats": 1,
                    "meansAbusePsyPhySex": 1,
                    "meansFalsePromises": 1,
                    "meansDrugsAlcohol": 1,
                    "meansDenyBasicNeeds": 1,
                    "meansExcessiveWorkHours": 1,
                    "meansWithholdDocs": 1,
                    "isForcedLabour": 1,
                    "isSexualExploit": 1,
                    "sectorOfLabourConstruction": 1,
                    "sectorOfSexProstitution": 1,
                    "recruiterRelationFamily": 1,
                }
            ]
        )

        mapped = self.mapper.map_records(raw)
        row = mapped.iloc[0]

        self.assertEqual(row["Gender"], "Feminine")
        self.assertEqual(row["TypeOfExploitation"], "Both")
        self.assertEqual(
            row["MethodsOfControl"],
            (
                "FalsePromises",
                "PsychoactiveSubstance",
                "ExcessiveWorkingHours",
                "WithholdsDocuments",
            ),
        )
        self.assertNotIn("DebtBondage", row["MethodsOfControl"])
        self.assertEqual(row["control.debt_bondage_or_takes_earnings"], 1.0)
        self.assertEqual(row["control.abuse"], 1.0)
        self.assertEqual(row["RelationshipToTrafficker"], "Familial")
        self.assertEqual(row["ForcedLabourIndustry"], ("F_Construction",))
        self.assertEqual(row["TypeOfSexExploitation"], ("Prostitution",))

    def test_model_config_selects_subset_from_mapped_records(self) -> None:
        from src.utils import generate_ctdc_style_synthetic_records

        raw = generate_ctdc_style_synthetic_records(n_records=20, random_state=3)
        mapped = self.mapper.map_records(raw)
        feature_config = ModelFeatureConfig.from_yaml(
            ROOT / "configs" / "ctdc_exploitation_type_features.yaml"
        )
        model_frame = build_ctdc_exploitation_model_frame(mapped, feature_config)

        self.assertEqual(len(feature_config.model_features), 19)
        self.assertEqual(
            list(model_frame.columns),
            ["case_id", "exploitation_type", *feature_config.model_features],
        )
        self.assertNotIn("Citizenship", model_frame.columns)
        self.assertFalse(model_frame["exploitation_type"].isna().any())
        self.assertEqual(feature_config.core_feature_status, "candidate_pending_review")
        self.assertEqual(
            feature_config.assess_inference_compatibility(model_frame).status,
            "core_features_pending",
        )

    def test_unknown_binary_is_not_converted_to_explicit_negative(self) -> None:
        mapped = self.mapper.map_records(
            pd.DataFrame([{"case_id": "unknown-1", "gender": None}])
        )

        self.assertTrue(pd.isna(mapped.loc[0, "control.abuse"]))
        self.assertTrue(pd.isna(mapped.loc[0, "control.false_promises"]))
        self.assertTrue(pd.isna(mapped.loc[0, "recruiter.friend"]))
        self.assertTrue(pd.isna(mapped.loc[0, "Gender"]))

    def test_ctdc_age_band_is_normalized_to_htcds_plus_value(self) -> None:
        mapped = self.mapper.map_records(
            pd.DataFrame([{"case_id": "age-1", "ageBroad": "09--17"}])
        )

        self.assertEqual(
            mapped.loc[0, "person.age_at_identification_broad"], "9--17"
        )

    def test_defined_core_features_are_required_for_compatibility(self) -> None:
        config = ModelFeatureConfig(
            task_name="test",
            label_field="label",
            binary_features=("core", "optional"),
            core_feature_status="defined",
            core_features=("core",),
        )
        missing_core = config.assess_inference_compatibility(
            pd.DataFrame({"core": [float("nan")], "optional": [1.0]})
        )
        missing_optional = config.assess_inference_compatibility(
            pd.DataFrame({"core": [1.0], "optional": [float("nan")]})
        )

        self.assertIsInstance(missing_core, ModelCompatibilityReport)
        self.assertEqual(missing_core.status, "incompatible")
        self.assertEqual(missing_optional.status, "compatible_with_missingness")


if __name__ == "__main__":
    unittest.main()
