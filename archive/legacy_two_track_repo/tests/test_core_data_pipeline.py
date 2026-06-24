from __future__ import annotations

import unittest

import pandas as pd

from standardized_AI_core.data_pipeline import standardize_dataframe


class CoreDataPipelineTests(unittest.TestCase):
    def test_standardize_dataframe_maps_columns(self) -> None:
        raw = pd.DataFrame(
            {
                "case_identifier": ["A-1"],
                "system": ["demo"],
                "age": [17],
            }
        )

        standardized = standardize_dataframe(
            raw,
            schema_mapping={
                "case_identifier": "record_id",
                "system": "source_system",
            },
            required_columns=["record_id", "source_system"],
            optional_columns=["age"],
        )

        self.assertEqual(list(standardized.columns), ["record_id", "source_system", "age"])
        self.assertEqual(standardized.loc[0, "record_id"], "A-1")

    def test_standardize_dataframe_raises_for_missing_required_column(self) -> None:
        raw = pd.DataFrame({"record_id": ["A-1"]})

        with self.assertRaises(ValueError):
            standardize_dataframe(
                raw,
                schema_mapping={},
                required_columns=["record_id", "source_system"],
            )


if __name__ == "__main__":
    unittest.main()
