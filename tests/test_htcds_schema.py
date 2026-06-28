from __future__ import annotations

from pathlib import Path
import unittest

from src.data_mapping import load_htcds_plus_schema, load_htcds_schema


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "HTCDS_standard" / "HTCDS Field Standards 2.0.xlsx"


class HTCDSSchemaTests(unittest.TestCase):
    def test_cleaned_workbook_loads_twenty_validated_fields(self) -> None:
        schema = load_htcds_schema(SCHEMA_PATH)

        self.assertEqual(len(schema), 20)
        self.assertTrue(schema.has_field("MethodsOfControl"))
        self.assertEqual(schema.get("AgeWhenTrafficked").field_type, "Number")

        methods = schema.get("MethodsOfControl").allowed_values
        self.assertIn("DebtBondage", methods)
        self.assertFalse(any(value != value.strip() for value in methods))
        self.assertTrue(
            schema.validate_value(
                "MethodsOfControl", ("FalsePromises", "WithholdsDocuments")
            )
        )
        self.assertFalse(schema.validate_value("MethodsOfControl", "NotAStandardValue"))

    def test_htcds_plus_registry_records_curated_fields_and_provenance(self) -> None:
        base_schema = load_htcds_schema(SCHEMA_PATH)
        schema = load_htcds_plus_schema(
            base_schema, ROOT / "HTCDS_standard" / "HTCDS+ Extensions.yaml"
        )

        abuse = schema.get("control.abuse")
        self.assertEqual(schema.schema_version, "0.1.0")
        self.assertEqual(abuse.field_type, "Binary")
        self.assertEqual(abuse.source_category, "curated_htcds_plus_group")
        self.assertIn("meansAbusePsyPhySex", abuse.source_fields)
        self.assertFalse(schema.has_field("CTDCControlAbuseComposite"))


if __name__ == "__main__":
    unittest.main()
