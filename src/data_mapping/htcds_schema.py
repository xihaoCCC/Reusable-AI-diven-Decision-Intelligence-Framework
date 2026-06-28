from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import pandas as pd


REQUIRED_SCHEMA_COLUMNS = (
    "Field Name",
    "Field Label",
    "Field Type",
    "Value Names (if picklist)",
    "Description",
)


@dataclass(frozen=True)
class HTCDSField:
    """One field from the cleaned HTCDS field-standard workbook."""

    name: str
    label: str
    field_type: str
    allowed_values: tuple[str, ...] = ()
    description: str = ""

    @property
    def is_picklist(self) -> bool:
        return "picklist" in self.field_type.lower()


class HTCDSSchema:
    """Validated, read-only collection of cleaned HTCDS field definitions."""

    def __init__(self, fields: Mapping[str, HTCDSField], source_path: Path) -> None:
        self._fields = dict(fields)
        self.source_path = source_path

    def __len__(self) -> int:
        return len(self._fields)

    def __iter__(self) -> Iterator[HTCDSField]:
        return iter(self._fields.values())

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self._fields)

    def has_field(self, name: str) -> bool:
        return name in self._fields

    def get(self, name: str) -> HTCDSField:
        try:
            return self._fields[name]
        except KeyError as exc:
            raise KeyError(f"Unknown HTCDS field: {name}") from exc

    def validate_value(self, field_name: str, value: object) -> bool:
        field = self.get(field_name)
        if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
            return True
        if not field.allowed_values:
            return True
        if any(value.startswith("Use ISO ") for value in field.allowed_values):
            return True

        values = value if isinstance(value, (list, tuple, set)) else [value]
        return all(str(item) in field.allowed_values for item in values)


def load_htcds_schema(
    workbook_path: str | Path,
    sheet_name: str = "Field_Standards",
) -> HTCDSSchema:
    """Load and validate the cleaned HTCDS Excel field standard."""

    path = Path(workbook_path)
    if not path.exists():
        raise FileNotFoundError(f"HTCDS schema workbook not found: {path}")

    frame = pd.read_excel(path, sheet_name=sheet_name)
    missing_columns = [
        column for column in REQUIRED_SCHEMA_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"HTCDS workbook is missing columns: {missing_columns}")

    fields: dict[str, HTCDSField] = {}
    for row_number, row in frame.iterrows():
        name = _clean_text(row["Field Name"])
        if not name:
            raise ValueError(f"HTCDS field name is empty at worksheet row {row_number + 2}")
        if name in fields:
            raise ValueError(f"Duplicate HTCDS field name: {name}")

        fields[name] = HTCDSField(
            name=name,
            label=_clean_text(row["Field Label"]),
            field_type=_clean_text(row["Field Type"]),
            allowed_values=_parse_allowed_values(row["Value Names (if picklist)"]),
            description=_clean_text(row["Description"]),
        )

    return HTCDSSchema(fields=fields, source_path=path)


def _parse_allowed_values(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(item.strip() for item in str(value).splitlines() if item.strip())


def _clean_text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()
