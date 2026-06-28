from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml

from src.data_mapping.htcds_schema import HTCDSField, HTCDSSchema


@dataclass(frozen=True)
class HTCDSPlusField:
    name: str
    label: str
    field_type: str
    source_category: str
    source_datasets: tuple[str, ...]
    source_fields: tuple[str, ...]
    allowed_values: tuple[object, ...]
    description: str


class HTCDSPlusSchema:
    """Official HTCDS fields plus curated project-owned analytical extensions."""

    def __init__(
        self,
        base_schema: HTCDSSchema,
        extension_fields: Mapping[str, HTCDSPlusField],
        schema_version: str,
        source_path: Path,
        missing_value_policy: Mapping[str, object],
    ) -> None:
        self.base_schema = base_schema
        self._extensions = dict(extension_fields)
        self.schema_version = schema_version
        self.source_path = source_path
        self.missing_value_policy = dict(missing_value_policy)

    @property
    def official_field_names(self) -> tuple[str, ...]:
        return self.base_schema.field_names

    @property
    def extension_field_names(self) -> tuple[str, ...]:
        return tuple(self._extensions)

    @property
    def field_names(self) -> tuple[str, ...]:
        return (*self.official_field_names, *self.extension_field_names)

    def has_field(self, name: str) -> bool:
        return self.base_schema.has_field(name) or name in self._extensions

    def get(self, name: str) -> HTCDSField | HTCDSPlusField:
        if self.base_schema.has_field(name):
            return self.base_schema.get(name)
        try:
            return self._extensions[name]
        except KeyError as exc:
            raise KeyError(f"Unknown HTCDS+ field: {name}") from exc

    def validate_value(self, field_name: str, value: object) -> bool:
        if self.base_schema.has_field(field_name):
            return self.base_schema.validate_value(field_name, value)
        field = self.get(field_name)
        if _is_unknown(value) or not field.allowed_values:
            return True
        if any(str(item).startswith("ISO ") for item in field.allowed_values):
            return True
        values = value if isinstance(value, (list, tuple, set)) else [value]
        return all(item in field.allowed_values for item in values)


def load_htcds_plus_schema(
    base_schema: HTCDSSchema,
    extension_registry_path: str | Path,
) -> HTCDSPlusSchema:
    path = Path(extension_registry_path)
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)

    fields: dict[str, HTCDSPlusField] = {}
    for entry in payload.get("fields", []):
        name = str(entry["name"]).strip()
        if not name or name in fields or base_schema.has_field(name):
            raise ValueError(f"Duplicate or invalid HTCDS+ extension field: {name}")
        fields[name] = HTCDSPlusField(
            name=name,
            label=str(entry["label"]).strip(),
            field_type=str(entry["field_type"]).strip(),
            source_category=str(entry["source_category"]).strip(),
            source_datasets=tuple(entry.get("source_datasets", [])),
            source_fields=tuple(entry.get("source_fields", [])),
            allowed_values=tuple(entry.get("allowed_values", [])),
            description=str(entry.get("description", "")).strip(),
        )

    return HTCDSPlusSchema(
        base_schema=base_schema,
        extension_fields=fields,
        schema_version=str(payload["schema_version"]),
        source_path=path,
        missing_value_policy=payload.get("missing_value_policy", {}),
    )


def _is_unknown(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set)):
        return False
    return bool(pd.isna(value))
