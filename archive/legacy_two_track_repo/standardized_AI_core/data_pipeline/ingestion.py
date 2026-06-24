from __future__ import annotations

from typing import Dict, Iterable, Sequence

import pandas as pd

from standardized_AI_core.schemas.standard_schema import validate_required_columns


def apply_schema_mapping(
    df: pd.DataFrame,
    schema_mapping: Dict[str, str],
) -> pd.DataFrame:
    """
    Rename local deployment columns into the framework's shared schema.
    """

    if not schema_mapping:
        return df.copy()
    return df.rename(columns=schema_mapping).copy()


def standardize_dataframe(
    df: pd.DataFrame,
    schema_mapping: Dict[str, str],
    required_columns: Sequence[str],
    optional_columns: Iterable[str] = (),
) -> pd.DataFrame:
    mapped = apply_schema_mapping(df, schema_mapping)
    validation = validate_required_columns(
        mapped,
        required_columns=required_columns,
        optional_columns=optional_columns,
    )
    if not validation.is_valid:
        raise ValueError(
            f"Input data is missing required columns: {validation.missing_columns}"
        )

    # The shared pipeline keeps required and optional fields in a predictable
    # order so downstream modules can rely on a stable interface.
    ordered_columns = list(required_columns) + [
        column for column in mapped.columns if column not in required_columns
    ]
    return mapped[ordered_columns]
