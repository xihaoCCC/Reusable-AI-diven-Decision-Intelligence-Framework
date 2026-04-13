from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import pandas as pd


@dataclass
class StandardRecord:
    """
    Minimal normalized record shape for cross-track processing.
    """

    record_id: str
    source_system: str
    event_timestamp: str | None = None
    narrative_text: str | None = None
    numeric_features: Dict[str, float] = field(default_factory=dict)
    categorical_features: Dict[str, str] = field(default_factory=dict)
    contextual_signals: Dict[str, float] = field(default_factory=dict)


@dataclass
class SchemaValidationResult:
    is_valid: bool
    missing_columns: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    optional_columns: Iterable[str] = (),
) -> SchemaValidationResult:
    required_set = list(required_columns)
    optional_set = set(optional_columns)

    missing = [column for column in required_set if column not in df.columns]
    allowed = set(required_set).union(optional_set)
    extra = [column for column in df.columns if column not in allowed]

    return SchemaValidationResult(
        is_valid=not missing,
        missing_columns=missing,
        extra_columns=extra,
    )
