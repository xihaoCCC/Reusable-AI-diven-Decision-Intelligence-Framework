from __future__ import annotations

from typing import Iterable, List

import pandas as pd


def build_reason_codes(
    row: pd.Series,
    signal_columns: Iterable[str],
    min_signal_strength: float = 0.0,
) -> List[str]:
    reason_codes: List[str] = []

    for column in signal_columns:
        value = row.get(column)
        if isinstance(value, (int, float)) and value > min_signal_strength:
            reason_codes.append(f"{column}:{value}")
        elif isinstance(value, str) and value.strip():
            reason_codes.append(f"{column}:{value.strip()}")

    return reason_codes


def build_reason_summary(
    row: pd.Series,
    signal_columns: Iterable[str],
    max_reasons: int = 4,
) -> str:
    reasons = build_reason_codes(row, signal_columns=signal_columns)
    if not reasons:
        return "no explicit reason codes available yet"
    return " | ".join(reasons[:max_reasons])
