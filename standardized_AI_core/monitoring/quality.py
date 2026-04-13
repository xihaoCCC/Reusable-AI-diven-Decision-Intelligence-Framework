from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


@dataclass
class DataQualityReport:
    row_count: int
    column_count: int
    missing_rate_by_column: Dict[str, float] = field(default_factory=dict)
    duplicate_row_count: int = 0


def summarize_dataframe_quality(df: pd.DataFrame) -> DataQualityReport:
    row_count = len(df)
    column_count = len(df.columns)
    missing_rate = {
        column: round(float(df[column].isna().mean()), 4) for column in df.columns
    }
    duplicate_row_count = int(df.duplicated().sum())

    return DataQualityReport(
        row_count=row_count,
        column_count=column_count,
        missing_rate_by_column=missing_rate,
        duplicate_row_count=duplicate_row_count,
    )
