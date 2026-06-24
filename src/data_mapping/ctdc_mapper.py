from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import pandas as pd


STANDARD_INDICATORS = [
    "threats",
    "abuse_indicators",
    "denial_basic_needs",
    "drugs_alcohol_control",
    "false_promises",
    "minor_young_indicator",
    "close_recruiter_relationship",
    "debt_bondage",
    "excessive_work_hours",
    "withheld_documents",
]


@dataclass
class DataMappingConfig:
    """Configuration for mapping agency-owned records into standard features."""

    id_col: str = "case_id"
    label_col: str = "exploitation_type"
    text_cols: Sequence[str] = ("intake_text", "case_notes", "referral_notes")
    local_to_standard: Mapping[str, str] = field(default_factory=dict)
    protected_fields: Sequence[str] = ("gender", "race_ethnicity", "nationality")


class CTDCMapper:
    """
    Lightweight CTDC-informed mapper.

    This module does not assume operational CTDC data access. It provides a
    reproducible mapping pattern for synthetic or local CTDC-style records.
    """

    def __init__(self, config: DataMappingConfig | None = None) -> None:
        self.config = config or DataMappingConfig()

    def map_records(self, records: pd.DataFrame) -> pd.DataFrame:
        working = records.rename(columns=dict(self.config.local_to_standard)).copy()
        self._ensure_required_columns(working)

        mapped = pd.DataFrame(index=working.index)
        mapped["case_id"] = working[self.config.id_col].astype(str)

        if self.config.label_col in working.columns:
            mapped["exploitation_type"] = working[self.config.label_col].map(
                self._normalize_label
            )

        text_cols = [col for col in self.config.text_cols if col in working.columns]
        mapped["source_text"] = (
            working[text_cols].fillna("").astype(str).agg(" | ".join, axis=1)
            if text_cols
            else ""
        )

        for indicator in STANDARD_INDICATORS:
            mapped[indicator] = self._indicator_series(working, indicator)

        mapped["control_score"] = mapped[
            [
                "threats",
                "drugs_alcohol_control",
                "debt_bondage",
                "withheld_documents",
            ]
        ].mean(axis=1)
        mapped["vulnerability_score"] = mapped[
            ["abuse_indicators", "denial_basic_needs", "minor_young_indicator"]
        ].mean(axis=1)
        mapped["relationship_score"] = mapped["close_recruiter_relationship"]
        mapped["data_quality_missing_indicator_count"] = mapped[
            STANDARD_INDICATORS
        ].isna().sum(axis=1)
        mapped["mapping_notes"] = "CTDC-informed synthetic/local feature mapping"

        return mapped

    def mapping_documentation(self) -> Dict[str, object]:
        return {
            "standard_indicators": STANDARD_INDICATORS,
            "protected_fields_restricted_by_default": list(
                self.config.protected_fields
            ),
            "label_classes": ["Sex", "Labor", "Both"],
        }

    def _ensure_required_columns(self, df: pd.DataFrame) -> None:
        if self.config.id_col not in df.columns:
            raise ValueError(f"Missing required id column: {self.config.id_col}")

    def _indicator_series(self, df: pd.DataFrame, indicator: str) -> pd.Series:
        if indicator in df.columns:
            return pd.to_numeric(df[indicator], errors="coerce").fillna(0).clip(0, 1)
        return pd.Series(0.0, index=df.index)

    def _normalize_label(self, value: object) -> str:
        text = str(value).strip().lower()
        if text in {"sex", "sexual", "sexual exploitation", "sex_only"}:
            return "Sex"
        if text in {"labor", "forced labor", "labour", "labor_only"}:
            return "Labor"
        if text in {"both", "mixed", "sex_and_labor", "sexual and labor"}:
            return "Both"
        return str(value)

