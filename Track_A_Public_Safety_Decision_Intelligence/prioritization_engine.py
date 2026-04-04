from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd


@dataclass
class TrackAConfig:
    """
    Configuration for a lightweight, configurable Track A prioritization engine.

    This starter version is intentionally generic and public-safe:
    - no sensitive business rules
    - no private model weights
    - easy to extend into a richer ML / rules hybrid pipeline
    """
    id_col: str = "case_id"
    text_cols: Sequence[str] = ("incident_text", "notes")
    age_col: Optional[str] = "age"
    prior_contacts_col: Optional[str] = "prior_contacts"
    missing_history_col: Optional[str] = "has_missing_history"
    location_risk_col: Optional[str] = "location_risk_score"
    review_capacity: int = 50

    # Domain-informed but generic starter keyword sets
    high_signal_terms: Sequence[str] = (
        "trafficking",
        "exploitation",
        "coercion",
        "minor",
        "runaway",
        "hotel",
        "tattoo",
        "older boyfriend",
        "commercial sex",
        "escort",
        "branding",
        "control",
    )
    medium_signal_terms: Sequence[str] = (
        "recruit",
        "transport",
        "cash",
        "fearful",
        "multiple phones",
        "unhoused",
        "substance use",
        "withdrawn",
    )

    # Simple public starter weights
    weight_high_terms: float = 3.0
    weight_medium_terms: float = 1.5
    weight_minor: float = 4.0
    weight_missing_history: float = 2.5
    weight_prior_contacts: float = 0.8
    weight_location_risk: float = 1.0

    high_priority_threshold: float = 75.0
    medium_priority_threshold: float = 50.0

    def validate(self) -> None:
        if self.review_capacity <= 0:
            raise ValueError("review_capacity must be positive")
        if self.high_priority_threshold <= self.medium_priority_threshold:
            raise ValueError(
                "high_priority_threshold must be greater than medium_priority_threshold"
            )


class TrackAPrioritizationEngine:
    """
    Initial Track A prioritization engine.

    Responsibilities:
    1. standardize input text
    2. extract simple risk signals
    3. compute a configurable risk score
    4. produce review-ready ranked outputs
    """

    def __init__(self, config: Optional[TrackAConfig] = None) -> None:
        self.config = config or TrackAConfig()
        self.config.validate()

    def prioritize(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(df)

        working = df.copy()
        working["_combined_text"] = self._combine_text_fields(working)

        high_hits, high_terms = self._count_keyword_hits(
            working["_combined_text"], self.config.high_signal_terms
        )
        med_hits, med_terms = self._count_keyword_hits(
            working["_combined_text"], self.config.medium_signal_terms
        )

        working["high_signal_hit_count"] = high_hits
        working["medium_signal_hit_count"] = med_hits
        working["matched_high_signal_terms"] = high_terms
        working["matched_medium_signal_terms"] = med_terms

        working["minor_flag"] = self._build_minor_flag(working)
        working["missing_history_flag"] = self._safe_numeric_flag(
            working, self.config.missing_history_col
        )
        working["prior_contacts_score"] = self._scaled_numeric(
            working, self.config.prior_contacts_col
        )
        working["location_risk_score_norm"] = self._scaled_numeric(
            working, self.config.location_risk_col
        )

        working["raw_score"] = (
            working["high_signal_hit_count"] * self.config.weight_high_terms
            + working["medium_signal_hit_count"] * self.config.weight_medium_terms
            + working["minor_flag"] * self.config.weight_minor
            + working["missing_history_flag"] * self.config.weight_missing_history
            + working["prior_contacts_score"] * self.config.weight_prior_contacts
            + working["location_risk_score_norm"] * self.config.weight_location_risk
        )

        working["risk_score"] = self._minmax_to_100(working["raw_score"])
        working["priority_band"] = working["risk_score"].apply(self._priority_band)
        working["recommended_action"] = working.apply(
            self._recommended_action, axis=1
        )
        working["top_signals"] = working.apply(self._build_top_signals, axis=1)

        working = working.sort_values(
            by=["risk_score", "high_signal_hit_count", "medium_signal_hit_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        working["within_review_capacity"] = False
        working.loc[: self.config.review_capacity - 1, "within_review_capacity"] = True

        ordered_cols = self._ordered_output_columns(working)
        return working[ordered_cols]

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = [c for c in [self.config.id_col, *self.config.text_cols] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _combine_text_fields(self, df: pd.DataFrame) -> pd.Series:
        return (
            df[list(self.config.text_cols)]
            .fillna("")
            .astype(str)
            .agg(" | ".join, axis=1)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    def _count_keyword_hits(
        self, text_series: pd.Series, keywords: Sequence[str]
    ) -> Tuple[pd.Series, pd.Series]:
        patterns = [re.compile(rf"\b{re.escape(term.lower())}\b") for term in keywords]

        hit_counts: List[int] = []
        matched_terms: List[List[str]] = []

        for text in text_series.fillna(""):
            matched: List[str] = []
            for term, pattern in zip(keywords, patterns):
                if pattern.search(text):
                    matched.append(term)
            hit_counts.append(len(matched))
            matched_terms.append(matched)

        return pd.Series(hit_counts, index=text_series.index), pd.Series(
            matched_terms, index=text_series.index
        )

    def _build_minor_flag(self, df: pd.DataFrame) -> pd.Series:
        if self.config.age_col is None or self.config.age_col not in df.columns:
            return pd.Series(0.0, index=df.index)
        age = pd.to_numeric(df[self.config.age_col], errors="coerce")
        return ((age >= 0) & (age < 18)).astype(float)

    def _safe_numeric_flag(self, df: pd.DataFrame, col: Optional[str]) -> pd.Series:
        if col is None or col not in df.columns:
            return pd.Series(0.0, index=df.index)
        values = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return (values > 0).astype(float)

    def _scaled_numeric(self, df: pd.DataFrame, col: Optional[str]) -> pd.Series:
        if col is None or col not in df.columns:
            return pd.Series(0.0, index=df.index)

        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == 0:
            return pd.Series(0.0, index=df.index)

        min_val = values.min(skipna=True)
        max_val = values.max(skipna=True)

        if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
            return values.fillna(0).astype(float)

        return ((values - min_val) / (max_val - min_val)).fillna(0).clip(0, 1)

    def _minmax_to_100(self, s: pd.Series) -> pd.Series:
        min_val = s.min()
        max_val = s.max()
        if min_val == max_val:
            return pd.Series(50.0, index=s.index)
        return ((s - min_val) / (max_val - min_val) * 100).round(2)

    def _priority_band(self, score: float) -> str:
        if score >= self.config.high_priority_threshold:
            return "high"
        if score >= self.config.medium_priority_threshold:
            return "medium"
        return "low"

    def _recommended_action(self, row: pd.Series) -> str:
        if row["priority_band"] == "high":
            return "Immediate analyst review"
        if row["priority_band"] == "medium":
            return "Queue for structured follow-up review"
        return "Retain for monitoring / batch review"

    def _build_top_signals(self, row: pd.Series) -> str:
        signals: List[str] = []

        if row.get("minor_flag", 0) == 1:
            signals.append("minor")
        if row.get("missing_history_flag", 0) == 1:
            signals.append("missing-history")
        if row.get("high_signal_hit_count", 0) > 0:
            terms = row.get("matched_high_signal_terms", [])
            if terms:
                signals.append(f"high-keywords:{', '.join(terms[:3])}")
        if row.get("medium_signal_hit_count", 0) > 0:
            terms = row.get("matched_medium_signal_terms", [])
            if terms:
                signals.append(f"medium-keywords:{', '.join(terms[:2])}")

        return " | ".join(signals[:4]) if signals else "no strong explicit signals"

    def _ordered_output_columns(self, df: pd.DataFrame) -> List[str]:
        preferred = [
            self.config.id_col,
            "risk_score",
            "priority_band",
            "recommended_action",
            "within_review_capacity",
            "top_signals",
            "high_signal_hit_count",
            "medium_signal_hit_count",
            "matched_high_signal_terms",
            "matched_medium_signal_terms",
            "minor_flag",
            "missing_history_flag",
            "prior_contacts_score",
            "location_risk_score_norm",
        ]
        remaining = [c for c in df.columns if c not in preferred and not c.startswith("_")]
        return [c for c in preferred if c in df.columns] + remaining


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "case_id": [1001, 1002, 1003],
            "incident_text": [
                "Minor involved, possible trafficking indicators near hotel.",
                "General disturbance report with no clear exploitation signal.",
                "Repeated runaway history, coercion suspected, commercial sex mention.",
            ],
            "notes": [
                "Older boyfriend noted. Missing history present.",
                "Routine case.",
                "Victim appeared fearful and withdrawn.",
            ],
            "age": [16, 27, 17],
            "prior_contacts": [3, 0, 5],
            "has_missing_history": [1, 0, 1],
            "location_risk_score": [0.9, 0.1, 0.8],
        }
    )

    engine = TrackAPrioritizationEngine()
    ranked = engine.prioritize(sample)
    print(ranked.head(10).to_string(index=False))