from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import yaml


@dataclass
class TriageScenario:
    scenario_id: str
    name: str
    target_probability: str
    confidence_threshold: float = 0.7
    review_capacity_k: int = 30
    route_label: str = "Human review"
    indicator_focus: Sequence[str] = field(default_factory=list)
    priority_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "confidence": 0.2,
            "target_probability": 0.35,
            "control": 0.2,
            "vulnerability": 0.15,
            "relationship": 0.1,
        }
    )


def load_scenario(path: str | Path) -> TriageScenario:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return TriageScenario(**payload)


def apply_triage_scenario(scored_records: pd.DataFrame, scenario: TriageScenario) -> pd.DataFrame:
    working = scored_records.copy()
    working["target_probability"] = _target_probability(working, scenario)
    weights = scenario.priority_weights

    working["priority_score"] = 100 * (
        weights.get("confidence", 0.0) * working["confidence"]
        + weights.get("target_probability", 0.0) * working["target_probability"]
        + weights.get("control", 0.0) * working["control_score"]
        + weights.get("vulnerability", 0.0) * working["vulnerability_score"]
        + weights.get("relationship", 0.0) * working["relationship_score"]
    )
    working["priority_score"] = working["priority_score"].round(2)
    working["meets_confidence_threshold"] = (
        working["confidence"] >= scenario.confidence_threshold
    )
    working["route_label"] = scenario.route_label
    working["scenario_id"] = scenario.scenario_id
    working["key_indicators"] = working.apply(
        lambda row: _key_indicators(row, scenario.indicator_focus), axis=1
    )
    working = working.sort_values(
        ["priority_score", "target_probability", "confidence"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    working["review_rank"] = working.index + 1
    working["selected_for_review"] = working["review_rank"] <= scenario.review_capacity_k
    return working


def _target_probability(df: pd.DataFrame, scenario: TriageScenario) -> pd.Series:
    target = scenario.target_probability
    if target == "P(Both)":
        return df["P(Both)"]
    if target == "P(Sex)":
        return df["P(Sex)"]
    if target == "P(Labor)":
        return df["P(Labor)"]
    if target == "max(P(Labor), P(Both))":
        return df[["P(Labor)", "P(Both)"]].max(axis=1)
    raise ValueError(f"Unsupported target_probability: {target}")


def _key_indicators(row: pd.Series, indicator_focus: Sequence[str]) -> str:
    active = [indicator for indicator in indicator_focus if row.get(indicator, 0) >= 0.5]
    return "; ".join(active[:6]) if active else "no configured indicator above threshold"

