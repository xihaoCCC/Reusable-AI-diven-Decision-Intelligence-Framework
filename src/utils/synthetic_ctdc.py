from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ctdc_style_synthetic_records(
    n_records: int = 200, random_state: int = 42
) -> pd.DataFrame:
    """Generate CTDC-informed synthetic records for reproducible demos only."""

    rng = np.random.default_rng(random_state)
    labels = rng.choice(["Sex", "Labor", "Both"], size=n_records, p=[0.45, 0.4, 0.15])
    rows = []
    for idx, label in enumerate(labels, start=1):
        sex_like = label in {"Sex", "Both"}
        labor_like = label in {"Labor", "Both"}
        both_like = label == "Both"

        row = {
            "case_id": f"CTDC-SYN-{idx:04d}",
            "exploitation_type": label,
            "intake_text": _text_for_label(label),
            "case_notes": "Synthetic CTDC-informed prototype record.",
            "referral_notes": "Generated for research workflow demonstration.",
            "threats": _draw_indicator(rng, 0.45 if both_like else 0.30),
            "abuse_indicators": _draw_indicator(rng, 0.55 if sex_like else 0.25),
            "denial_basic_needs": _draw_indicator(rng, 0.55 if labor_like else 0.25),
            "drugs_alcohol_control": _draw_indicator(rng, 0.35 if sex_like else 0.12),
            "false_promises": _draw_indicator(rng, 0.45 if labor_like else 0.18),
            "minor_young_indicator": _draw_indicator(rng, 0.35 if sex_like else 0.10),
            "close_recruiter_relationship": _draw_indicator(
                rng, 0.45 if sex_like else 0.20
            ),
            "debt_bondage": _draw_indicator(rng, 0.55 if labor_like else 0.10),
            "excessive_work_hours": _draw_indicator(rng, 0.70 if labor_like else 0.08),
            "withheld_documents": _draw_indicator(rng, 0.50 if labor_like else 0.08),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _draw_indicator(rng: np.random.Generator, probability: float) -> float:
    return float(rng.binomial(1, probability))


def _text_for_label(label: str) -> str:
    if label == "Sex":
        return "Hotline/intake indicators include sexual exploitation concerns."
    if label == "Labor":
        return "Referral indicators include forced labor and workplace control concerns."
    return "Record includes mixed sexual and labor exploitation indicators."

