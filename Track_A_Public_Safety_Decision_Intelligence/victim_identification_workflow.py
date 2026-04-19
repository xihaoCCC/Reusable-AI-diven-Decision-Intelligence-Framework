from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from standardized_AI_core.models import (
    GradientBoostingVictimClassifier,
    LogisticRegressionVictimClassifier,
    RandomForestVictimClassifier,
)
from Track_A_Public_Safety_Decision_Intelligence.prioritization_engine import (
    TrackAConfig,
    TrackAPrioritizationEngine,
)


CLASSIFIER_REGISTRY = {
    "logistic_regression": LogisticRegressionVictimClassifier,
    "random_forest": RandomForestVictimClassifier,
    "gradient_boosting": GradientBoostingVictimClassifier,
}


@dataclass
class TrackAWorkflowConfig:
    id_col: str = "case_id"
    classifier_model: str = "logistic_regression"
    rule_score_weight: float = 0.6
    victim_probability_weight: float = 0.4


def train_victim_identification_model(
    training_df: pd.DataFrame,
    model_name: str = "logistic_regression",
):
    """
    Train a structured-data classifier for victim-identification support.

    The training frame should include the configured demographic/context
    features plus an `is_victim` label column.
    """

    if model_name not in CLASSIFIER_REGISTRY:
        supported = ", ".join(sorted(CLASSIFIER_REGISTRY))
        raise ValueError(f"Unsupported classifier '{model_name}'. Use one of: {supported}")

    model = CLASSIFIER_REGISTRY[model_name]()
    return model.fit(training_df)


def score_victim_likelihood(
    cases_df: pd.DataFrame,
    trained_model,
    id_col: str = "case_id",
) -> pd.DataFrame:
    """
    Score cases with a trained victim-identification model.
    """

    return trained_model.score_records(cases_df, id_col=id_col)


def build_track_a_review_queue(
    cases_df: pd.DataFrame,
    trained_model=None,
    prioritization_config: Optional[TrackAConfig] = None,
    workflow_config: Optional[TrackAWorkflowConfig] = None,
) -> pd.DataFrame:
    """
    Build a review queue by combining rule-based prioritization with optional
    ML victim-identification probability.
    """

    workflow_config = workflow_config or TrackAWorkflowConfig()
    prioritization_config = prioritization_config or TrackAConfig(
        id_col=workflow_config.id_col
    )

    engine = TrackAPrioritizationEngine(config=prioritization_config)
    ranked = engine.prioritize(cases_df)

    if trained_model is None:
        ranked["combined_priority_score"] = ranked["risk_score"]
        ranked["victim_probability"] = pd.NA
        ranked["predicted_label"] = pd.NA
        return ranked

    model_scores = score_victim_likelihood(
        cases_df=cases_df,
        trained_model=trained_model,
        id_col=workflow_config.id_col,
    )
    ranked = ranked.merge(model_scores, on=workflow_config.id_col, how="left")
    ranked["victim_probability"] = ranked["victim_probability"].fillna(0.0)

    total_weight = (
        workflow_config.rule_score_weight
        + workflow_config.victim_probability_weight
    )
    if total_weight <= 0:
        raise ValueError("At least one Track A workflow weight must be positive")

    ranked["combined_priority_score"] = (
        ranked["risk_score"] * workflow_config.rule_score_weight
        + ranked["victim_probability"] * 100 * workflow_config.victim_probability_weight
    ) / total_weight
    ranked["combined_priority_score"] = ranked["combined_priority_score"].round(2)
    ranked["priority_band"] = ranked["combined_priority_score"].apply(
        lambda score: _priority_band(score, prioritization_config)
    )
    ranked["recommended_action"] = ranked["priority_band"].apply(_recommended_action)
    ranked["top_signals"] = ranked.apply(_append_model_signal, axis=1)

    return ranked.sort_values(
        ["combined_priority_score", "risk_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def summarize_review_queue(review_queue: pd.DataFrame) -> Dict[str, object]:
    """
    Produce lightweight operational summaries for review planning.
    """

    return {
        "case_count": int(len(review_queue)),
        "priority_band_counts": review_queue["priority_band"].value_counts().to_dict(),
        "within_capacity_count": int(
            review_queue.get("within_review_capacity", pd.Series(dtype=bool)).sum()
        ),
        "average_combined_priority_score": round(
            float(review_queue["combined_priority_score"].mean()), 2
        ),
    }


def _priority_band(score: float, config: TrackAConfig) -> str:
    if score >= config.high_priority_threshold:
        return "high"
    if score >= config.medium_priority_threshold:
        return "medium"
    return "low"


def _recommended_action(priority_band: str) -> str:
    if priority_band == "high":
        return "Immediate analyst review"
    if priority_band == "medium":
        return "Queue for structured follow-up review"
    return "Retain for monitoring / batch review"


def _append_model_signal(row: pd.Series) -> str:
    base = row.get("top_signals", "")
    probability = row.get("victim_probability")
    if pd.isna(probability):
        return base
    model_signal = f"ml-victim-probability:{float(probability):.2f}"
    return f"{base} | {model_signal}" if base else model_signal


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "case_id": [1001, 1002, 1003, 1004],
            "incident_text": [
                "Minor involved, possible trafficking indicators near hotel.",
                "General disturbance report with no clear exploitation signal.",
                "Repeated runaway history, coercion suspected, commercial sex mention.",
                "Outreach record notes unstable housing and recruitment concern.",
            ],
            "notes": [
                "Older boyfriend noted. Missing history present.",
                "Routine case.",
                "Victim appeared fearful and withdrawn.",
                "Multiple prior contacts and cash control reported.",
            ],
            "age": [16, 27, 17, 22],
            "prior_contacts": [3, 0, 5, 2],
            "has_missing_history": [1, 0, 1, 0],
            "location_risk_score": [0.9, 0.1, 0.8, 0.6],
            "housing_status": ["unstable", "stable", "unstable", "unstable"],
            "gender": ["female", "male", "female", "female"],
            "is_victim": [1, 0, 1, 1],
        }
    )

    classifier = train_victim_identification_model(sample)
    queue = build_track_a_review_queue(sample, trained_model=classifier)
    print(queue.head(10).to_string(index=False))
    print(summarize_review_queue(queue))
