from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class ModelCompatibilityReport:
    status: str
    missing_core_features: tuple[str, ...]
    missing_non_core_features: tuple[str, ...]

    @property
    def is_compatible(self) -> bool:
        return self.status in {"compatible", "compatible_with_missingness"}


@dataclass(frozen=True)
class ModelFeatureConfig:
    """Task feature contract kept separate from the HTCDS+ bridge schema."""

    task_name: str
    label_field: str
    numeric_features: tuple[str, ...] = ()
    binary_features: tuple[str, ...] = ()
    core_feature_status: str = "pending_model_training"
    core_feature_selection_method: str = "top_k_feature_importance"
    core_feature_top_k: int = 0
    core_features: tuple[str, ...] = ()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelFeatureConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        core_policy = payload.get("core_feature_policy", {})
        config = cls(
            task_name=str(payload["task_name"]),
            label_field=str(payload["label_field"]),
            numeric_features=tuple(payload.get("numeric_features", [])),
            binary_features=tuple(payload.get("binary_features", [])),
            core_feature_status=str(
                core_policy.get("status", "pending_model_training")
            ),
            core_feature_selection_method=str(
                core_policy.get("selection_method", "top_k_feature_importance")
            ),
            core_feature_top_k=int(core_policy.get("top_k", 0)),
            core_features=tuple(core_policy.get("features", [])),
        )
        config._validate_contract()
        return config

    def validate_columns(self, available_columns: set[str]) -> None:
        required = {self.label_field, *self.model_features}
        missing = sorted(required - available_columns)
        if missing:
            raise ValueError(f"Model feature configuration requires missing fields: {missing}")

    def assess_inference_compatibility(
        self, feature_frame: pd.DataFrame
    ) -> ModelCompatibilityReport:
        unavailable = {
            feature
            for feature in self.model_features
            if feature not in feature_frame.columns or feature_frame[feature].isna().all()
        }
        missing_core = tuple(sorted(unavailable.intersection(self.core_features)))
        missing_non_core = tuple(sorted(unavailable - set(self.core_features)))
        if self.core_feature_status != "defined":
            status = "core_features_pending"
        elif missing_core:
            status = "incompatible"
        elif missing_non_core:
            status = "compatible_with_missingness"
        else:
            status = "compatible"
        return ModelCompatibilityReport(status, missing_core, missing_non_core)

    def _validate_contract(self) -> None:
        if set(self.numeric_features).intersection(self.binary_features):
            raise ValueError("A model feature cannot be both numeric and binary")
        invalid_core = sorted(set(self.core_features) - set(self.model_features))
        if invalid_core:
            raise ValueError(f"Core features are not model features: {invalid_core}")
        if self.core_features and self.core_feature_status != "defined":
            raise ValueError("core_feature_status must be 'defined' when features are listed")

    @property
    def model_features(self) -> tuple[str, ...]:
        return (*self.numeric_features, *self.binary_features)
