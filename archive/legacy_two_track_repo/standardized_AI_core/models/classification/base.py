from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class VictimClassifierConfig:
    """
    Shared config for Track A classification models.

    These models are designed for structured demographic and case-context
    features rather than free-text inputs.
    """

    label_col: str = "is_victim"
    numeric_features: Sequence[str] = field(
        default_factory=lambda: [
            "age",
            "prior_contacts",
            "location_risk_score",
            "household_instability_score",
        ]
    )
    categorical_features: Sequence[str] = field(
        default_factory=lambda: [
            "gender",
            "race_ethnicity",
            "housing_status",
            "employment_status",
            "school_enrollment_status",
        ]
    )


class BaseVictimClassifier(ABC):
    def __init__(self, config: Optional[VictimClassifierConfig] = None) -> None:
        self.config = config or VictimClassifierConfig()
        self.pipeline_: Optional[Pipeline] = None
        self.feature_columns_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "BaseVictimClassifier":
        self._validate_training_input(df)
        X = self._select_features(df)
        y = df[self.config.label_col]

        self.feature_columns_ = list(X.columns)
        preprocessor = self._build_preprocessor(X)
        estimator = self._build_estimator()
        self.pipeline_ = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("estimator", estimator),
            ]
        )
        self.pipeline_.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        pipeline = self._require_fitted_pipeline()
        X = self._select_features(df)
        predictions = pipeline.predict(X)
        return pd.Series(predictions, index=df.index, name="predicted_label")

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        pipeline = self._require_fitted_pipeline()
        X = self._select_features(df)
        probabilities = pipeline.predict_proba(X)[:, 1]
        return pd.Series(probabilities, index=df.index, name="victim_probability")

    def score_records(
        self,
        df: pd.DataFrame,
        id_col: str = "record_id",
    ) -> pd.DataFrame:
        output = pd.DataFrame(index=df.index)
        if id_col in df.columns:
            output[id_col] = df[id_col]
        output["predicted_label"] = self.predict(df)
        output["victim_probability"] = self.predict_proba(df)
        return output.sort_values("victim_probability", ascending=False).reset_index(
            drop=True
        )

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = [
            column for column in self.config.numeric_features if column in X.columns
        ]
        categorical_columns = [
            column for column in self.config.categorical_features if column in X.columns
        ]

        transformers = []
        if numeric_columns:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_columns,
                )
            )

        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", _build_one_hot_encoder()),
                        ]
                    ),
                    categorical_columns,
                )
            )

        if not transformers:
            raise ValueError(
                "No configured feature columns were found in the input dataframe"
            )

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_columns = [
            column
            for column in [
                *self.config.numeric_features,
                *self.config.categorical_features,
            ]
            if column in df.columns
        ]
        if not feature_columns:
            raise ValueError("No supported classification feature columns found")
        return df[feature_columns].copy()

    def _validate_training_input(self, df: pd.DataFrame) -> None:
        if self.config.label_col not in df.columns:
            raise ValueError(
                f"Missing required label column: {self.config.label_col}"
            )

    def _require_fitted_pipeline(self) -> Pipeline:
        if self.pipeline_ is None:
            raise ValueError("Model must be fitted before prediction")
        return self.pipeline_

    @abstractmethod
    def _build_estimator(self):
        raise NotImplementedError


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
