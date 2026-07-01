from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def build_logistic_preprocessor(
    numeric_features: Sequence[str],
    binary_features: Sequence[str],
    numeric_imputation_strategy: str = "mean",
    add_numeric_missing_indicators: bool = True,
) -> ColumnTransformer:
    """Build linear-model preprocessing with nominal binary-state encoding."""

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy=numeric_imputation_strategy,
                                add_indicator=add_numeric_missing_indicators,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            )
        )
    if binary_features:
        categories = [["No", "Yes", "Unknown"] for _ in binary_features]
        references = ["No" for _ in binary_features]
        transformers.append(
            (
                "binary_three_state",
                Pipeline(
                    [
                        (
                            "state_labels",
                            FunctionTransformer(_binary_to_state_labels, validate=False),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(
                                categories=categories,
                                drop=references,
                                handle_unknown="error",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                list(binary_features),
            )
        )
    if not transformers:
        raise ValueError("At least one numeric or binary feature must be configured")
    return ColumnTransformer(transformers, remainder="drop")


def build_native_missing_preprocessor(
    model_features: Sequence[str],
) -> ColumnTransformer:
    """Select ordered numeric columns while preserving NaN for tree models."""

    if not model_features:
        raise ValueError("At least one model feature must be configured")
    return ColumnTransformer(
        [("native_numeric", "passthrough", list(model_features))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _binary_to_state_labels(values) -> np.ndarray:
    frame = pd.DataFrame(values)
    invalid = ~(frame.isna() | frame.eq(0) | frame.eq(1))
    if invalid.to_numpy().any():
        invalid_values = sorted(
            {str(value) for value in frame.where(invalid).stack().unique()}
        )
        raise ValueError(
            "Binary features must contain only 0, 1, or missing values; "
            f"found {invalid_values}"
        )

    states = np.full(frame.shape, "Unknown", dtype=object)
    states[frame.eq(0).to_numpy()] = "No"
    states[frame.eq(1).to_numpy()] = "Yes"
    return states
