from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def classification_summary(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, object]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }

