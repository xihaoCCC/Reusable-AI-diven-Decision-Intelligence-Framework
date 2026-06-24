from __future__ import annotations

from typing import Dict


def build_deployment_template(track_name: str) -> Dict[str, object]:
    """
    Starter deployment template for new adopters of the framework.
    """

    return {
        "track_name": track_name,
        "schema_mapping": {},
        "enabled_modules": [],
        "thresholds": {},
        "capacity_limits": {},
        "workflow_integration": {
            "output_format": "csv",
            "dashboard_enabled": False,
            "human_review_required": True,
        },
    }
