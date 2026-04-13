from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CoreConfig:
    """
    Shared configuration for reusable framework behavior.
    """

    framework_name: str = "Reusable AI Decision-Intelligence Framework"
    standard_id_column: str = "record_id"
    standard_timestamp_column: str = "event_timestamp"
    required_core_fields: List[str] = field(
        default_factory=lambda: ["record_id", "source_system"]
    )
    track_name: Optional[str] = None
    output_dir: str = "outputs"


@dataclass
class DeploymentConfig:
    """
    Deployment-layer configuration kept intentionally thin so the heavy lifting
    stays in the shared core.
    """

    deployment_name: str
    schema_mapping: Dict[str, str] = field(default_factory=dict)
    enabled_modules: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    capacity_limits: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
