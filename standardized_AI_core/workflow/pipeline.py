from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import pandas as pd

from standardized_AI_core.config.base import CoreConfig, DeploymentConfig
from standardized_AI_core.data_pipeline.ingestion import standardize_dataframe
from standardized_AI_core.decision.base import BaseDecisionModule
from standardized_AI_core.monitoring.quality import summarize_dataframe_quality


class DecisionIntelligencePipeline:
    """
    Early shared pipeline that standardizes input data, captures basic data
    quality telemetry, and delegates decision logic to a track-specific module.
    """

    def __init__(
        self,
        core_config: CoreConfig,
        deployment_config: DeploymentConfig,
        decision_module: BaseDecisionModule,
    ) -> None:
        self.core_config = core_config
        self.deployment_config = deployment_config
        self.decision_module = decision_module

    def run(
        self,
        df: pd.DataFrame,
        optional_columns: Iterable[str] = (),
    ) -> tuple[pd.DataFrame, dict]:
        standardized = standardize_dataframe(
            df=df,
            schema_mapping=self.deployment_config.schema_mapping,
            required_columns=self.core_config.required_core_fields,
            optional_columns=optional_columns,
        )
        quality_report = summarize_dataframe_quality(standardized)
        decisions = self.decision_module.run(standardized)

        metadata = {
            "framework_name": self.core_config.framework_name,
            "track_name": self.core_config.track_name,
            "deployment_name": self.deployment_config.deployment_name,
            "data_quality": asdict(quality_report),
        }
        return decisions, metadata
