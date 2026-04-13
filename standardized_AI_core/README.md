# Standardized AI Core

This directory contains the reusable shared components of the decision-intelligence framework.

Current shared asset kept in place:
- `models/`: placeholder/model storage area for future reusable model artifacts

New shared-core modules added in this phase:
- `config/`: shared framework and deployment configuration models
- `schemas/`: standard record shape and schema validation helpers
- `data_pipeline/`: schema mapping and input standardization utilities
- `decision/`: reusable decision-module contracts and starter prioritizer
- `explainability/`: reason-code and explanation helpers
- `monitoring/`: basic data-quality telemetry
- `templates/`: deployment template builders
- `workflow/`: early end-to-end shared pipeline orchestration

The intent is to keep domain-heavy logic in the deployment tracks while centralizing reusable ingestion, validation, decision-support plumbing, and explainability/monitoring patterns here.
