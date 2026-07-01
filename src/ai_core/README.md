# Standardized AI Core

The AI Core contains reusable, versioned analytical components for anti-trafficking
decision support. Models produce review signals for the Configurable Decision Layer;
they do not make final legal, service, eligibility, or investigative decisions.

## Artifact Registry

| Artifact | Version | Status | Task | Outputs |
|---|---|---|---|---|
| [CTDC-informed XGBoost exploitation-type classifier](artifacts/exploitation_type/ctdc_xgboost/v0.1.0/README.md) | `0.1.0` | Research release | Three-class exploitation-type classification | `P(Sex)`, `P(Labor)`, `P(Both)`, predicted class, confidence |

`Research release` means the artifact is packaged and reusable but has not been field
validated or approved for autonomous or operational decision-making.

## Artifact Layout

Future model artifacts should use this structure:

```text
artifacts/
└── <task>/
    └── <artifact_name>/
        └── <semantic_version>/
            ├── model.joblib
            ├── label_encoder.joblib
            ├── feature_config.yaml
            ├── artifact_manifest.json
            ├── requirements.txt
            ├── checksums.sha256
            ├── README.md
            └── MODEL_CARD.md
```

Only inference and reuse files belong in a released artifact directory. Training
notebooks, alternate candidates, plots, feature-importance outputs, and exploratory
evaluation tables remain in the controlled development workspace.

## Common Loading Pattern

```python
from src.ai_core.exploitation_type import ExploitationTypeArtifactPredictor

predictor = ExploitationTypeArtifactPredictor("path/to/artifact/version")
predictions = predictor.predict_proba(htcds_plus_feature_frame)
```

Input records must already be mapped into the artifact's HTCDS+ feature contract.
Always read the artifact's model card and perform local validation before use.
