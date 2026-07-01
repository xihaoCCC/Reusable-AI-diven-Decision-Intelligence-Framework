# Model Card: Model Name

Use this structure for every versioned model artifact added to the Standardized AI
Core. Keep the version directory minimal and place exploratory outputs in
`Local_runner/`.

## Model Details

- **Artifact ID:** Stable machine-readable identifier
- **Version:** Semantic version
- **Status:** Research release, validated release, or deprecated
- **Model:** Algorithm and calibration method
- **Framework schema:** Compatible HTCDS+ version
- **Task:** Specific prediction or scoring task

List every output field and define its meaning.

## Intended Use

Describe intended users, supported decision-support workflows, and required human
review. State clearly that model outputs are review signals rather than final
determinations.

## Training Data

Document source datasets, versions, filters, population, date range, class
distribution, and source fingerprints without redistributing restricted data.

## Input Contract

Document:

- ordered features and types;
- mappings to HTCDS+ concepts;
- fitted transformations;
- missing-value behavior;
- final Core features and optional features; and
- compatibility behavior when fields are absent.

Unknown is not equivalent to an explicit negative.

## Evaluation Summary

Report held-out and cross-validation results appropriate to the task, including:

- aggregate and per-class metrics;
- probability and calibration metrics;
- subgroup and missingness analyses;
- temporal or distribution-shift checks; and
- downstream decision-support metrics where available.

Identify the evaluation population and avoid presenting synthetic-data results as
field validation.

## Limitations

Document class imbalance, weak classes or subgroups, calibration limitations,
distribution shift, sensitive features, unresolved Core-feature review, and other
known failure modes.

## Prohibited Uses

List autonomous, punitive, surveillance, profiling, or unsupported decision contexts
for which the artifact must not be used.

## Human Review And Governance

State reviewer authority, local-validation requirements, privacy controls, access
restrictions, audit logging, monitoring, escalation, and appeal expectations.

## Runtime And Serialization

Record the supported Python and package versions. Explain how to install the runtime,
verify checksums, and load the model. Warn users never to deserialize untrusted model
files.

## Required Version Directory

```text
<version>/
├── model.joblib
├── label_encoder.joblib       # When class-index decoding is external to the model
├── feature_config.yaml
├── artifact_manifest.json
├── requirements.txt
├── checksums.sha256
├── README.md
└── MODEL_CARD.md
```

Omit files that are genuinely unnecessary for a specific artifact, but do not omit
the manifest, feature contract, runtime specification, integrity verification, or
model documentation.
