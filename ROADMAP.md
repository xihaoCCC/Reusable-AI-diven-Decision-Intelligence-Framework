# Development Roadmap

This roadmap separates local model development from release-quality framework
components. Work may begin in the sibling `Local_runner/` workspace, but only reviewed
and reproducible code and artifacts should be promoted into this repository.

## Phase 0: Repository And Data Controls

Goals:

- Keep CTDC source data and codebook out of Git until redistribution rights are known.
- Establish `Local_runner/` as the workspace for raw data, exploratory notebooks,
  tuning runs, caches, checkpoints, and candidate artifacts.
- Define the public artifact directory, metadata schema, and release policy.
- Pin reproducible runtime and training dependencies.

Deliverables:

- Data acquisition and local path instructions.
- Dataset fingerprinting script using file size and SHA-256 without publishing data.
- Environment lock or pinned training requirements.
- AI Core artifact specification and model-card template.
- Automated checks that reject raw or sensitive data from commits.

## Phase 1: Rebuild The CTDC Training Pipeline

Convert the publication notebook into modular, testable training code in
`Local_runner/`.

### Data Preparation

- Reproduce the paper filters: U.S. exploitation records, registration year from 2016,
  and identifiable `Sex`, `Labor`, or `Both` labels.
- Make the mixed-exploitation quality filter explicit and configurable.
- Preserve train/test isolation before any resampling, tuning, calibration, or feature
  analysis.
- Produce data-quality, missingness, filtering, and class-distribution reports.
- Record source-data fingerprints and transformation configuration.

### Feature Pipeline

- Reimplement the CTDC mapping as a fitted preprocessing pipeline.
- Retain age and gender as candidate model features.
- Encode gender without artificial order and represent missing/unknown values.
- Encode age brackets with explicit missingness and minor/young indicators.
- Preserve unknown binary controls as `NA`; reserve zero for explicit negatives.
- Compare mean, median, most-frequent, and model-native missing-value handling while
  keeping mean imputation as the initial numeric default.
- Keep an ordered feature contract that inference must validate.

### Baselines And Candidate Models

- Rebuild multinomial logistic regression as the transparent baseline.
- Tune XGBoost as the leading candidate using stratified cross-validation.
- Compare at least one additional strong tabular baseline, such as LightGBM,
  CatBoost, or histogram gradient boosting, subject to dependency and licensing review.
- Treat class imbalance through class/sample weighting and evaluate whether additional
  resampling improves minority-class performance without distorting probabilities.
- Use deterministic seeds and record all hyperparameters.

## Phase 2: Strengthen Validation

Predictive metrics:

- accuracy and balanced accuracy;
- macro- and weighted-F1;
- per-class precision, recall, and F1;
- confusion matrices;
- multiclass log loss and Brier score; and
- stratified cross-validation uncertainty.

Probability quality:

- reliability diagrams and expected calibration error;
- classwise and overall calibration;
- comparison of raw probabilities with Platt, isotonic, or temperature-style
  calibration where appropriate; and
- confidence-threshold coverage/performance tables.

Sensitive-feature and subgroup review:

- full-feature versus no-gender/age ablation;
- age, gender, and intersectional subgroup performance where sample sizes permit;
- missingness subgroup analysis;
- feature-importance stability; and
- documentation of operational restrictions and known limitations.

Robustness:

- repeated train/test splits or nested cross-validation;
- temporal holdout if the data supports it;
- country/year/filter sensitivity analysis;
- simulated distribution-shift stress tests; and
- input-schema and missing-feature failure tests.

Model selection should not optimize only aggregate accuracy. Selection should consider
mixed-exploitation performance, calibrated probabilities, stability, interpretability,
runtime, artifact size, and downstream triage behavior.

For each selected full- or reduced-feature variant, use feature-importance and
stability analysis to propose a top-K Core set. Review Core candidates for leakage,
sensitive-attribute implications, calibration impact, and domain plausibility before
publishing them in the model README.

## Phase 3: Validate Priority Scoring And Triage

Move the priority formula into a standalone reusable scoring API with:

- strict validation that inputs are scaled to `[0, 1]`;
- configurable target-probability expressions;
- configurable control, vulnerability, and relationship aggregations;
- validated weights and documented normalization;
- deterministic top-K ranking and tie handling;
- scenario-specific routing functions; and
- transparent component-level score contributions.

Evaluate scenarios with:

- precision@K and recall@K;
- enrichment over baseline prevalence;
- workload reduction;
- confidence-threshold share;
- queue stability under weight perturbations;
- class and route composition; and
- sensitivity to `K`, thresholds, and missing indicators.

The paper's priority formulas are demonstrations. Operational weights and proxy
high-priority definitions require domain expertise, ethical review, and local policy
approval.

## Phase 4: Package The First AI Core Artifact

Target structure:

```text
artifacts/
└── ctdc_exploitation_type/
    └── v1/
        ├── model.*
        ├── logistic_baseline.*
        ├── preprocessor.*
        ├── label_encoder.*
        ├── feature_schema.json
        ├── artifact_manifest.json
        ├── metrics.json
        ├── threshold_analysis.csv
        ├── model_card.md
        ├── training_config.yaml
        └── checksums.sha256
```

The artifact manifest should include:

- task name and semantic version;
- source-data name and fingerprint, without redistributing data;
- training date and code commit;
- target-label definitions;
- ordered input schema and preprocessing contract;
- output schema;
- library/runtime versions;
- evaluation summary;
- intended uses and prohibited uses;
- sensitive-feature declaration;
- known limitations; and
- artifact checksums.

Promotion gates:

1. Reproducible training run from configuration.
2. Passing unit, integration, schema, and serialization tests.
3. Evaluation report reviewed against predefined acceptance criteria.
4. Model card and responsible-use review complete.
5. No raw CTDC or agency-owned data in the bundle.
6. Artifact licensing and distribution method confirmed.

If binary artifacts are unsuitable for ordinary Git, publish them through a versioned
release or model registry and keep the manifest and download verification logic in the
repository.

## Phase 5: Integrate Artifact Inference

Add:

- artifact registry and discovery API;
- verified download or local-install workflow;
- common `load`, `validate_input`, `predict_proba`, and `explain` interface;
- schema mapping from standardized framework concepts to model features;
- graceful missing-field diagnostics;
- batch inference and sample CLI;
- artifact compatibility tests; and
- cold-start example using the released model without retraining.

The current fallback demonstration classifier should then be clearly separated from the
released CTDC artifact.

## Phase 6: Complete The Framework Layers

Operational Data Mapping:

- mapping-spec schema;
- HTCDS concept registry;
- mapping-quality and coverage reports;
- configurable missingness policies; and
- local field retention and protected-field controls.

AI Core:

- artifact registry;
- scoring/ranking modules;
- evaluation suite;
- explainability utilities;
- drift and data-quality monitoring; and
- interfaces for future task modules.

Configurable Decision Layer:

- validated scenario schema;
- target-probability expression support;
- routing and escalation rules;
- protected-field policies;
- configuration versioning; and
- scenario comparison tools.

Human Review And Presentation:

- review-queue schema;
- correction, override, and referral feedback capture;
- reason and confidence displays;
- audit events;
- export templates; and
- a lightweight reference interface or dashboard.

## Phase 7: Local Learning And Future Artifacts

- Define a safe local-training/fine-tuning workflow.
- Separate organization data and artifacts from public framework files.
- Support threshold recalibration before full retraining.
- Add reviewer-feedback schemas and monitoring reports.
- Develop future artifacts only when task-aligned data, governance, and evaluation are
  sufficient.
- Candidate future tasks include service-need prediction, urgency scoring, referral
  routing, text-mining assistance, and operational workload forecasting within
  anti-trafficking programs.

## Immediate Next Sprint

1. Scaffold `Local_runner/ctdc_exploitation_type/`.
2. Move the publication notebook there as a read-only reference copy.
3. Implement reproducible data preparation and feature-building modules.
4. Add dataset fingerprinting and training configuration.
5. Reproduce the paper metrics exactly before changing model logic.
6. Add improved cross-validation, calibration, subgroup, and robustness evaluation.
7. Tune candidate models and define model-selection criteria.
8. Create a candidate artifact bundle and exercise the public inference API.
9. Review documentation, governance, and distribution before release.
