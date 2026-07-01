# Changelog

## V3 CTDC Candidate Training

- Added a reusable HTCDS+-based exploitation-type training package under
  `src/ai_core/exploitation_type/`.
- Added temporal validation, class weighting, logistic and XGBoost comparison,
  probability calibration, per-class metrics, threshold analysis, and permutation
  importance.
- Added candidate artifact packaging with preprocessing, label encoding, provenance,
  evaluation files, checksums, and an inference loader.
- Added a separate training dependency file and configuration-driven v3 trainer.
- Split preprocessing by estimator: XGBoost now uses native `NaN` handling, while
  logistic regression uses nominal `No`/`Yes`/`Unknown` binary states with `No` as the
  reference category.
- Kept raw CTDC data, exploratory notebook state, and candidate binary artifacts in
  `Local_runner/` pending review and distribution approval.
- Added the calibrated CTDC-informed XGBoost pipeline to the AI Core as a minimal,
  versioned research release with a manifest, runtime lock, checksums, and model card.

## Current Refocus

Removed or archived:

- Moved the prior Track B supply-chain decision-intelligence package into `archive/legacy_two_track_repo/`.
- Moved old forecasting models, Track B tests, and Track B sample demand data out of the active public-facing workflow.
- Moved the old Track A package and previous shared-core implementation into the same archive to avoid mixed two-track framing.

Added:

- New four-layer `src/` organization aligned with the submitted anti-trafficking paper.
- CTDC-informed synthetic record generator.
- Operational data mapping layer for standardized trafficking-related indicators.
- Exploitation-type classifier with logistic regression baseline and optional XGBoost primary module.
- Configurable decision-layer scenario scoring.
- Human-review queue export with reviewer-action placeholders.
- Scenario YAML files for a small NGO multidisciplinary queue and a labor-exploitation task-force queue.
- Responsible-use, data-mapping, framework-overview, and triage-configuration documentation.

Alignment:

- The active repository now focuses on frontline anti-trafficking triage, routing, and resource-allocation decision support.
- Public-facing documentation no longer presents the project as a two-track framework.
- The prototype emphasizes human-in-the-loop review, local validation, privacy governance, and non-autonomous decision support.

## Framework-First Clarification

- Reframed the repository as the reusable anti-trafficking decision-support framework,
  rather than as a repository for one publication prototype.
- Defined the CTDC exploitation-type classifier as the first AI Core task module in a
  future registry of versioned pre-trained and rule-informed artifacts.
- Added durable project context in `AGENTS.md`, an artifact contract, and a staged
  model-development roadmap.
- Documented `Local_runner/` as the workspace for raw data, exploratory training,
  tuning outputs, checkpoints, and candidate artifacts.
- Excluded the CTDC dataset and codebook from Git until redistribution permission is
  confirmed.
- Clarified that age and gender may be modeled only with explicit disclosure,
  sensitive-feature evaluation, governance, and local validation.

## HTCDS Mapping Layer

- Added a reusable loader for the cleaned 20-field HTCDS Excel standard.
- Added an explicit CTDC-to-HTCDS mapping specification and coverage report.
- Classified CTDC fields as covered, partial, or source-specific extensions.
- Preserved lossy CTDC composite-control and recruiter signals as explicit extensions
  instead of expanding them into unsupported precise HTCDS values.
- Added a separate YAML configuration for the 19 features selected by the current
  exploitation-type classification task.
- Updated the public-safe synthetic pipeline to exercise raw CTDC-shaped input,
  standardized mapping, model feature selection, classification, and triage end to end.

## HTCDS+ Bridge Schema

- Named the project bridge schema HTCDS+ and added a central extension/provenance
  registry.
- Replaced CTDC-prefixed reusable fields with source-neutral HTCDS+ names.
- Grouped physical, psychological, and sexual control abuse as `control.abuse` while
  retaining official HTCDS definitions in the base schema.
- Preserved unknown binary and categorical values as `NA`; zero now requires an
  explicit negative.
- Added configurable mean numeric imputation with missingness indicators.
- Added model compatibility and Core-feature metadata. The current Core list remains
  pending until the improved model is trained and reviewed.
