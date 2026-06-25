# Changelog

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
