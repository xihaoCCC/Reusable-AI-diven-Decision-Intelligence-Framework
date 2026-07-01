# Project Context

Read this file before changing the repository.

## Purpose

This repository is the codebase for a reusable AI-driven decision-support framework
for frontline anti-trafficking operations. It is broader than any single research
prototype or publication.

The framework helps nonprofits, victim-service providers, outreach partners, and
public-safety agencies map heterogeneous agency-owned records into standardized
concepts, apply reusable AI/data-science artifacts, configure organization-specific
triage and routing rules, and present explainable review queues to trained staff.

The system supports human decision-making. It is not an autonomous
victim-identification system and does not make final legal, service, investigative, or
case-management decisions.

## Relationship To The Paper

The paper, "From Detection to Frontline Decision Support: A Configurable AI
Framework for Anti-Trafficking Triage and Resource Allocation," formally introduces
the framework and demonstrates one cold-start deployment pattern.

The CTDC-informed exploitation-type classification prototype is the first implemented
AI Core task module. It is not the full scope or permanent identity of the repository.
Future task modules may include service-need prediction, urgency scoring, referral
routing, text mining, online-signal extraction, and other responsibly developed
components.

## Architecture

1. Operational Data Mapping Layer
2. Standardized AI Core
3. Configurable Decision Layer
4. Human Review and Presentation Layer

The AI Core is a modular artifact registry and technical backbone. It should contain
versioned pre-trained models, rule-informed modules, scoring/ranking components,
evaluation utilities, monitoring functions, explanation tools, schemas, and artifact
metadata.

The Configurable Decision Layer owns organization-specific parameters such as target
probabilities, priority weights, review capacity K, confidence thresholds, escalation
criteria, protected-field restrictions, service/referral rules, and routing logic.

## CTDC Exploitation-Type Module

The first pre-trained task module predicts:

- Sex / sexual exploitation only
- Labor / forced labor only
- Both / mixed sexual and labor exploitation

It outputs `P(Sex)`, `P(Labor)`, `P(Both)`, and confidence. The intended artifact
bundle should include the trained XGBoost model, logistic-regression baseline,
preprocessing objects, label encoder, ordered feature schema, training/evaluation
metadata, model card, threshold analysis, interpretability outputs, and checksums.

Age and gender may be included as model features. Their use must be documented,
audited, evaluated through ablation/fairness analysis, and configurable or restricted
for downstream operational use.

The paper's self-defined priority-score formula is a reusable scoring component:

`100 * (w_conf*confidence + w_target*target_probability + w_control*control
+ w_vulnerability*vulnerability + w_relationship*relationship)`

The scoring engine is reusable; scenario-specific weights and feature groups belong in
configuration.

## Local Training Workspace

Use the sibling folder `../Local_runner/` for local or temporary training work that
should not be committed, including raw CTDC data, exploratory notebooks, tuning
outputs, model checkpoints, caches, large intermediate tables, and candidate artifacts.

Only reviewed, reproducible source code, approved documentation, and release-ready
artifact bundles should be promoted into this repository.

## Data Policy

Do not commit the CTDC Global Synthetic Dataset or CTDC codebook until redistribution
permission has been confirmed. Do not commit real agency-owned anti-trafficking data.
Use download/setup instructions, local paths, or controlled artifact release mechanisms.

## Current Priorities

The v3 candidate foundation now includes HTCDS+ preparation, temporal validation,
weighted logistic/XGBoost training, calibration, evaluation, candidate serialization,
and artifact reload inference. The working notebook and candidate binaries remain in
`Local_runner/`.

1. Improve and stress-test minority `Both` performance, especially the calibrated
   argmax recall tradeoff.
2. Add repeated-seed, ablation, subgroup, missingness, and classwise calibration
   analyses.
3. Evaluate full and reduced feature variants and Core-feature stability.
4. Define acceptance criteria and complete model-card and governance review.
5. Promote a versioned artifact only after CTDC distribution terms are confirmed.
6. Continue implementing all four framework layers beyond the initial prototype.

## HTCDS+ Mapping Conventions

HTCDS+ is the project's canonical bridge schema. The official foundation is stored at
`HTCDS_standard/HTCDS Field Standards 2.0.xlsx`; curated fields and provenance are in
`HTCDS_standard/HTCDS+ Extensions.yaml`. Keep official fields, project extensions,
dataset raw fields, model features, and inference-time availability separate.

Do not force a lossy source field into a more precise HTCDS value. Mark partial
mappings and promote reusable extensions to source-neutral HTCDS+ names. Unknown
binary values remain `NA`; zero is reserved for explicit negatives. The current CTDC
mapping is documented in `docs/ctdc_htcds_mapping.md`.

Model preprocessing is task- and estimator-specific. The CTDC XGBoost candidate keeps
`NaN` for native missing-value handling. Its logistic baseline encodes binary fields as
nominal `No`, `Yes`, and `Unknown` states, using `No` as the reference category.

Do not label model features Core before training. Each released model variant must
publish an importance-reviewed Core list, and inference data must cover all Core
features.
