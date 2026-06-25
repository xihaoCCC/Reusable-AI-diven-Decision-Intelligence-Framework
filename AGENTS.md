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

1. Rebuild and validate the CTDC exploitation-type training pipeline.
2. Produce a strong, reproducible, versioned pre-trained artifact bundle.
3. Add artifact loading/inference APIs and model-card documentation.
4. Expand evaluation, threshold analysis, interpretability, sensitive-feature auditing,
   monitoring, and local-learning support.
5. Continue implementing all four framework layers beyond the initial prototype.

