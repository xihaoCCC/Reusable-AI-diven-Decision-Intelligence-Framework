# Framework Overview

This repository implements a reusable AI-driven decision-support framework for
frontline anti-trafficking operations. The publication introduces the architecture and
demonstrates one CTDC-informed module; the repository is the broader, evolving
technical implementation.

## Design Goals

- Low-overhead adoption by organizations without dedicated data-science teams.
- Reuse of reviewed AI and data-science artifacts across organizations.
- Clear separation between analytical signals and local operational policy.
- Cold-start deployment followed by progressive local learning.
- Explainable, auditable, human-in-the-loop review.
- Privacy, governance, and sensitive-field controls by design.

## 1. Operational Data Mapping Layer

The mapping layer converts agency-owned records into documented standardized concepts
while retaining useful local fields. Inputs may include hotline logs, intake forms,
referral records, outreach notes, case notes, service records, and public-safety
records.

Responsibilities include:

- local field normalization;
- missingness policies and warnings;
- local-to-standard concept mapping;
- HTCDS-aligned mapping documentation;
- data-quality and mapping-coverage checks;
- protected-field restrictions; and
- stable feature contracts for downstream artifacts.

## 2. Standardized AI Core

The AI Core is a modular technical backbone, not a single classifier. It can contain:

- pre-trained task modules;
- rule-informed scoring and ranking modules;
- model and feature schemas;
- evaluation and threshold-analysis utilities;
- calibration tools;
- explanation and interpretability functions;
- data-quality, drift, and performance monitors; and
- artifact metadata and loading interfaces.

The CTDC-informed exploitation-type classifier is the first task module. Future modules
may address service needs, urgency, referral routing, text mining, or other tasks when
appropriate data and governance are available.

See [ai_core_artifacts.md](ai_core_artifacts.md).

## 3. Configurable Decision Layer

The decision layer converts AI Core outputs into organization-specific triage,
routing, and resource-allocation logic.

Configuration may define:

- target probabilities;
- review capacity;
- confidence thresholds;
- priority weights;
- control, vulnerability, and relationship indicators;
- escalation criteria;
- protected-field policies; and
- service/referral routing.

The reusable scoring engine and organization-specific configuration remain separate so
that local policy is visible, reviewable, and changeable without retraining a model.

## 4. Human Review And Presentation Layer

The presentation layer produces ranked review queues and feedback interfaces. Outputs
may include predicted task labels, probability distributions, confidence, component
scores, key indicators, missing-data warnings, suggested routes, reviewer actions, and
audit fields.

Reviewer corrections and decisions can become governed feedback signals for threshold
adjustment, monitoring, and future local learning.

## Deployment Modes

In **cold-start mode**, organizations use suitable pre-trained or rule-informed modules
and validate them against local needs before operational use.

In **local-learning mode**, organizations with sufficient governed labels or reviewer
feedback can recalibrate, fine-tune, retrain, or replace task modules while preserving
the framework interfaces.

