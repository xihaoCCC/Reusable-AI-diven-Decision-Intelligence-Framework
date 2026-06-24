# Framework Overview

This repository implements the submitted paper's four-layer framework for frontline anti-trafficking decision support.

## 1. Operational Data Mapping Layer

The mapping layer converts agency-owned records into standardized trafficking-related features. Inputs may include hotline logs, intake forms, referral records, outreach notes, case notes, service records, and public-safety records.

Core responsibilities:

- Normalize local field names.
- Handle missing indicators explicitly.
- Map local concepts into standardized indicator groups.
- Document mapping assumptions.
- Restrict protected or sensitive attributes by default.
- Support CTDC/HTCDS-style alignment without requiring any organization to expose raw operational records.

## 2. Standardized AI Core

The AI core contains reusable task modules and utilities. In the current prototype, the central task module is an exploitation-type classifier with three classes: `Sex`, `Labor`, and `Both`.

The classifier outputs soft probabilities and confidence scores that can be reused across multiple decision scenarios without retraining the underlying model.

## 3. Configurable Decision Layer

The decision layer converts model outputs into organization-specific triage priorities. Scenario configuration files define target probabilities, review capacity, confidence thresholds, indicator focus, route labels, and scoring weights.

This separation is important: a small service provider and a labor task force may use the same AI core but prioritize different review queues.

## 4. Human Review and Presentation Layer

The presentation layer produces ranked review queues for trained staff. Queue outputs include predicted exploitation type, probability outputs, priority score, key indicators, suggested route, and reviewer-action placeholders.

Reviewer actions can later become feedback signals for monitoring, threshold adjustment, and local-learning deployment.

