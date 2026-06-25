# Responsible Use

This project is a reusable decision-support framework. It is not an autonomous
victim-identification system and must not be used as a substitute for trained
professional judgment.

## Human Authority

AI outputs are review signals, not determinations. Human reviewers should retain authority over final legal, service, investigative, outreach, and case-management decisions.

## Local Validation

The CTDC-informed prototype is not field validation. Before any real-world deployment, organizations should conduct local validation using their own governance process, data definitions, review capacity, service pathways, and error-cost assumptions.

## Privacy And Governance

Anti-trafficking records can contain highly sensitive information. Any operational deployment requires access controls, privacy safeguards, data minimization, retention policies, audit logging, reviewer training, and governance procedures for data sharing and model monitoring.

## Protected And Sensitive Attributes

Protected or sensitive attributes should be restricted, audited, or used only when
ethically, legally, and operationally justified. A pre-trained artifact may include age
or gender when they are relevant to the documented training task, but this requires
feature disclosure, ablation and subgroup evaluation, model-card documentation, local
review, and appropriate downstream controls.

Model feature inclusion and decision-policy use are separate questions. An organization
should not automatically use a sensitive model input in routing, eligibility, or
resource-allocation rules.

## Pre-Trained Artifacts

Pre-trained modules require local validation for schema fit, distribution shift,
calibration, subgroup behavior, and operational error costs. Artifact metadata should
identify the training source, task, labels, features, evaluation, intended uses, and
known limitations.

## Misuse Boundaries

The CTDC-informed experiment should not be represented as proof of real-world
deployment performance. The framework should not be used to deny services, automate
enforcement actions, or make final decisions without human review and local
accountability.
