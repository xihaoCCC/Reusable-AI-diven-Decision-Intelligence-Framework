# Anti Huamn Trafficking Decision Support Toolkit (AHTDST)

TDST is a **plug-and-play, privacy-aware, data-driven decision-support toolkit** designed to help anti-human-trafficking stakeholders (e.g., NGOs, social service partners, and public agencies) **prioritize outreach, triage leads, optimize resource allocation, and coordinate response workflows** using configurable data pipelines, **AI/ML-ready risk scoring**, and interactive dashboards.

This toolkit is built and maintained from the perspective of a **data scientist with domain knowledge in anti-trafficking analytics**, with an emphasis on **replicability, explainability, and responsible use**.

> **Important note:** This repository is designed to be **replicable across organizations and jurisdictions**. It includes **only synthetic/demo data** and provides guidance for secure integration with an organization’s own data systems. **Do not upload or run this project using real case data without appropriate legal authority, privacy controls, and partner approvals.**

---

## Why this project exists

Human trafficking response work often involves:
- fragmented data sources and inconsistent schemas,
- limited analytic capacity at frontline organizations,
- high-volume leads that require careful prioritization,
- the need for secure, auditable, and explainable decision support.

TDST aims to reduce adoption barriers by providing a reusable foundation for **data-driven and AI-enabled decision support**, including:
- a **configurable pipeline** that can run with minimal engineering effort,
- **transparent scoring** (baseline analytics today, ML-ready interfaces for expansion),
- **evaluation components** aligned to real-world triage,
- a **standardized data schema** and validation utilities,
- Interactive **dashboards and reporting** modules for direct display and operational use.

---

## What TDST provides (current / planned)

### Core capabilities
- **Data ingestion & validation**: load partner data (CSV now; database connectors planned) and validate against a standard schema
- **Feature engineering**: modular, documented transformations to support consistent modeling across partners
- **Risk / priority scoring**:
  - baseline scoring strategies for immediate triage support
  - ML-ready model interfaces (e.g., logistic regression / tree-based models) for partner-specific training and evaluation
- **Evaluation framework**: metrics and review workflows appropriate for operational decision support (not just offline accuracy)
- **Reporting & export**: generate prioritized review outputs with explainability summaries where applicable

### Optional modules (as the project expands)
- **Dashboard** for triage queue management, monitoring trends, and model/explainability summaries
- **API service** for integration into partner systems and case management workflows
- **Deployment templates** (Docker-first) for scalable rollout
- **AI ChatBot** for easier results retriving and better streaming

---

## Who it’s for
TDST is intended for:
- NGOs and service providers supporting victims and survivors,
- partner agencies coordinating response efforts,
- teams seeking a reusable **data/AI foundation** without building everything from scratch.

This toolkit is **not** a substitute for professional judgment. It is meant to **support** decision-making with transparent, auditable signals.

---

## Design principles
TDST is designed around:
- **Replicability**: standardized schema + configuration-driven setup across jurisdictions
- **Privacy & safety**: strong default guidance to reduce risk when working with sensitive data
- **Explainability**: outputs should be interpretable, reviewable, and auditable
- **Operational realism**: focuses on triage and workflow integration, not just model performance
- **Responsible AI**: emphasizes evaluation, monitoring, and human-in-the-loop review

---

## Quick start (placeholder)
This repository is under active development. A full quickstart guide will be provided in `docs/03_quickstart.md`.

Intended flow:
1. Configure data sources and outputs in a YAML config file (examples in `configs/`)
2. Run the pipeline to generate **data-driven / ML-ready** prioritized review outputs
3. Review evaluation metrics and calibrate thresholds for operational use
4. (Optional) Launch a dashboard for case triage and monitoring



