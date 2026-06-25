# Reusable AI-Driven Decision Support for Anti-Trafficking Operations

This repository is the implementation foundation for a reusable, configurable, and
human-in-the-loop AI decision-support framework for frontline anti-trafficking
operations.

It is designed to help nonprofits, victim-service providers, outreach partners, and
public-safety agencies convert heterogeneous agency-owned records into standardized
features, reusable AI signals, configurable triage priorities, routing suggestions, and
review queues.

The repository is broader than a single model or research prototype. Its long-term
purpose is to maintain the framework's reusable components and a growing collection of
reviewed, versioned AI Core artifacts that organizations can evaluate and adapt for
local tasks.

This project is **not** an autonomous victim-identification system. It does not make
final legal, service, investigative, eligibility, or case-management decisions.
Outputs are review signals for trained human reviewers.

## Why This Framework Exists

Frontline anti-trafficking organizations often hold useful hotline, intake, referral,
outreach, case-note, service, and public-safety records, but may lack the staff,
infrastructure, or labeled data needed to build operational AI systems from scratch.

The framework addresses this adoption gap by separating:

- local data mapping from reusable technical components;
- model outputs from organization-specific decision rules;
- cold-start use of pre-trained artifacts from later local learning; and
- AI suggestions from accountable human decisions.

## Framework Architecture

### 1. Operational Data Mapping Layer

Maps heterogeneous local records to documented, standardized trafficking-related
concepts while preserving relevant local fields. Responsibilities include field
normalization, missingness handling, local-to-standard concept mapping, HTCDS-aligned
documentation, data-quality checks, and protected-field controls.

### 2. Standardized AI Core

Provides reusable components that organizations do not need to build independently:

- versioned pre-trained task modules;
- rule-informed scoring and ranking modules;
- probability and confidence outputs;
- evaluation and threshold-analysis utilities;
- monitoring and drift checks;
- explanation and interpretability tools;
- schemas, model cards, and artifact provenance; and
- common loading and inference interfaces.

The AI Core is intended to expand over time. Potential modules include
exploitation-type suggestion, service-need prediction, urgency scoring, referral
routing, text mining, and other responsibly developed anti-trafficking tasks.

### 3. Configurable Decision Layer

Translates reusable AI outputs into organization-specific triage and resource-allocation
logic. Configuration may include review capacity `K`, confidence thresholds, target
probabilities, priority weights, escalation criteria, indicator groups, protected-field
restrictions, service/referral rules, and routing logic.

The same AI Core output can support different organizational missions without
retraining the underlying model.

### 4. Human Review and Presentation Layer

Presents ranked queues, probability distributions, confidence, priority scores, key
indicators, missing-data warnings, suggested routes, reviewer-action fields, feedback
capture, and audit-ready records.

AI suggests, scores, ranks, and explains. Trained staff review, contextualize, correct,
override, and decide.

## Deployment Modes

**Cold-start deployment** allows organizations with little or no local labeled data to
begin with reviewed pre-trained task modules or rule-informed scoring. Local validation
is still required before operational use.

**Local-learning deployment** allows organizations with sufficient labeled history or
reviewer feedback to train, fine-tune, recalibrate, or replace modules using locally
governed data and site-specific evaluation.

## First AI Core Module: CTDC Exploitation-Type Classification

The first implemented task module is informed by the CTDC Global Synthetic Dataset
v2026 and predicts three exploitation-type classes:

- `Sex`: sex / sexual exploitation only
- `Labor`: labor / forced labor only
- `Both`: mixed sexual and labor exploitation

Its interface produces:

- `P(Sex)`
- `P(Labor)`
- `P(Both)`
- `confidence = max(P(Sex), P(Labor), P(Both))`

The published prototype used multinomial logistic regression as a transparent baseline
and XGBoost as the primary pre-trained module. The current repository code is an early
implementation. The next development phase will rebuild the training pipeline,
strengthen validation and calibration, and publish a versioned artifact bundle with
preprocessing objects, feature schemas, model cards, evaluation reports, and loading
APIs.

Age and gender are retained as candidate model features. Their contribution and risks
must be evaluated through ablation, subgroup analysis, documentation, and governance
controls. Their inclusion in a trained model does not imply that an organization should
use them operationally without local ethical and legal review.

## Reusable Priority Scoring

The framework includes a configurable priority-scoring component:

```text
Priority_i,s = 100 * (
  w_conf * Conf_i
  + w_target * P_target,i,s
  + w_control * Control_i,s
  + w_vulnerability * Vulnerability_i,s
  + w_relationship * Relationship_i,s
)
```

The scoring engine is reusable. Each organization or scenario controls its target
probability, indicator groups, weights, confidence threshold, review capacity, and
routing rules through configuration.

The repository currently demonstrates:

- a small-NGO multidisciplinary queue using `P(Both)`; and
- a labor-focused task-force queue using `max(P(Labor), P(Both))`.

These scenarios are demonstrations, not universal operational policies.

## Relationship To The Publication

The paper **From Detection to Frontline Decision Support: A Configurable AI Framework
for Anti-Trafficking Triage and Resource Allocation** formally introduces the
architecture and demonstrates a CTDC-informed cold-start prototype.

The paper is the publication describing the framework. This repository is the evolving
framework codebase. The CTDC experiment is the first reference implementation and
reproduction case, not the repository's complete scope.

## Repository Structure

```text
.
├── configs/                 # Organization/scenario decision configuration
├── docs/                    # Architecture, governance, artifact, and roadmap docs
├── examples/                # Runnable end-to-end examples
├── notebooks/               # Public reproduction and demonstration notebooks
├── outputs/                 # Sample tables, figures, and review queues
├── sample_data/             # Documentation and approved public-safe samples only
├── src/
│   ├── data_mapping/        # Operational Data Mapping Layer
│   ├── ai_core/             # Reusable models, scoring, evaluation, explanations
│   ├── decision_layer/      # Configurable triage and routing
│   ├── human_review/        # Review-queue and feedback interfaces
│   ├── evaluation/          # Predictive and operational evaluation
│   └── utils/
├── tests/
├── AGENTS.md                # Durable project context for coding agents
└── ROADMAP.md               # Planned model and framework development
```

Local exploratory training, raw data, tuning runs, candidate models, and temporary
outputs belong in the sibling `Local_runner/` workspace rather than this public repo.

## Current Runnable Demonstration

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/run_ctdc_exploitation_type_prototype.py
```

The current example produces two sample review queues under
`outputs/sample_review_queues/`. It demonstrates the framework interface but should
not be confused with the forthcoming release-quality CTDC artifact.

## Data Availability

The CTDC Global Synthetic Dataset and codebook are not committed because
redistribution permission has not yet been confirmed. Users must obtain data through
an authorized source and follow the applicable terms.

No real hotline, intake, referral, outreach, case-note, service, victim, or
public-safety records are included.

## Responsible Use

- Outputs are review signals, not determinations.
- Human reviewers retain authority over final decisions.
- Local validation is required before real-world deployment.
- Anti-trafficking data require access controls, data minimization, privacy safeguards,
  retention policies, audit logging, and careful governance.
- Sensitive attributes require explicit justification, documentation, auditing, and
  appropriate restrictions.
- Pre-trained artifacts must be evaluated for distribution shift, calibration,
  subgroup performance, and task fit before local use.
- The CTDC-informed experiment is not field validation or proof of deployment
  performance.

See [docs/responsible_use.md](docs/responsible_use.md).

## Development Roadmap

See [ROADMAP.md](ROADMAP.md) for the model-retraining, artifact-release, and broader
framework implementation plan.

## Citation

> Cao, X. *From Detection to Frontline Decision Support: A Configurable AI Framework
> for Anti-Trafficking Triage and Resource Allocation*.

Publication metadata will be added when the final citation details are available.

## License

See [LICENSE](LICENSE).
