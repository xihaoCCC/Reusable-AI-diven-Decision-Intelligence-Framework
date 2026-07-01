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

Frontline anti-trafficking organizations often hold valuable operational records,
including hotline logs, intake forms, referrals, outreach notes, case notes,
service records, and public-safety information. However, many of these organizations
lack the data science staff, technical infrastructure, or labeled local data needed
to build AI-enabled decision-support systems from scratch.

This framework addresses that adoption gap by:

- **Mapping heterogeneous frontline records into usable decision-support features**: The framework helps convert locally defined and often incomplete records into standardized trafficking-related concepts through field normalization, missingness handling, data-quality checks, and operational data mapping.
- **Providing a reusable AI Core that organizations do not need to build from scratch**: The framework supports reusable technical components, including classification modules, scoring logic, ranking tools, evaluation utilities, monitoring functions, and explanation outputs. This enables cold-start adoption with pre-trained or rule-informed modules and supports later local learning as more labels and reviewer feedback become available.
- **Preserving local flexibility and human accountability**: The framework separates model outputs from organization-specific decision rules, allowing each organization to configure review capacity, confidence thresholds, priority weights, escalation rules, and referral pathways. AI-generated outputs are presented as suggestions and explanations for human review, not automated final decisions.

## Framework Architecture

### 1. Operational Data Mapping Layer

Maps heterogeneous local records into **HTCDS+**, the project's canonical bridge
schema. HTCDS+ retains official HTCDS fields and adds curated analytical concepts with
documented types, values, provenance, and missing-value semantics.

Dataset raw fields, HTCDS+ fields, model-specific features, and inference-time
availability remain separate. See [docs/htcds_plus_schema.md](docs/htcds_plus_schema.md)
and [docs/ctdc_htcds_mapping.md](docs/ctdc_htcds_mapping.md).

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

Each released model publishes its HTCDS+ version, complete feature list, missing-value
policy, and an importance-reviewed top-K **Core** feature list. Local data must cover
all Core features. Missing non-Core features may use the artifact's validated
imputation policy.

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

## HTCDS+ And Missing Data

HTCDS+ uses `NA` for not collected, not applicable, unknown, or not reported.

- Binary `0` means explicitly negative; unknown remains `NA`.
- Categorical unknown remains `NA`.
- Numeric unknown remains `NA` until model-specific preprocessing.
- XGBoost receives `NaN` directly and learns default missing-value branches.
- Logistic regression treats each binary input as a nominal `No`, `Yes`, or `Unknown`
  state. `No` is the reference state; separate `Yes` and `Unknown` dummy variables
  avoid imposing an ordinal relationship.
- Logistic numeric inputs currently use training-set mean imputation with missingness
  indicators.

The current analytical schema groups physical, psychological, and sexual abuse under
`control.abuse`. Official HTCDS definitions remain unchanged in the base standard.

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
and XGBoost as the primary pre-trained module. The v3 candidate-training workflow now
maps CTDC records into HTCDS+, uses a temporal train/validation/test split, applies
class weighting, compares logistic-regression and XGBoost candidates, calibrates soft
probabilities, and exports a reloadable candidate bundle with preprocessing, label,
configuration, evaluation, provenance, and checksum files.

Run the reusable trainer with locally obtained CTDC data:

```bash
pip install -r requirements-training.txt
python examples/train_ctdc_exploitation_type_candidate.py \
  --data /path/to/ctdc_global_synthetic_data_v2026.csv \
  --output /path/to/Local_runner/artifacts/ctdc_exploitation_type/v3_candidate
```

The exploratory `trafficking_type_classification_v3.ipynb` remains in the local
training workspace. The public AI Core uses the reusable modules under
`src/ai_core/exploitation_type/`, never the notebook itself. Candidate bundles are not
released artifacts and must remain outside the public repository until reviewed.

We expect to train multiple compatible variants, including full- and reduced-feature
models. Core features will be assigned only after training, feature-importance and
stability analysis, leakage review, and domain review. The current Core list is
therefore marked `candidate_pending_review`.

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
├── HTCDS_standard/          # Cleaned field standard and control definitions
│   └── HTCDS+ Extensions.yaml # Curated analytical fields and provenance
├── docs/                    # Architecture, governance, artifact, and roadmap docs
├── examples/                # Runnable end-to-end examples
├── notebooks/               # Public reproduction and demonstration notebooks
├── outputs/                 # Sample tables, figures, and review queues
├── sample_data/             # Documentation and approved public-safe samples only
├── src/
│   ├── data_mapping/        # Operational Data Mapping Layer
│   ├── ai_core/             # Reusable models, scoring, evaluation, explanations
│   │   └── artifacts/       # Minimal versioned inference bundles and model cards
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

The current demonstration produces two sample review queues under
`outputs/sample_review_queues/`. Its locally trained fallback demonstrates the
framework interface and is separate from the packaged AI Core research release.

The AI Core now includes a packaged CTDC-informed exploitation-type research release.
See [src/ai_core/README.md](src/ai_core/README.md) for the artifact registry and model
documentation. Research release status does not imply field validation or operational
approval.

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
