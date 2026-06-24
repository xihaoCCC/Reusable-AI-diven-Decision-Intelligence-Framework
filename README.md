# From Detection to Frontline Decision Support

Research prototype for **From Detection to Frontline Decision Support: A Configurable AI Framework for Anti-Trafficking Triage and Resource Allocation**.

This repository implements a configurable AI-enabled decision-support framework for frontline anti-trafficking organizations. It is designed to help nonprofits, victim-service providers, outreach partners, and public-safety agencies convert heterogeneous agency-owned records into explainable triage, routing, and review-queue signals.

This project is **not** an autonomous victim-identification system. It does not make final legal, service, investigative, or case-management decisions. Outputs are review signals for trained human reviewers.

## Motivation

Frontline anti-trafficking work is often framed as a detection problem: identify whether a record indicates trafficking. The paper behind this repository argues that detection alone is not enough. Many organizations also need low-overhead tools for triage, routing, review capacity planning, and resource allocation.

Operational data may arrive as hotline logs, intake forms, referral records, outreach notes, case notes, service records, or public-safety records. These records are heterogeneous, sensitive, and locally governed. The framework therefore separates reusable AI components from local data mapping and configurable decision rules.

## Framework Architecture

The repository follows the paper's four-layer framework.

1. **Operational Data Mapping Layer**  
   Converts heterogeneous agency-owned records into standardized trafficking-related features. This includes field normalization, missingness handling, local-to-standard concept mapping, CTDC/HTCDS-style alignment, data-quality checks, and mapping documentation.

2. **Standardized AI Core**  
   Provides reusable AI/data-science components that frontline organizations do not need to build from scratch. The prototype includes an exploitation-type classifier, probability outputs, confidence scoring, evaluation utilities, monitoring-ready outputs, and explanation-friendly indicators.

3. **Configurable Decision Layer**  
   Converts AI outputs into organization-specific triage logic. Scenario configuration files define review capacity `K`, confidence thresholds, target probabilities, priority weights, escalation criteria, indicator focus, route labels, and protected-field restrictions.

4. **Human Review and Presentation Layer**  
   Produces ranked review queues with predicted exploitation type, probability outputs, priority scores, key indicators, suggested routes, and reviewer-action placeholders. AI suggests, scores, ranks, and explains; trained staff review, contextualize, correct, override, and decide.

## Deployment Modes

**Cold-start deployment** supports organizations with little or no local labeled data. They can use reusable pre-trained task modules or rule-informed scoring, then validate outputs locally before operational use.

**Local-learning deployment** supports organizations with enough labeled local records or reviewer feedback. They can train or fine-tune task modules with their own records, evaluate performance locally, and use reviewer actions as feedback for monitoring and future model updates.

## CTDC-Informed Prototype

The submitted paper includes a CTDC-informed prototype using the CTDC Global Synthetic Dataset v2026. This repository mirrors that workflow with synthetic CTDC-style records generated at runtime for reproducible demonstration.

The prototype trains an exploitation-type classifier for three classes:

- `Sex`: sex / sexual exploitation only
- `Labor`: labor / forced labor only
- `Both`: mixed sexual and labor exploitation

It includes logistic regression as a transparent baseline and an XGBoost-style primary task module. If `xgboost` is installed, the classifier uses it; otherwise it falls back to scikit-learn's histogram gradient boosting so the example remains runnable in lightweight environments.

The model outputs:

- `P(Sex)`
- `P(Labor)`
- `P(Both)`
- `confidence = max predicted probability`

## Example Triage Scenarios

Scenario A, **Small NGO multidisciplinary triage queue**, uses `P(Both)` as the target probability. It prioritizes records that may require broader multidisciplinary coordination.

Scenario B, **Labor-exploitation task force**, uses `max(P(Labor), P(Both))` as the target probability. It prioritizes labor or mixed-exploitation signals.

The priority score follows the paper prototype:

```text
Priority_i,s = 100 * (
  w_conf * Conf_i
  + w_target * P_target,i,s
  + w_control * Control_i,s
  + w_vulnerability * Vulnerability_i,s
  + w_relationship * Relationship_i,s
)
```

The default review capacity is `K = 30` records from a simulated local review set of 200 records in the runnable example. You can change `review_capacity_k` in the scenario YAML files.

## Repository Structure

```text
.
├── README.md
├── configs/
│   ├── scenario_labor_task_force.yaml
│   └── scenario_small_ngo_multidisciplinary.yaml
├── docs/
│   ├── data_mapping.md
│   ├── framework_overview.md
│   ├── responsible_use.md
│   └── triage_configuration.md
├── examples/
│   └── run_ctdc_exploitation_type_prototype.py
├── outputs/
│   ├── figures/
│   ├── sample_review_queues/
│   └── tables/
├── sample_data/
├── src/
│   ├── ai_core/
│   ├── data_mapping/
│   ├── decision_layer/
│   ├── evaluation/
│   ├── human_review/
│   └── utils/
└── tests/
```

Legacy two-track material, including the prior supply-chain forecasting track, has been moved to `archive/legacy_two_track_repo/` and is not part of the active prototype.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional:

```bash
pip install xgboost
```

## Run The Prototype

```bash
python examples/run_ctdc_exploitation_type_prototype.py
```

Expected outputs:

- Top-K multidisciplinary review queue:
  `outputs/sample_review_queues/small_ngo_multidisciplinary_top_k_review_queue.csv`
- Top-K labor task-force review queue:
  `outputs/sample_review_queues/labor_task_force_top_k_review_queue.csv`

Each queue includes `case_id`, route label, priority score, predicted exploitation type, probability outputs, confidence, target probability, key indicators, review selection, and reviewer-action placeholders.

## Responsible Use

This repository supports decision support, not automated final decision-making.

- Outputs are review signals, not determinations.
- Human reviewers should retain authority over final decisions.
- Local validation is required before any real-world deployment.
- Anti-trafficking data are sensitive and require access control, privacy safeguards, audit logging, and careful governance.
- Protected or sensitive attributes should be restricted, audited, or used only when ethically and operationally justified.
- The CTDC-informed prototype is not field validation and should not be represented as proof of real-world deployment performance.

See [docs/responsible_use.md](docs/responsible_use.md) for more detail.

## CTDC Synthetic Data Acknowledgment

The paper prototype is informed by the CTDC Global Synthetic Dataset v2026. This repository does not include real CTDC records or real operational records. The runnable example generates synthetic CTDC-style records for research and prototype demonstration.

## Citation

Citation placeholder:

> Cao, X. (submitted). *From Detection to Frontline Decision Support: A Configurable AI Framework for Anti-Trafficking Triage and Resource Allocation*.

## License

See [LICENSE](LICENSE).
