# Triage Configuration

Scenario configuration translates reusable AI Core outputs into visible,
organization-specific triage logic. The included YAML files reproduce the two
demonstration scenarios from the publication notebook; they are not universal
operational policies.

## Small NGO Multidisciplinary Scenario

`configs/scenario_small_ngo_multidisciplinary.yaml` uses:

- `target_probability: P(Both)`
- `confidence_threshold: 0.70`
- `review_capacity_k: 30`
- route label: `Multidisciplinary review`
- confidence weight: `0.30`
- target-probability weight: `0.35`
- control weight: `0.20`
- vulnerability weight: `0.05`
- relationship weight: `0.10`

Its indicator focus includes threats, abuse, denial of basic needs, drugs/alcohol
control, false promises, minor/young indicators, and close recruiter relationships.

## Labor Task Force Scenario

`configs/scenario_labor_task_force.yaml` uses:

- `target_probability: max(P(Labor), P(Both))`
- `confidence_threshold: 0.60`
- `review_capacity_k: 30`
- route label: `Labor-focused review or labor-plus-multidisciplinary review`
- confidence weight: `0.15`
- target-probability weight: `0.40`
- control weight: `0.30`
- vulnerability weight: `0.10`
- relationship weight: `0.05`

Its indicator focus includes debt bondage, excessive work hours, withheld documents,
denial of basic needs, threats, and recruiter relationships.

## Priority Score

```text
Priority_i,s = 100 * (
  w_conf * Conf_i
  + w_target * P_target,i,s
  + w_control * Control_i,s
  + w_vulnerability * Vulnerability_i,s
  + w_relationship * Relationship_i,s
)
```

Weights must sum to one, and every component must be scaled to `[0, 1]`.

The scoring function is a reusable framework component. Target-probability
expressions, indicator groups, aggregation rules, weights, confidence thresholds,
review capacity, and routing remain configurable.

## Current Implementation Status

The current demonstration computes shared control, vulnerability, and relationship
summary fields before applying a scenario. A planned revision will compute these
components from each scenario's configured feature groups, matching the publication
notebook more exactly and making component contributions auditable.

Real deployments should establish weights, indicators, thresholds, and routing through
domain expertise, operational policy, ethical review, and local validation.
