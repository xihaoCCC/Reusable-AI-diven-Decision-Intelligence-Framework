# Triage Configuration

Scenario YAML files define how model outputs become review queues.

## Small NGO Multidisciplinary Scenario

`configs/scenario_small_ngo_multidisciplinary.yaml` uses:

- `target_probability: P(Both)`
- `confidence_threshold: 0.70`
- `review_capacity_k: 30`
- route label: `Multidisciplinary review`

This scenario prioritizes records that may require broader service coordination.

## Labor Task Force Scenario

`configs/scenario_labor_task_force.yaml` uses:

- `target_probability: max(P(Labor), P(Both))`
- `confidence_threshold: 0.60`
- `review_capacity_k: 30`
- route label: `Labor-focused review or labor-plus-multidisciplinary review`

This scenario prioritizes labor or mixed-exploitation signals.

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

Organizations can adjust the weights, target probability, confidence threshold, and review capacity without retraining the AI core.

