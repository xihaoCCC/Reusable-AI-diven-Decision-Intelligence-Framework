# AI Core Artifacts

The Standardized AI Core is a growing registry of reusable anti-trafficking
data-science components. A component may be a pre-trained model, rule-informed module,
scoring algorithm, ranking function, evaluator, monitor, or explanation utility.

## Artifact Principles

Each released pre-trained model should be:

- task-specific rather than presented as a universal trafficking model;
- versioned and immutable after release;
- reproducible from documented code and configuration;
- packaged with its preprocessing and input schema;
- evaluated for predictive performance and probability quality;
- documented with intended uses, limitations, and prohibited uses;
- auditable for sensitive-feature use;
- loadable through a common inference interface; and
- independent from organization-specific routing and policy decisions.

## Common Interface

A model artifact should support:

```text
load(artifact_reference)
validate_input(records)
predict_proba(records)
predict(records)
explain(records)
metadata()
```

Task modules may extend this contract, but they should not bypass schema validation or
artifact metadata.

## CTDC Exploitation-Type Artifact

The first planned release artifact predicts `Sex`, `Labor`, and `Both` exploitation
types and returns:

- `P(Sex)`
- `P(Labor)`
- `P(Both)`
- predicted exploitation type
- confidence
- explanation-ready feature information

Its bundle should include the primary model, transparent baseline, fitted
preprocessing, label encoding, ordered feature schema, model card, training
configuration, metrics, threshold analysis, provenance, and checksums.

Age and gender are candidate features in this module. The artifact must declare them
explicitly, report ablation and subgroup analyses, and warn downstream users that
sensitive-feature use requires local justification and governance.

## Artifact Versus Configuration

The artifact supplies reusable analytical signals. It does not encode an
organization's final operational policy.

The Configurable Decision Layer owns:

- target-probability selection;
- priority weights;
- review capacity `K`;
- confidence thresholds;
- control, vulnerability, and relationship indicator groups;
- escalation rules; and
- service/referral routing.

This separation allows one artifact to support different missions while keeping local
policy visible and reviewable.

## Promotion From Local Development

Raw data, exploratory notebooks, tuning runs, and candidate models remain in
`Local_runner/`. A candidate is promoted only after it passes the release gates in
[ROADMAP.md](../ROADMAP.md).

