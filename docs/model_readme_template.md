# Model Name And Version

## Task

Describe the prediction task, output classes, intended users, and HTCDS+ schema
version.

## Training Data

Document source datasets, date/version, filters, fingerprints, population, and known
limitations without redistributing restricted data.

## Feature Contract

List all model features, types, transformations, and missing-value policies.

## Core Features

Document:

- final Core feature list;
- top-K value;
- importance method or methods;
- cross-validation or resampling stability;
- leakage and sensitive-feature review; and
- domain-review rationale.

Local inference data must cover every Core feature.

## Non-Core Missing Features

Describe permitted imputation strategies and the compatibility threshold. Unknown
binary and categorical values remain `NA`; zero is never a substitute for unknown.

## Evaluation

Report predictive, calibration, subgroup, robustness, and downstream triage metrics.

## Intended And Prohibited Uses

State appropriate decision-support uses, human-review requirements, and prohibited
automated decisions.
