# Data Mapping

The operational data mapping layer translates heterogeneous local records into standardized features that the AI core and decision layer can consume.

## Supported Input Concepts

Example local sources include hotline logs, intake forms, referral records, outreach notes, case notes, service records, and public-safety records.

## Standard Indicator Groups

The prototype currently maps the following indicators:

- `threats`
- `abuse_indicators`
- `denial_basic_needs`
- `drugs_alcohol_control`
- `false_promises`
- `minor_young_indicator`
- `close_recruiter_relationship`
- `debt_bondage`
- `excessive_work_hours`
- `withheld_documents`

These indicators are then summarized into control, vulnerability, and relationship scores used by the configurable decision layer.

## Mapping Documentation

Every local deployment should document:

- Source fields used.
- Local-to-standard concept mappings.
- Missingness assumptions.
- Fields excluded from modeling or scoring.
- Protected fields and how they are restricted.
- Data-quality checks and known limitations.

