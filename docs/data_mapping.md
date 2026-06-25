# Data Mapping

The Operational Data Mapping Layer translates heterogeneous local records into
documented standardized concepts that multiple AI Core artifacts and decision
configurations can consume.

## Supported Input Concepts

Example local sources include hotline logs, intake forms, referral records, outreach notes, case notes, service records, and public-safety records.

## Standard Indicator Groups

The first implementation maps the following indicators:

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

These indicators can be selected and aggregated differently by each scenario. The
mapping layer should expose standardized indicators and mapping metadata; the
Configurable Decision Layer should decide how they contribute to a particular
operational score.

The CTDC exploitation-type module also uses age, gender, means-of-control, and
recruiter-relationship fields according to its versioned feature contract. Future
artifacts may require different standardized concepts.

## Mapping Documentation

Every local deployment should document:

- Source fields used.
- Local-to-standard concept mappings.
- Missingness assumptions.
- Fields excluded from modeling or scoring.
- Protected fields and how they are restricted.
- Data-quality checks and known limitations.
- Mapping version and compatible artifact versions.
- Mapping coverage, confidence, and unresolved local fields.
