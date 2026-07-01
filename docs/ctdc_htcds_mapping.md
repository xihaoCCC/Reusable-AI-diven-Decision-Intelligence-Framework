# CTDC To HTCDS+ Mapping

The CTDC adapter treats the CTDC Global Synthetic Dataset as a source dataset and maps
its 27 fields into official HTCDS fields and curated HTCDS+ analytical fields.

## Coverage

- 15 covered source fields
- 8 partial source mappings
- 4 source concepts that required HTCDS+ extensions

Coverage describes the relationship between CTDC and official HTCDS. Regardless of
coverage category, reusable analytical outputs now use HTCDS+ names rather than CTDC
prefixes.

## Partial Controls

| CTDC source | HTCDS+ output | Meaning |
|---|---|---|
| `meansDebtBondageEarnings` | `control.debt_bondage_or_takes_earnings` | Source cannot distinguish its two components. |
| `meansThreats` | `control.threats` | Grouped threats concept; source cannot identify the subtype. |
| `meansAbusePsyPhySex` | `control.abuse` | Groups physical, psychological, and sexual abuse. |
| `meansDenyBasicNeeds` | `control.restricted_needs` | Groups finance, movement, medical-care, and necessities restrictions. |

The mapper does not expand a composite positive into several unsupported precise
values.

## Other Partial Mappings

- `sectorOfLabourAgriculture` maps to the broader official
  `A_AgricultureForestryAndFishing` industry value.
- intimate-partner, friend, and other recruiter flags map broadly to official
  `RelationshipToTrafficker = Acquainted`, while their detailed HTCDS+ fields remain
  available for modeling.

## Promoted Source Concepts

| CTDC source | HTCDS+ field | Current role |
|---|---|---|
| `yearOfRegistration` | `case.registration_year` | Filtering and provenance |
| `ageBroad` | `person.age_at_identification_broad` | Selected model input source |
| `CountryOfExploitation` | `exploitation.country` | Filtering and provenance |
| `traffickMonths` | `trafficking.duration_broad` | Available, not currently selected |

Age at identification is deliberately distinct from official `AgeWhenTrafficked`.

## Current Model Contract

The current configuration selects 19 features: one ordinal age feature and 18 binary
gender, age, control, and recruiter indicators. XGBoost receives their `NaN` values
directly. Logistic regression mean-imputes the numeric age input with a missingness
indicator and treats each binary input as a nominal `No`, `Yes`, or `Unknown` state,
using `No` as the dummy-variable reference.

The v3 candidate bundle proposes a top-K list from validation-period permutation
importance. The public feature contract remains `candidate_pending_review` until
stability, leakage, sensitive-feature, and domain review define the final Core list.

## Coverage Command

```bash
python examples/report_ctdc_htcds_coverage.py \
  --ctdc-csv ../Local_runner/sample_data/ctdc_global_synthetic_data_v2026.csv \
  --output outputs/tables/ctdc_htcds_coverage.csv
```

Only the CSV header is read for availability reporting.
