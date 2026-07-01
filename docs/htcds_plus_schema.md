# HTCDS+ Field Registry

HTCDS+ is this project's canonical bridge schema for reusable anti-trafficking data
mapping and AI Core interoperability. It uses official HTCDS fields as its foundation
and adds curated analytical fields when the official standard does not provide a
reusable or model-friendly representation.

HTCDS+ is project-defined and HTCDS-aligned. It is not an official revision of the IOM
HTCDS standard.

## Sources Of Truth

- Official fields and picklists:
  `HTCDS_standard/HTCDS Field Standards 2.0.xlsx`
- Project-owned fields, provenance, values, and missing policy:
  `HTCDS_standard/HTCDS+ Extensions.yaml`
- Methods-of-control definitions:
  `HTCDS_standard/Methods Of Control Definitions.md`

The runtime loader combines the official workbook and extension registry without
modifying official HTCDS definitions.

## Concept Categories

| Category | Meaning |
|---|---|
| `official_htcds` | Field retained from the official cleaned HTCDS list. |
| `derived_from_htcds` | Analytical field derived directly from an official HTCDS concept. |
| `curated_htcds_plus_group` | Project grouping of multiple related HTCDS concepts. |
| `curated_from_ctdc` | Reusable concept first promoted from CTDC. |
| Dataset raw field | Source-only name that remains inside a dataset adapter. |
| Model feature | Task-specific encoded or derived input recorded in a model specification. |

## Official HTCDS Foundation

| Standard name | Type | Values or format | Source |
|---|---|---|---|
| `FirstName` | Text | Free text | Official HTCDS |
| `MiddleName` | Text | Free text | Official HTCDS |
| `LastName` | Text | Free text | Official HTCDS |
| `Gender` | Picklist | Feminine; Masculine; TransgenderFeminine; TransgenderMasculine; NonConforming; NonSpecifiedUnknown | Official HTCDS |
| `Deceased` | Checkbox | Boolean | Official HTCDS |
| `MaritalStatus` | Picklist | Single; Married; Widowed; Divorced; Separated; RegisteredPartnership | Official HTCDS |
| `NumberOfChildren` | Number | Numeric | Official HTCDS |
| `PrimaryLanguage` | Picklist | ISO 639-2 language code | Official HTCDS |
| `HighestLevelOfEducation` | Picklist | Education levels listed in the workbook | Official HTCDS |
| `Citizenship` | Multiselect Picklist | ISO 3166 country code | Official HTCDS |
| `TraffickingStatus` | Picklist | PossibleVictimOfTrafficking; ActualVictimOfTrafficking; NotAVictimOfTrafficking | Official HTCDS |
| `RelationshipToTrafficker` | Picklist | Familial; Acquainted; Unaquainted | Official HTCDS |
| `AgeWhenTrafficked` | Number | Age in years | Official HTCDS |
| `MethodOfRecruitment` | Multiselect Picklist | Recruitment methods listed in the workbook | Official HTCDS |
| `MethodsOfControl` | Multiselect Picklist | Official control values listed in the workbook | Official HTCDS |
| `TypeOfExploitation` | Multiselect Picklist | ForcedLabour; SexualExploitation; Both; Other | Official HTCDS |
| `ForcedLabourIndustry` | Picklist | ISIC-aligned industry values listed in the workbook | Official HTCDS |
| `TypeOfSexExploitation` | Multiselect Picklist | Prostitution; Pornography; RemoteInteractiveServices; PrivateSexualServices | Official HTCDS |
| `Status` | Picklist | Active; FollowUp; Closed; ClosedReferred | Official HTCDS |
| `CaseComment` | Long Text | Free text | Official HTCDS |

## Curated HTCDS+ Fields

| Standard name | Type | Values | Source category | Initial source |
|---|---|---|---|---|
| `case.registration_year` | Number | Numeric year | `curated_from_ctdc` | `yearOfRegistration` |
| `person.age_at_identification_broad` | Picklist | 0--8; 9--17; 18--20; 21--23; 24--26; 27--29; 30--38; 39--47; 48+ | `curated_from_ctdc` | `ageBroad` |
| `exploitation.country` | Picklist | ISO 3166-1 alpha-3 | `curated_from_ctdc` | `CountryOfExploitation` |
| `trafficking.duration_broad` | Picklist | 0--12; 13--24; 25+ months | `curated_from_ctdc` | `traffickMonths` |
| `control.debt_bondage_or_takes_earnings` | Binary | 0; 1; NA | `curated_from_ctdc` | `meansDebtBondageEarnings` |
| `control.threats` | Binary | 0; 1; NA | `curated_htcds_plus_group` | HTCDS threat fields; `meansThreats` |
| `control.abuse` | Binary | 0; 1; NA | `curated_htcds_plus_group` | Physical, psychological, or sexual abuse; `meansAbusePsyPhySex` |
| `control.false_promises` | Binary | 0; 1; NA | `derived_from_htcds` | `FalsePromises`; `meansFalsePromises` |
| `control.psychoactive_substances` | Binary | 0; 1; NA | `derived_from_htcds` | `PsychoactiveSubstance`; `meansDrugsAlcohol` |
| `control.restricted_needs` | Binary | 0; 1; NA | `curated_from_ctdc` | `meansDenyBasicNeeds` |
| `control.excessive_working_hours` | Binary | 0; 1; NA | `derived_from_htcds` | `ExcessiveWorkingHours`; `meansExcessiveWorkHours` |
| `control.withholds_documents` | Binary | 0; 1; NA | `derived_from_htcds` | `WithholdsDocuments`; `meansWithholdDocs` |
| `recruiter.intimate_partner` | Binary | 0; 1; NA | `curated_from_ctdc` | `recruiterRelationIntimatePartner` |
| `recruiter.friend` | Binary | 0; 1; NA | `curated_from_ctdc` | `recruiterRelationFriend` |
| `recruiter.family` | Binary | 0; 1; NA | `curated_from_ctdc` | `recruiterRelationFamily` |
| `recruiter.other` | Binary | 0; 1; NA | `curated_from_ctdc` | `recruiterRelationOther` |

## Grouped Abuse Concept

HTCDS+ uses `control.abuse` as the model-friendly grouped representation of:

- physical abuse;
- psychological abuse; and
- sexual abuse used as a method of control.

When a source provides separate values, `control.abuse = 1` if any is explicitly
positive. When CTDC provides `meansAbusePsyPhySex`, its value maps directly to the
grouped field. HTCDS+ does not claim which subtype occurred when the source is
composite.

## Missing-Value Semantics

The initial HTCDS+ policy groups not collected, not applicable, unknown, and not
reported as `NA`.

- Binary `0` means explicitly negative.
- Binary `1` means explicitly positive.
- Binary `NA` means unknown.
- Categorical unknown values remain `NA`.
- Numeric unknown values remain `NA` in the canonical dataset.
- A model handles numeric values according to its versioned specification. XGBoost
  currently uses native `NaN`; logistic regression uses training-set mean imputation
  with missingness indicators for numeric inputs.
- Binary values remain `NA` through mapping. Logistic regression treats them as a
  nominal `Unknown` state alongside `No` and `Yes`; XGBoost receives `NaN` directly.

Canonical mapping never replaces an unknown binary value with zero.

## Versioning

Adding, renaming, regrouping, or changing the meaning of a field requires an HTCDS+
schema-version change. Dataset adapters and model artifacts must record the HTCDS+
version they support.
