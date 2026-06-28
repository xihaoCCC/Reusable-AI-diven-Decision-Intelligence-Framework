# Data Mapping

The Operational Data Mapping Layer translates heterogeneous source records into
HTCDS+, the project's canonical bridge schema for AI Core interoperability.

## Separation Of Responsibilities

- **Official HTCDS fields** retain their official names and definitions.
- **HTCDS+ fields** are curated reusable analytical concepts with provenance.
- **Dataset raw fields** remain inside adapters such as the CTDC mapper.
- **Model feature lists** select and transform a task-specific subset.
- **Inference-time available fields** determine model compatibility.

See [htcds_plus_schema.md](htcds_plus_schema.md) for the central field registry.

## Schema Loaders

`load_htcds_schema` reads the cleaned Excel workbook. `load_htcds_plus_schema` combines
that base with `HTCDS+ Extensions.yaml`. The combined schema validates official and
project-owned field names, types, allowed values, provenance, and missing-value policy.

## CTDC Adapter

`CTDCHTCDSPlusMapper` maps CTDC records into official HTCDS and curated HTCDS+ fields.
The coverage report still identifies covered, partial, and extension source mappings,
but reusable output names no longer carry CTDC prefixes.

See [ctdc_htcds_mapping.md](ctdc_htcds_mapping.md) for the source comparison.

## Missing Values

Canonical mapping preserves unknown values as `NA`:

- binary `0` is reserved for an explicit negative;
- binary `1` is an explicit positive;
- categorical and numeric unknowns remain `NA`.

Imputation belongs to a versioned model feature contract, not the mapping layer. The
current classification configuration uses mean numeric imputation, most-frequent binary
imputation, and missingness indicators. These are demonstration defaults that must be
validated for the released artifact.

## Model Feature Selection

`configs/ctdc_exploitation_type_features.yaml` defines the current task's selected
features, imputation policy, and Core-feature policy separately from HTCDS+.

Core features remain pending until the improved model is trained and feature importance
is reviewed. Once defined, inference data must cover every Core feature; missing
non-Core features may follow the artifact's documented imputation policy.

## Future Dataset Support

Each new dataset adds an adapter from raw fields to HTCDS+. A new concept is promoted
only when it is useful, reusable, semantically stable, typed, documented, and not
already represented. Dataset quirks that do not meet those criteria remain inside the
adapter rather than expanding the canonical schema.
