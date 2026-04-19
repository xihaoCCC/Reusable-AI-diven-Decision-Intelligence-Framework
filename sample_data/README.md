# Sample Data

This folder contains synthetic, public-safe data for quick demos and basic tests.

## Files

- `track_a_synthetic_cases.csv`: 200 synthetic Track A case records with 180 non-victim examples and 20 victim-labeled examples.
- `track_b_synthetic_demand.csv`: 3 synthetic Track B daily demand series spanning January 1, 2026 through March 31, 2026.

## Important Notes

- These records are fully synthetic and should not be interpreted as real operational data.
- Track A labels are included only to support model smoke tests and workflow demos.
- Track B demand values are intentionally scaled differently by department so forecasting models can be tested across small, medium, and large demand magnitudes.
