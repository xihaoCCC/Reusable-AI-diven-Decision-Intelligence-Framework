# Shared Models

This package contains simple, reusable starter models that fit the two deployment tracks.

Classification models for Track A:
- `LogisticRegressionVictimClassifier`
- `RandomForestVictimClassifier`
- `GradientBoostingVictimClassifier`

These are intended for structured victim-identification support using demographic and related numeric/categorical features.

Forecasting models for Track B:
- `MovingAverageForecaster`
- `SeasonalNaiveForecaster`

These are lightweight demand-forecasting baselines for early experimentation and benchmarking.
