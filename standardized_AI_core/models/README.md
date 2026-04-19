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
- `TreeBasedDemandForecaster`
- `ARIMAForecaster`
- `SimpleMLPForecaster`

The first two are lightweight demand-forecasting baselines for early experimentation and benchmarking. `TreeBasedDemandForecaster` adds a nonlinear Random Forest model with lag, rolling-average, and calendar features. `ARIMAForecaster` provides a classical statistical baseline. `SimpleMLPForecaster` lives in a separate script because it depends on PyTorch and is more complex than the other starter models.

Optional dependencies:
- `ARIMAForecaster` requires `statsmodels`.
- `SimpleMLPForecaster` requires `torch`.
