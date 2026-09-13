import numpy as np
import pandas as pd

from caribbean_short_term_forecast.src.physics_features import derive_hourly_physics
from caribbean_short_term_forecast.src.train_physics_baselines import fit_tuned_ridge
from caribbean_short_term_forecast.src.train_residual_physics_xgb import expanding_oof_ridge


def test_ridge_scaler_is_unchanged_by_2017_values():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(70, 3))
    years = np.repeat(np.arange(2011, 2018), 10)
    target = features[:, 0] * 0.3
    scaler_a, model_a, alpha_a, _ = fit_tuned_ridge(features, target, years, (0.1, 1.0))
    altered = features.copy(); altered[years == 2017] += 1e6
    scaler_b, model_b, alpha_b, _ = fit_tuned_ridge(altered, target, years, (0.1, 1.0))
    np.testing.assert_array_equal(scaler_a.mean_, scaler_b.mean_)
    np.testing.assert_array_equal(model_a.coef_, model_b.coef_)
    assert alpha_a == alpha_b


def test_oof_ridge_respects_72_hour_year_boundary_gap():
    times = pd.date_range("2011-01-01", "2016-12-31 23:00", freq="12h")
    features = np.column_stack([np.arange(len(times)), np.sin(np.arange(len(times)))])
    target = features[:, 0] * 0.001
    prediction, mask, folds = expanding_oof_ridge(features, target, times, gap_hours=72)
    assert mask.any() and np.isfinite(prediction[mask]).all()
    for fold in folds:
        boundary = pd.Timestamp(f"{fold['prediction_year']}-01-01") - pd.Timedelta(hours=72)
        assert pd.Timestamp(fold["train_max_time"]) <= boundary


def test_hourly_feature_names_shape_dtype_and_training_pressure_reference():
    atmosphere = np.zeros((12, 3, 3, 3), dtype=np.float32)
    atmosphere[:, 0] = 5.0
    atmosphere[:, 2] = 101_000.0
    atmosphere[8:, 2] = 90_000.0  # validation-like values must not enter reference
    training = np.zeros(12, dtype=bool); training[:8] = True
    result = derive_hourly_physics(
        atmosphere, [11.0, 12.0, 13.0], [-63.0, -62.0, -61.0],
        12.0, -62.0, training,
        {"spatial_regions": {"station": 0.0}, "onshore_bearing_deg_clockwise_from_north": None},
        chunk_hours=4,
    )
    assert result.values.shape == (12, len(result.names))
    assert result.values.dtype == np.float32
    assert len(result.names) == len(result.units)
    assert result.metadata["pressure_climatology_pa"] == 101_000.0
    assert not result.metadata["onshore_features_available"]
