import numpy as np
import pandas as pd

from src.direct_multistep_baselines import (
    build_feature_matrix,
    direct_labels,
    direct_origins,
    predict_selected_models,
    select_alphas_and_fit,
)


def test_direct_origins_require_complete_24_targets_without_year_crossing():
    times = pd.date_range("2016-12-29", "2017-01-05", freq="h")
    atmosphere_valid = np.ones(len(times), dtype=bool)
    surge = np.arange(len(times), dtype=np.float32)
    origins = direct_origins(times, atmosphere_valid, surge, {2016, 2017}, 24, 24)
    assert len(origins)
    assert all(times[o].year == times[o + 24].year for o in origins)
    surge[origins[-1] + 10] = np.nan
    filtered = direct_origins(times, atmosphere_valid, surge, {2016, 2017}, 24, 24)
    assert origins[-1] not in filtered


def test_direct_labels_and_feature_boundaries():
    summaries = np.arange(80 * 3 * 4, dtype=np.float32).reshape(80, 3, 4)
    surge = np.arange(80, dtype=np.float32)
    origins = np.asarray([30, 31])
    labels = direct_labels(surge, origins, output_steps=3)
    assert labels[0].tolist() == [31.0, 32.0, 33.0]
    features = build_feature_matrix(
        summaries, surge, origins, include_surge=True,
        era5_mode="past_future", input_steps=2, output_steps=3,
    )
    assert features.shape == (2, (2 + 3) * 12 + 2)
    np.testing.assert_array_equal(features[0, -2:], [29.0, 30.0])


def test_alpha_selection_returns_one_choice_and_prediction_per_lead():
    rng = np.random.default_rng(7)
    x_early = rng.normal(size=(50, 4))
    x_tuning = rng.normal(size=(20, 4))
    x_all = np.vstack([x_early, x_tuning])
    weights = rng.normal(size=(4, 3))
    y_early = x_early @ weights
    y_tuning = x_tuning @ weights
    y_all = x_all @ weights
    scaler, models, chosen, tuning_rmse = select_alphas_and_fit(
        x_early, y_early, x_tuning, y_tuning, x_all, y_all,
        alphas=(0.01, 1.0),
    )
    predicted = predict_selected_models(scaler, models, chosen, x_tuning)
    assert chosen.shape == (3,)
    assert tuning_rmse.shape == (3,)
    assert predicted.shape == y_tuning.shape
    assert np.sqrt(np.mean((predicted - y_tuning) ** 2)) < 0.02
