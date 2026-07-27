import numpy as np

from caribbean_short_term_forecast.src.evaluate_baselines import (
    build_ridge_features,
    summarise_atmosphere,
)


def test_compact_ridge_feature_shape() -> None:
    atmosphere = np.arange(10 * 3 * 2 * 2, dtype=np.float32).reshape(10, 3, 2, 2)
    surge = np.linspace(-0.1, 0.1, 10, dtype=np.float32)
    summaries = summarise_atmosphere(atmosphere, chunk_hours=3)
    features = build_ridge_features(summaries, surge, [4, 5], input_steps=4)
    assert summaries.shape == (10, 3, 4)
    assert features.shape == (2, 4 * 3 * 4 + 4)
    assert np.allclose(features[0, -4:], surge[:4])
