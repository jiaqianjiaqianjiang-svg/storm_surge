import numpy as np
import pandas as pd

from caribbean_short_term_forecast.src.dataset_builder import build_datasets, valid_targets


def test_scheme_b_shapes_and_temporal_split() -> None:
    times = pd.date_range("2011-01-01", periods=36, freq="h")
    atmosphere = np.random.default_rng(7).normal(size=(36, 3, 40, 40)).astype("float32")
    surge = np.linspace(-0.2, 0.3, 36, dtype="float32")
    train, validation, report = build_datasets(atmosphere, surge, times, input_steps=24, train_ratio=0.8)
    weather, history, label = train[0]
    assert tuple(weather.shape) == (72, 40, 40)
    assert tuple(history.shape) == (24,)
    assert label.ndim == 0
    assert train.targets[-1] < validation.targets[0]
    assert report["valid_samples"] == 12


def test_missing_hour_skips_crossing_windows() -> None:
    times = pd.date_range("2011-01-01", periods=30, freq="h").delete(10)
    atmosphere = np.ones((29, 3, 2, 2), dtype="float32")
    surge = np.ones(29, dtype="float32")
    targets, skipped = valid_targets(times, atmosphere, surge, input_steps=4)
    assert skipped["non_contiguous"] > 0
    assert targets
