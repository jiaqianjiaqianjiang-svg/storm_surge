import numpy as np
import pandas as pd

from caribbean_short_term_forecast.src.dataset_builder import (
    build_datasets,
    build_year_datasets,
    valid_targets,
)


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


def test_year_split_keeps_test_targets_independent() -> None:
    times = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(f"{year}-01-01", periods=48, freq="h").values
                for year in (2011, 2012, 2013, 2014)
            ]
        )
    )
    atmosphere = np.random.default_rng(11).normal(
        size=(len(times), 3, 2, 2)
    ).astype("float32")
    surge = np.linspace(-0.2, 0.2, len(times), dtype="float32")
    train, validation, test, report = build_year_datasets(
        atmosphere, surge, times, input_steps=4,
        train_start_year=2011, train_end_year=2012,
        validation_year=2013, test_year=2014,
    )
    assert set(train.times[train.targets].year) == {2011, 2012}
    assert set(validation.times[validation.targets].year) == {2013}
    assert set(test.times[test.targets].year) == {2014}
    assert report["test_samples"] == len(test)


def test_ablation_dataset_skips_unused_large_input() -> None:
    times = pd.date_range("2011-01-01", periods=12, freq="h")
    atmosphere = np.ones((12, 3, 2, 2), dtype="float32")
    surge = np.linspace(-0.1, 0.1, 12, dtype="float32")
    train, _, _ = build_datasets(
        atmosphere, surge, times, input_steps=4, train_ratio=0.8,
        model_type="surge_mlp",
    )
    weather, history, _ = train[0]
    assert weather.numel() == 0
    assert tuple(history.shape) == (4,)
