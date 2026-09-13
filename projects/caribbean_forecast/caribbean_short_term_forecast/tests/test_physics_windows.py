import numpy as np
import pandas as pd
import pytest

from caribbean_short_term_forecast.src.physics_features import (
    build_origin_physics_features,
    future_window,
)


NAMES = [
    "pressure_local_pa",
    "wind_stress_east_local_pa",
    "wind_stress_north_local_pa",
    "wind_stress_magnitude_local_pa",
]


def test_six_hour_feature_cannot_see_hours_seven_to_twenty_four():
    times = pd.date_range("2017-01-01", periods=80, freq="h")
    values = np.column_stack([
        101_000 + np.arange(80), np.ones(80), np.ones(80) * 2, np.ones(80) * 3,
    ])
    first, names = build_origin_physics_features(values, NAMES, times, [30], 6)
    changed = values.copy()
    changed[37:55] = 1e9
    second, names_second = build_origin_physics_features(changed, NAMES, times, [30], 6)
    np.testing.assert_array_equal(first, second)
    assert names == names_second
    assert future_window(30, 6) == slice(31, 37)


def test_non_hourly_or_nan_physics_window_is_rejected():
    times = pd.date_range("2017-01-01", periods=60, freq="h").to_series().reset_index(drop=True)
    times.iloc[20] += pd.Timedelta(minutes=30)
    values = np.ones((60, len(NAMES)))
    with pytest.raises(ValueError, match="non-hourly"):
        build_origin_physics_features(values, NAMES, times, [30], 6)
    good_times = pd.date_range("2017-01-01", periods=60, freq="h")
    values[31, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        build_origin_physics_features(values, NAMES, good_times, [30], 6)
