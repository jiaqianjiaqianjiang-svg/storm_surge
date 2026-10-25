"""Checks for the read-only stage-1 event analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.xiamen_forecast.analyze_typhoon_stage1 import (
    DEFAULT_SOURCE,
    EVENTS,
    high_frequency_diagnostics,
    lead_band_summary,
    lead_summary,
    load_case,
    recompute_case,
)


@pytest.mark.parametrize("event_id", tuple(EVENTS))
def test_compact_metrics_reproduce_original_results(event_id: str) -> None:
    if not (DEFAULT_SOURCE / event_id / "forecasts_72h.csv").is_file():
        pytest.skip("Compact results not present in this checkout")
    case = load_case(DEFAULT_SOURCE, event_id)
    windows, errors, exact, coverage = recompute_case(*case)
    assert len(coverage) == 3
    assert set(exact.n_origins) == {3}
    assert set(lead_summary(errors).n_origins) == {3}
    assert set(windows.horizon_hours) == {24, 48, 72}


def test_lead_band_summary_uses_hours_within_each_origin() -> None:
    errors = pd.DataFrame({
        "event_id": ["test"] * 72,
        "origin_offset_hours": [-72] * 72,
        "model": ["cnn_gru"] * 72,
        "lead_hours": np.arange(1, 73),
        "error_cm": np.r_[np.ones(12), np.ones(12) * 2, np.ones(24) * 3, np.ones(24) * 4],
        "absolute_error_cm": np.r_[np.ones(12), np.ones(12) * 2, np.ones(24) * 3, np.ones(24) * 4],
    })
    result = lead_band_summary(errors)
    assert result.n_hours.tolist() == [12, 12, 24, 24]
    np.testing.assert_allclose(result.rmse_cm, [1, 2, 3, 4])


def test_constant_persistence_has_undefined_autocorrelation() -> None:
    forecasts = pd.DataFrame({
        "event_id": ["test"] * 4,
        "model": ["persistence"] * 4,
        "origin_offset_hours": [-72] * 4,
        "lead_hours": [1, 2, 3, 4],
        "valid_time_utc": pd.date_range("1997-01-01", periods=4, freq="h").astype(str),
        "surge_cm": [5.0] * 4,
    })
    result = high_frequency_diagnostics(forecasts)
    assert result.max_abs_hourly_change_cm.iloc[0] == 0
    assert np.isnan(result.lag1_autocorrelation.iloc[0])


def test_stage1_entry_rejects_existing_output(tmp_path: Path) -> None:
    from src.xiamen_forecast.analyze_typhoon_stage1 import main

    with pytest.raises(SystemExit, match="2"):
        main(["--output-dir", str(tmp_path)])
