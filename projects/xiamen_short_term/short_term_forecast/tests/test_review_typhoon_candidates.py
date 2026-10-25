"""只测试候选审查的计算与入口，不产生真实事件结论。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.xiamen_forecast import review_typhoon_candidates as review
from src.xiamen_forecast.review_typhoon_candidates import (
    oscillation_summary,
    overlap_hours,
    weather_table,
    wind_from_degrees,
)
from src.xiamen_forecast.run_typhoon_events import MAIN_MODELS, summarise_exact_leads


def test_overlapping_windows_and_wind_direction():
    assert overlap_hours(
        pd.Timestamp("1997-08-26 07:00"), pd.Timestamp("1997-08-31 07:00"),
        pd.Timestamp("1997-08-27 04:00"), pd.Timestamp("1997-09-01 04:00"),
    ) == 99
    assert wind_from_degrees(0, -1) == pytest.approx(0)
    assert wind_from_degrees(-1, 0) == pytest.approx(90)


def test_oscillation_summary_is_descriptive():
    hour = np.arange(120)
    surge_cm = 3 * np.sin(2 * np.pi * hour / 12) + 0.01 * hour
    acf, spectrum, summary = oscillation_summary(surge_cm)
    assert len(acf) == 36
    assert spectrum.relative_power.sum() == pytest.approx(1)
    assert summary["largest_2_to_36h_period_hours"] == pytest.approx(12)
    assert summary["acf_lag12"] > 0.9


def test_weather_table_checks_source_and_units():
    times = pd.date_range("1997-08-01", periods=2, freq="h")
    values = np.zeros((2, 3, 40, 40), dtype=np.float32)
    values[:, 0] = -3
    values[:, 1] = 4
    values[:, 2] = 100000
    raw = xr.DataArray(values, dims=("time", "variable", "latitude", "longitude"),
                       coords={"time": times, "variable": ["U10", "V10", "MSL"]})
    table = weather_table(raw, values.copy(), times, np.array([0.5, 0.6]))
    assert table.observed_surge_cm.tolist() == pytest.approx([50, 60])
    assert table.center_wind_speed_mps.tolist() == pytest.approx([5, 5])
    assert table.center_msl_hpa.tolist() == pytest.approx([1000, 1000])
    altered = values.copy()
    altered[0, 0, 0, 0] = 8
    with pytest.raises(ValueError, match="do not match"):
        weather_table(raw, altered, times, np.array([0.5, 0.6]))


def test_frozen_main_models_keep_other_losses_in_full_metrics():
    assert MAIN_MODELS == ("cnn_gru", "cnn_gru_rollout6", "wmse_wmse")
    metrics = pd.DataFrame({
        "model": ["persistence", "ridge", "cnn_gru", "cnn_gru_rollout6", "wmse_wmse", "awmse_awmse"],
        "horizon_hours": [72] * 6,
        "endpoint_error_cm": [1, 2, 3, 4, 5, 6],
    })
    assert set(summarise_exact_leads(metrics).model) == set(metrics.model)


def test_review_cli_keeps_candidate_pending_and_refuses_overwrite(tmp_path, monkeypatch):
    t0 = pd.Timestamp("1997-08-04 06:00:00")
    times = pd.date_range(t0 - pd.Timedelta(hours=96), periods=192, freq="h")
    atmosphere = np.zeros((len(times), 3, 40, 40), dtype=np.float32)
    atmosphere[:, 0] = -3
    atmosphere[:, 1] = 4
    atmosphere[:, 2] = 100000
    surge = np.linspace(0, 0.5, len(times), dtype=np.float32)
    window = pd.date_range(t0 - pd.Timedelta(hours=72), periods=120, freq="h")
    positions = times.get_indexer(window)
    raw = xr.DataArray(
        atmosphere[positions], dims=("time", "variable", "latitude", "longitude"),
        coords={"time": window, "variable": ["U10", "V10", "MSL"],
                "latitude": np.linspace(30, 39, 40), "longitude": np.linspace(115, 124, 40)},
    )
    candidate_dir = tmp_path / "candidate_package"
    candidate_dir.mkdir()
    source = tmp_path / "ibtracs.csv"
    source.write_text("synthetic source", encoding="utf-8")
    (candidate_dir / "source_and_protocol.json").write_text(
        json.dumps({"best_track_sha256": review.sha256_file(source)}), encoding="utf-8",
    )
    pd.DataFrame([{
        "event_id": "lianyungang_TEST", "station": "lianyungang", "storm_name": "VICTOR",
        "storm_sid": "TEST", "closest_time_utc": t0.isoformat(),
        "window_start_utc": window[0].isoformat(),
        "window_end_exclusive_utc": (window[-1] + pd.Timedelta(hours=1)).isoformat(),
        "closest_distance_km": 335.65,
        "max_observed_surge_cm": float(surge[positions].max() * 100),
    }]).to_csv(candidate_dir / "candidate_events.csv", index=False)
    station = SimpleNamespace(station_id="lianyungang", name="Lianyungang", latitude=34.75,
                              longitude=119.417, grid_size=40, region_size_degrees=10,
                              dataset_dir=tmp_path, era5_files={"U10": tmp_path / "u.nc"})
    monkeypatch.setattr(review, "get_station_config", lambda name: station)
    monkeypatch.setattr(review, "validate_dataset_identity", lambda *args: {
        "array_schema": {"atmosphere_variable_order": ["U10", "V10", "MSL"]},
    })
    monkeypatch.setattr(review, "load_prepared", lambda *args: (atmosphere, surge, times))
    monkeypatch.setattr(review, "load_era5_files", lambda *args, **kwargs: raw)
    monkeypatch.setattr(review, "read_ibtracs_wp", lambda *args: pd.DataFrame({
        "SID": ["TEST", "TEST"], "LAT": [34.5, 35.0], "LON": [119.0, 119.6],
        "time": [t0 - pd.Timedelta(hours=6), t0],
    }))
    output = tmp_path / "review"
    arguments = ["--track-csv", str(source), "--candidate-csv", str(candidate_dir / "candidate_events.csv"),
                 "--output-dir", str(output), "--event-ids", "lianyungang_TEST"]
    review.main(arguments)
    result = pd.read_csv(output / "candidate_review_summary.csv")
    assert result.loc[0, "review_decision"] == "pending"
    assert (output / "lianyungang_TEST" / "surge_weather_review.png").stat().st_size > 1000
    assert (output / "lianyungang_TEST" / "regional_wind_review.png").stat().st_size > 1000
    assert (output / "lianyungang_TEST" / "oscillation_review.png").stat().st_size > 1000
    assert not list(tmp_path.glob(".typhoon_review_*"))
    with pytest.raises(SystemExit):
        review.main(arguments)
