"""历史台风工具的小型合成测试；不代表真实台风或实验结果。"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from src.xiamen_forecast import run_typhoon_events as workflow
from src.xiamen_forecast.run_typhoon_events import (
    _plot_errors,
    _plot_process,
    _plot_track,
    run,
    scan,
    summarise_exact_leads,
)
from src.xiamen_forecast.station_config import get_station_config
from src.xiamen_forecast.typhoon_events import (
    build_candidates,
    inspect_event_coverage,
    read_ibtracs_wp,
    score_forecast_window,
)


def _csv(path):
    path.write_text(
        "SID,SEASON,NAME,ISO_TIME,LAT,LON,BASIN,WMO_WIND,WMO_PRES,TRACK_TYPE\n"
        ",,,UTC,degrees_north,degrees_east,,kt,mb,\n"
        "1997001,1997,TEST,1997-08-01 12:30:00,34.7,119.4,WP,60,990,MAIN\n"
        "1997001,1997,TEST,1997-08-01 18:30:00,34.8,119.5,WP,65,985,MAIN\n"
        "1997001,1997,TEST,1997-08-01 15:30:00,40,130,WP,65,985,spur\n"
        "1997002,1997,OTHER,1997-08-01 12:00:00,0,150,WP,60,990,MAIN\n"
        "1997003,1997,ATL,1997-08-01 12:00:00,34.7,119.4,NA,60,990,MAIN\n",
        encoding="utf-8",
    )


def test_best_track_screen_and_hour_alignment(tmp_path):
    source = tmp_path / "ibtracs.csv"
    _csv(source)
    tracks = read_ibtracs_wp(source)
    assert set(tracks.SID.astype(str)) == {"1997001", "1997002"}
    assert len(tracks.loc[tracks.SID == 1997001]) == 2
    cases = build_candidates(tracks, [get_station_config("lianyungang")])
    assert len(cases) == 1
    assert cases.iloc[0].storm_sid == "1997001"
    assert pd.Timestamp(cases.iloc[0].closest_time_utc).minute == 0
    assert pd.Timestamp(cases.iloc[0].nearest_track_time_utc).minute == 30
    assert cases.iloc[0].review_decision == "pending"


def test_coverage_and_peak_metrics():
    times = pd.date_range("1997-08-01", periods=240, freq="h")
    surge = np.full(240, 0.1, dtype=np.float32)
    atmosphere = np.zeros((240, 3, 2, 2), dtype=np.float32)
    t0 = pd.Timestamp("1997-08-06 00:00")
    coverage = inspect_event_coverage(times, surge, atmosphere, t0, 1997)
    assert coverage["data_complete"]
    assert len(coverage["valid_origin_positions"]) == 3
    surge[coverage["valid_origin_positions"][0] + 3] = np.nan
    broken = inspect_event_coverage(times, surge, atmosphere, t0, 1997)
    assert not broken["data_complete"]
    assert broken["valid_origin_count"] < 3

    observed = np.zeros(72)
    predicted = np.zeros(72)
    observed[4] = 0.5
    predicted[5] = 0.4
    result = score_forecast_window(observed, predicted, t0, 24)
    assert result["peak_height_error_cm"] == pytest.approx(-10)
    assert result["peak_time_error_hours"] == 1
    assert result["peak_underestimated"] is True
    assert result["peak_underestimation_cm"] == pytest.approx(10)
    no_positive = score_forecast_window(np.full(72, -0.1), predicted, t0, 72)
    assert np.isnan(no_positive["observed_peak_cm"])
    assert no_positive["peak_underestimated"] is None


def test_exact_lead_summary_is_not_cumulative_rmse():
    source = pd.DataFrame({
        "model": ["cnn_gru", "cnn_gru", "cnn_gru"],
        "horizon_hours": [24, 24, 72],
        "endpoint_error_cm": [3.0, 4.0, 10.0],
    })
    summary = summarise_exact_leads(source)
    row = summary.loc[summary.lead_hours == 24].iloc[0]
    assert row.n_origins == 2
    assert row.lead_rmse_cm == pytest.approx(np.sqrt(12.5))
    assert row.lead_mae_cm == pytest.approx(3.5)


def test_scan_stays_pending_and_run_refuses_unreviewed(tmp_path):
    source = tmp_path / "ibtracs.csv"
    _csv(source)
    output = tmp_path / "package"
    scan(argparse.Namespace(
        track_csv=source, results_dir=output, year=1997, seed=42,
        stations=["lianyungang"], max_distance_km=500,
    ))
    candidates = pd.read_csv(output / "candidate_events.csv")
    assert len(candidates) == 1
    assert candidates.iloc[0].review_decision == "pending"
    assert not (output / "all_event_metrics.csv").exists()
    assert json.loads((output / "source_and_protocol.json").read_text())["best_track_sha256"]
    with pytest.raises(ValueError, match="No manually confirmed"):
        run(argparse.Namespace(
            track_csv=source, results_dir=output, year=1997, seed=42,
            stations=["lianyungang"], device="cpu",
        ))


def test_figure_helpers_write_png(tmp_path):
    track = pd.DataFrame({
        "time": pd.date_range("1997-08-01", periods=4, freq="6h"),
        "LAT": [34.0, 34.5, 34.8, 35.0],
        "LON": [119.0, 119.2, 119.5, 120.0],
    })
    _plot_track(track, "lianyungang", pd.Timestamp("1997-08-01 12:00"), tmp_path / "track.png")
    forecasts = []
    metrics = []
    for offset in (-72, -48, -24):
        for model in ("observed", "cnn_gru", "cnn_gru_rollout6", "wmse_mse", "awmse_awmse"):
            for lead in range(72):
                forecasts.append({
                    "origin_offset_hours": offset,
                    "model": model,
                    "valid_time_utc": (pd.Timestamp("1997-08-01") + pd.Timedelta(hours=lead)).isoformat(),
                    "surge_cm": float(lead) / 10,
                })
            if model != "observed":
                for horizon in (24, 48, 72):
                    metrics.append({
                        "model": model, "horizon_hours": horizon,
                        "rmse_cm": 2.0, "absolute_peak_height_error_cm": 1.0,
                        "peak_time_error_hours": -1.0, "endpoint_error_cm": 2.0,
                    })
    _plot_process(pd.DataFrame(forecasts), tmp_path / "process.png")
    _plot_errors(pd.DataFrame(metrics), summarise_exact_leads(pd.DataFrame(metrics)), tmp_path / "errors.png")
    assert all((tmp_path / name).stat().st_size > 1000 for name in ("track.png", "process.png", "errors.png"))


def test_confirmed_case_exports_common_origins(tmp_path, monkeypatch):
    times = pd.date_range("1997-08-01", periods=240, freq="h")
    atmosphere = np.zeros((240, 3, 2, 2), dtype=np.float32)
    surge = np.full(240, 0.1, dtype=np.float32)
    surge[120:130] = np.linspace(0.1, 0.6, 10)
    monkeypatch.setattr(workflow, "read_station_year", lambda station, year: (atmosphere, surge, times))
    files = [tmp_path / name for name in ("ridge.joblib", "cnn.pth", "rollout.pth")]
    for file in files:
        file.write_bytes(b"synthetic-test-only")
    monkeypatch.setattr(workflow, "missing_model_files", lambda station, seed: [])
    monkeypatch.setattr(workflow, "model_files", lambda station, seed: (*files, {}))
    monkeypatch.setattr(workflow, "load_models", lambda *args: (
        {"cnn_gru": object(), "cnn_gru_rollout6": object()},
        {"cnn_gru": {"scalers": {}}},
        {"cnn_gru": files[1], "cnn_gru_rollout6": files[2]},
    ))
    monkeypatch.setattr(workflow.joblib, "load", lambda path: object())
    monkeypatch.setattr(workflow, "summarise_atmosphere", lambda atmosphere: np.zeros((240, 1, 1)))
    calls = []

    def fake_recursive(atmosphere, surge, origins, summaries, ridge, models, scalers, input_steps, max_lead, batch_size, device):
        calls.append(origins.copy())
        assert len(origins) == 3 and input_steps == 24 and max_lead == 72
        assert list(models) == ["cnn_gru", "cnn_gru_rollout6"]
        return {
            "persistence": np.full((3, 72), 0.1),
            "ridge": np.full((3, 72), 0.2),
            "cnn_gru": np.full((3, 72), 0.3),
            "cnn_gru_rollout6": np.full((3, 72), 0.4),
        }

    monkeypatch.setattr(workflow, "recursive_predictions", fake_recursive)
    track = pd.DataFrame({
        "time": pd.date_range("1997-08-05", periods=4, freq="6h"),
        "LAT": [24.0, 24.2, 24.4, 24.6],
        "LON": [117.8, 118.0, 118.2, 118.4],
    })
    output = tmp_path / "results"
    output.mkdir()
    row = pd.Series({
        "event_id": "xiamen_TEST", "station": "xiamen", "storm_sid": "TEST",
        "storm_name": "TEST", "closest_time_utc": times[120].isoformat(),
        "nearest_track_time_utc": times[120].isoformat(),
        "closest_distance_km": 22.0, "review_note": "Synthetic unit test only",
    })
    metrics = workflow.run_one(row, track, argparse.Namespace(year=1997, seed=42, device="cpu"), output)
    assert len(calls) == 1
    assert len(metrics) == 3 * 4 * 3
    event = output / "xiamen_TEST"
    assert len(pd.read_csv(event / "forecasts_72h.csv")) == 3 * 5 * 72
    assert len(pd.read_csv(event / "exact_lead_metrics.csv")) == 4 * 3
    assert (event / "研究报告.md").is_file()
    assert all((event / f"figure{index}_{name}.png").stat().st_size > 1000 for index, name in (
        (1, "best_track"), (2, "fixed_origin_forecasts"), (3, "event_errors"),
    ))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        workflow.run_one(row, track, argparse.Namespace(year=1997, seed=42, device="cpu"), output)
    assert not list(output.glob(".typhoon_event_*"))
