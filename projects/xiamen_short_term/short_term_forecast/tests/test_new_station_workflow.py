from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.xiamen_forecast import audit_station_sources as source_audit
from src.xiamen_forecast import run_station_workflow as workflow
from src.xiamen_forecast import station_config
from src.xiamen_forecast.plan_typhoon_holdout import make_exclusions, verify_source_catalog
from src.xiamen_forecast.run_final_test import build_commands
from src.xiamen_forecast.run_validation_workflow import build_validation_steps


def test_annual_coverage_selects_actual_training_start_with_internal_gap() -> None:
    times = pd.date_range("1989-01-01", "1997-12-31 23:00", freq="h", tz="UTC")
    times = times[times.year != 1990]
    coverage = source_audit.annual_coverage(
        times, pd.Timestamp("1989-01-01", tz="UTC"),
        pd.Timestamp("1997-12-31 23:00", tz="UTC"), 1989, 1997,
    )
    assert source_audit.choose_training_start(coverage, 0.8) == 1989
    assert coverage[1]["matched_hours"] == 0
    with_one_missing_training_year = [
        {**row, "matched_fraction": 0.1} if row["year"] == 1993 else row
        for row in coverage
    ]
    assert source_audit.choose_training_start(with_one_missing_training_year, 0.8) == 1989
    with pytest.raises(ValueError, match="1996 validation"):
        source_audit.choose_training_start(coverage[:-1], 0.8)


def test_source_audit_creates_shanwei_config_only_from_checked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    gesla = source / "shanwei-gesla"
    gesla.touch()
    files = {name: source / f"shanwei_{name}.nc" for name in ("U10", "V10", "MSL")}
    for path in files.values():
        path.touch()
    times = pd.date_range("1991-01-01", "1997-12-31 23:00", freq="h", tz="UTC")
    raw = pd.DataFrame({"datetime": times, "water_level": 0.1})
    monkeypatch.setattr(source_audit, "load_tide_gauge", lambda *a, **k: raw)
    monkeypatch.setattr(source_audit, "quality_control", lambda frame: (frame, {"final_record_count": len(frame)}))
    monkeypatch.setattr(source_audit, "inspect_era5_files", lambda paths: {
        "status": "passed", "errors": [], "warnings": [],
        "variables": {
            name: {"file": str(files[name]),
                   "time_range": ["1990-01-01T00:00:00", "1997-12-31T23:00:00"],
                   "latitude_range": [15.0, 30.0], "longitude_range": [110.0, 125.0]}
            for name in files
        },
    })
    config = tmp_path / "outputs" / "station_configs" / "shanwei.json"
    report = source_audit.configure(
        gesla, files, 22.8, 115.3, tmp_path / "outputs" / "source_audit",
        config_path=config,
    )
    assert report["selected_training_years"] == [1991, 1995]
    monkeypatch.setattr(station_config, "SHANWEI_CONFIG", config)
    station = station_config.get_station_config("shanwei")
    assert station.train_start_year == 1991
    assert station.era5_files["U10"] == files["U10"]
    assert station.dataset_dir != station_config.get_station_config("xiamen").dataset_dir
    with pytest.raises(FileExistsError, match="already exists"):
        source_audit.configure(gesla, files, 22.8, 115.3, tmp_path / "outputs" / "source_audit", config_path=config)
    assert len(station_config.STATIONS) == 3


def test_core_profiles_preserve_existing_full_default() -> None:
    full = build_validation_steps("lianyungang")
    core = build_validation_steps("lianyungang", profile="core")
    assert len(full) == 13
    training = [step for step in core if "src.xiamen_forecast.train_xiamen" in step.command]
    assert len(training) == 1
    assert training[0].command[training[0].command.index("--model-type") + 1] == "cnn_gru"
    rolling = next(step for step in core if "src.xiamen_forecast.rolling_diagnostics" in step.command)
    assert rolling.command[rolling.command.index("--models") + 1] == "cnn_gru"
    assert "--validation-only" in training[0].command
    test_commands = build_commands(1997, 42, "cpu", "en", Path("rollout.pth"),
                                   "lianyungang", "core")
    assert "--existing-ridge" in test_commands[0]
    assert test_commands[1][test_commands[1].index("--models") + 1] == "cnn_gru"
    assert "--existing-ridge" not in build_commands(1997, 42, "cpu", "en", Path("rollout.pth"))[0]


def test_validation_lock_detects_changed_weight_and_review_is_new_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(station_config, "MODULE_ROOT", tmp_path / "project")
    monkeypatch.setattr(workflow, "MODULE_ROOT", tmp_path / "project")
    station = station_config.get_station_config("lianyungang")
    station.processed_root.mkdir(parents=True)
    (station.processed_root / "prepared_dataset_audit.json").write_text(
        json.dumps({"status": "passed"}), encoding="utf-8"
    )
    validation = workflow.result_package("lianyungang", 1996, 42)
    validation.mkdir(parents=True)
    (validation / "RESULTS_SUMMARY.md").write_text("validated", encoding="utf-8")
    for path in workflow.model_paths("lianyungang", 42).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"trained on validation protocol")
    lock = workflow.create_validation_lock("lianyungang", 42)
    workflow.require_validation_lock("lianyungang", 42)
    with pytest.raises(FileExistsError):
        workflow.create_validation_lock("lianyungang", 42)
    workflow.model_paths("lianyungang", 42)["cnn_gru"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        workflow.require_validation_lock("lianyungang", 42)
    assert lock.is_file()


def test_missing_shanwei_audit_fails_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(station_config, "SHANWEI_CONFIG", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="source audit"):
        station_config.get_station_config("shanwei")


def test_training_event_holdout_buffers_and_rejects_unreviewed_or_cross_split() -> None:
    event = {
        "event_id": "xiamen_test_case", "station": "xiamen", "storm_sid": "1990001",
        "split": "train", "window_start_utc": "1990-08-01T00:00:00",
        "window_end_exclusive_utc": "1990-08-06T00:00:00",
        "data_complete_same_split": "True", "input_evaluation_cross_split": "False",
        "review_decision": "yes", "review_note": "path, local wind and observed surge checked",
    }
    result = make_exclusions(pd.DataFrame([event]))
    verify_source_catalog(pd.DataFrame([event]), pd.DataFrame([event]))
    with pytest.raises(ValueError, match="changed source"):
        verify_source_catalog(pd.DataFrame([{**event, "storm_sid": "edited"}]), pd.DataFrame([event]))
    assert result.loc[0, "exclude_start_utc"] == "1990-07-31T00:00:00+00:00"
    assert result.loc[0, "exclude_end_exclusive_utc"] == "1990-08-09T00:00:00+00:00"
    with pytest.raises(ValueError, match="No manually reviewed"):
        make_exclusions(pd.DataFrame([{**event, "review_decision": "pending"}]))
    with pytest.raises(ValueError, match="crosses the training split"):
        make_exclusions(pd.DataFrame([{**event, "window_start_utc": "1995-12-30T00:00:00",
                                             "window_end_exclusive_utc": "1996-01-01T00:00:00"}]))


def test_core_review_reports_real_metrics_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(station_config, "MODULE_ROOT", tmp_path / "project")
    monkeypatch.setattr(workflow, "MODULE_ROOT", tmp_path / "project")
    source = workflow.result_package("beihai", 1997, 42)
    source.mkdir(parents=True)
    rows = [
        {"lead_hours": lead, "model": model, "rmse_cm": float(lead), "mae_cm": float(lead) / 2,
         "positive_p99_n": 88, "positive_p99_threshold_cm": 50.0,
         "positive_p99_rmse_cm": 20.0, "positive_p99_bias_cm": -5.0,
         "positive_p99_underestimation_ratio": 0.75}
        for lead in (1, 24, 48, 72) for model in ("persistence", "ridge", "cnn_gru", "cnn_gru_rollout6")
    ]
    pd.DataFrame(rows).to_csv(source / "rolling_metrics_long.csv", index=False)
    events = tmp_path / "events"
    events.mkdir()
    pd.DataFrame([{
        "event_id": "beihai_case", "station": "beihai", "model": "cnn_gru",
        "horizon_hours": 72, "origin_time_utc": "1997-08-01T00:00:00",
        "rmse_cm": 4.0, "mae_cm": 3.0, "peak_height_error_cm": -2.0,
        "absolute_peak_height_error_cm": 2.0, "peak_time_error_hours": 3.0,
    }]).to_csv(events / "all_event_metrics.csv", index=False)
    review = workflow.write_core_review("beihai", 42, events)
    content = (review / "结论报告.md").read_text(encoding="utf-8")
    assert "cnn_gru | 72 | 72.00 | 36.00" in content
    assert "cnn_gru | 72 | 88 | 50.00 | 20.00 | -5.00 | 75.0%" in content
    assert "beihai_case | cnn_gru | 72 | 1 | 4.00 | -2.00 | +3.0" in content
    assert (review / "event_case_summary.csv").is_file()
    assert (source / "rolling_metrics_long.csv").is_file()
    with pytest.raises(FileExistsError):
        workflow.write_core_review("beihai", 42, events)
