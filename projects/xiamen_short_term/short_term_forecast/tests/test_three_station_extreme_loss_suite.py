from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.xiamen_forecast import extreme_loss_suite as suite
from src.xiamen_forecast import station_config
from src.xiamen_forecast.extreme_loss_suite import LOSS_SCHEMES
from src.xiamen_forecast.rolling_diagnostics import build_metrics
from src.xiamen_forecast.run_three_station_extreme_loss_suite import (
    Task,
    one_step_task,
    run_task,
    rollout_task,
)
from src.xiamen_forecast import summarize_three_station_extreme_losses as summary_module


def _write(path: Path, content: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _loss_config(loss_type: str) -> dict[str, object]:
    return {
        "loss_type": loss_type,
        "alpha": 2.0,
        "beta": 1.0,
        "huber_delta_cm": 5.0,
        "q90_m": None if loss_type == "mse" else 0.2,
        "q99_m": None if loss_type == "mse" else 0.4,
    }


def test_fixed_scheme_matrix_has_shared_wmse_base_and_isolated_rollouts() -> None:
    assert [item.scheme_id for item in LOSS_SCHEMES] == [
        "mse/mse", "wmse/mse", "wmse/wmse", "awmse/awmse", "whuber/whuber"
    ]
    assert LOSS_SCHEMES[1].one_step_directory == LOSS_SCHEMES[2].one_step_directory
    assert len({item.rollout_directory for item in LOSS_SCHEMES}) == 5
    assert LOSS_SCHEMES[0].rollout_directory == "cnn_gru_rollout6"


def test_artifact_audit_distinguishes_compatible_legacy_and_mismatch(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(station_config, "MODULE_ROOT", tmp_path)
    station = station_config.get_station_config("xiamen")
    root = suite.one_step_path(station, "wmse")
    for name in suite.ONE_STEP_REQUIRED_FILES:
        _write(root / name)
    config = {
        "station": "xiamen", "model_type": "cnn_gru", "loss_type": "wmse",
        "validation_only": True, "seed": 42, "input_steps": 24,
        "batch_size": 32, "epochs": 50, "patience": 8, "lr": 0.001,
        "train_start_year": 1970, "train_end_year": 1995,
        "validation_year": 1996,
    }
    (root / "training_config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "extreme_loss_config.json").write_text(
        json.dumps(_loss_config("wmse")), encoding="utf-8"
    )
    assert suite.inspect_one_step_artifact(station, "wmse")["status"] == "compatible"
    del config["batch_size"]
    (root / "training_config.json").write_text(json.dumps(config), encoding="utf-8")
    assert suite.inspect_one_step_artifact(station, "wmse")["status"] == "legacy_metadata"
    config["batch_size"] = 64
    (root / "training_config.json").write_text(json.dumps(config), encoding="utf-8")
    assert suite.inspect_one_step_artifact(station, "wmse")["status"] == "configuration_mismatch"


def test_rollout_audit_checks_the_recorded_one_step_dependency(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(station_config, "MODULE_ROOT", tmp_path)
    station = station_config.get_station_config("lianyungang")
    scheme = next(item for item in LOSS_SCHEMES if item.scheme_id == "wmse/wmse")
    root = suite.rollout_path(station, scheme)
    for name in suite.ROLLOUT_REQUIRED_FILES:
        _write(root / name)
    metadata = {
        "station": "lianyungang", "training_years": [1975, 1995],
        "validation_year": 1996, "test_year_loaded": False,
        "base_model_type": "cnn_gru", "base_experiment": "cnn_gru_extreme_wmse",
        "base_checkpoint": str(suite.one_step_path(station, "wmse") / "best_model.pth"),
        "rollout_steps": 6, "teacher_forcing_epochs": 5, "seed": 42,
        "epochs": 20, "patience": 5, "batch_size": 256,
        "embedding_batch_size": 1024, "learning_rate": 0.0001,
    }
    (root / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "extreme_loss_config.json").write_text(
        json.dumps(_loss_config("wmse")), encoding="utf-8"
    )
    assert suite.inspect_rollout_artifact(station, scheme)["status"] == "compatible"
    metadata["base_checkpoint"] = str(
        suite.one_step_path(station, "awmse") / "best_model.pth"
    )
    (root / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert suite.inspect_rollout_artifact(station, scheme)["status"] == "configuration_mismatch"


def test_generated_training_commands_are_explicit_and_dependency_matched(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(station_config, "MODULE_ROOT", tmp_path)
    station = station_config.get_station_config("beihai")
    one = one_step_task(station, "awmse", 42, "cuda", 0)
    command = list(one.command)
    assert command[command.index("--station") + 1] == "beihai"
    assert command[command.index("--loss-type") + 1] == "awmse"
    assert command[command.index("--validation-year") + 1] == "1996"
    assert command[command.index("--test-year") + 1] == "1997"
    assert "--validation-only" in command
    rollout = rollout_task(station, LOSS_SCHEMES[3], 42, "cuda", 0)
    rollout_command = list(rollout.command)
    assert rollout_command[rollout_command.index("--loss-type") + 1] == "awmse"
    base = Path(rollout_command[rollout_command.index("--base-checkpoint") + 1])
    assert base.parent.name == "cnn_gru_extreme_awmse"
    assert rollout.output_dir.name == "cnn_gru_extreme_awmse_rollout6_awmse"


def test_scheduler_never_silently_overwrites_conflicting_or_incomplete_results(
    tmp_path: Path,
) -> None:
    state: dict[str, object] = {}
    task = Task(
        "synthetic", "synthetic conflict", ("must-not-run",), tmp_path / "result",
        lambda: {
            "status": "configuration_mismatch", "path": str(tmp_path / "result"),
            "missing_files": [], "mismatches": ["seed=7, expected 42"],
            "missing_metadata": [],
        },
    )
    run_task(task, True, False, False, tmp_path / "logs", state)
    assert state["synthetic"]["status"] == "blocked_configuration_mismatch"
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        run_task(task, False, False, False, tmp_path / "logs", {})
    incomplete = Task(
        "partial", "partial result", ("must-not-run",), tmp_path / "partial",
        lambda: {
            "status": "incomplete", "path": str(tmp_path / "partial"),
            "missing_files": ["best_model.pth"], "mismatches": [],
            "missing_metadata": [],
        },
    )
    with pytest.raises(RuntimeError, match="Incomplete directory"):
        run_task(incomplete, False, False, False, tmp_path / "logs", {})


def test_rollout_metrics_include_positive_event_peak_height_and_timing() -> None:
    times = pd.date_range("1996-01-01", periods=240, freq="h")
    surge = np.sin(np.arange(240) / 8) * 0.2
    surge[80] = 0.8
    surge[160] = 0.7
    origins = np.arange(24, 168, dtype=np.int64)
    predictions = {
        "ridge": np.repeat(surge[origins, None], 72, axis=1),
        "cnn_gru_rollout6": np.repeat(surge[origins, None], 72, axis=1),
    }
    frame = build_metrics(surge, origins, predictions, times)
    expected = {
        "positive_event_count",
        "positive_event_peak_rmse_cm",
        "positive_event_peak_bias_cm",
        "positive_event_timing_mae_hours",
        "positive_event_underestimation_ratio",
    }
    assert expected.issubset(frame.columns)
    assert set(frame.lead_hours) == {1, 3, 6, 12, 24, 48, 72}
    assert (frame.positive_event_count > 0).all()


def test_cross_station_summary_writes_required_tables_figures_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(station_config, "MODULE_ROOT", tmp_path / "project")
    losses = ("mse", "wmse", "awmse", "whuber")
    for station_id in ("xiamen", "lianyungang", "beihai"):
        station = station_config.get_station_config(station_id)
        one_root = station.experiment_root / "extreme_loss_suite_one_step_1996_seed42"
        one_root.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "experiment": loss,
                    "display_name": loss.upper(),
                    "common_samples": 100,
                    "rmse_cm": 4.0 + index,
                    "mae_cm": 3.0,
                    "bias_cm": -0.2,
                    "pearson_r": 0.98,
                    "positive_p99_rmse_cm": 8.0 - index,
                    "positive_p99_bias_cm": -2.0,
                    "positive_p99_underestimation_ratio": 0.7,
                    "positive_event_peak_rmse_cm": 7.0,
                    "positive_event_peak_bias_cm": -1.0,
                    "positive_event_timing_mae_hours": 2.0,
                }
                for index, loss in enumerate(losses)
            ]
        ).to_csv(one_root / "validation_extreme_loss_summary.csv", index=False)
        details = {
            loss: {
                f"positive_extreme_p{percentile}": {
                    "n": 10, "threshold_cm": float(percentile),
                    "rmse_cm": 8.0, "bias_cm": -2.0,
                    "underestimation_ratio": 0.7,
                }
                for percentile in (90, 95, 99)
            }
            for loss in losses
        }
        (one_root / "validation_extreme_loss_details.json").write_text(
            json.dumps(details), encoding="utf-8"
        )
        rolling_rows = []
        for scheme in LOSS_SCHEMES:
            model = "cnn_gru_rollout6" if scheme.scheme_id == "mse/mse" else scheme.rolling_label
            for lead in (1, 3, 6, 12, 24, 48, 72):
                rolling_rows.append(
                    {
                        "lead_hours": lead, "model": model, "n": 100,
                        "rmse_cm": 4.0 + lead / 10, "mae_cm": 3.0,
                        "bias_cm": -0.2, "pearson_r": 0.95,
                        "positive_p99_rmse_cm": 9.0,
                        "positive_p99_bias_cm": -2.0,
                        "positive_p99_underestimation_ratio": 0.7,
                        "positive_event_count": 5,
                        "positive_event_peak_rmse_cm": 8.0,
                        "positive_event_peak_bias_cm": -1.0,
                        "positive_event_timing_mae_hours": 3.0,
                    }
                )
        rolling_root = station.experiment_root / "extreme_loss_suite_rolling_1996_seed42"
        rolling_root.mkdir(parents=True)
        pd.DataFrame(rolling_rows).to_csv(
            rolling_root / "rolling_metrics_long.csv", index=False
        )
        for loss in losses:
            root = suite.one_step_path(station, loss)
            root.mkdir(parents=True, exist_ok=True)
            (root / "extreme_loss_config.json").write_text(
                json.dumps(_loss_config(loss)), encoding="utf-8"
            )
        for scheme in LOSS_SCHEMES:
            root = suite.rollout_path(station, scheme)
            root.mkdir(parents=True, exist_ok=True)
            (root / "extreme_loss_config.json").write_text(
                json.dumps(_loss_config(scheme.rollout_loss)), encoding="utf-8"
            )
    output = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        ["summarize", "--stations", "xiamen", "lianyungang", "beihai", "--output-dir", str(output)],
    )
    summary_module.main()
    expected = {
        "three_station_extreme_loss_validation.csv",
        "three_station_extreme_rollout_validation.csv",
        "three_station_extreme_loss_comparison.png",
        "three_station_extreme_rollout_comparison.png",
        "three_station_extreme_loss_report.md",
        "experiment_metadata.json",
    }
    assert expected == {path.name for path in output.iterdir()}
    rollout = pd.read_csv(output / "three_station_extreme_rollout_validation.csv")
    assert len(rollout) == 3 * 5 * 7
    assert set(rollout.scheme) == {item.scheme_id for item in LOSS_SCHEMES}
    wmse_mse = rollout.loc[rollout.scheme == "wmse/mse"].iloc[0]
    assert wmse_mse.one_step_training_loss_q90_m == 0.2
    assert pd.isna(wmse_mse.rollout_training_loss_q90_m)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        summary_module.main()
