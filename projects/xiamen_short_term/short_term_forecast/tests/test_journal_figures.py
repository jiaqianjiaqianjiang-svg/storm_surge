from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.short_term_forecast.journal_figures import io_utils
from src.short_term_forecast.journal_figures.make_all_journal_figures import make_all_journal_figures
from src.short_term_forecast.journal_figures.make_xiamen_journal_figures import make_xiamen_journal_figures
from src.short_term_forecast.journal_figures.plot_event_comparison import plot_event_comparison
from src.short_term_forecast.journal_figures.plot_peak_analysis import plot_peak_analysis
from src.short_term_forecast.journal_figures.plot_residual_diagnostics import plot_residual_diagnostics
from src.short_term_forecast.journal_figures.plot_rolling_forecast import plot_rolling_forecast
from src.short_term_forecast.journal_figures.plot_window_comparison import (
    plot_window_metrics,
    plot_window_scatter,
    plot_window_timeseries,
    select_window_experiments,
)
from src.xiamen_forecast.export_final_results import copy_journal_figures


def _write_prediction_csv(path: Path, n: int = 24, meters: bool = True, rolling: bool = False, persistence: bool = False) -> None:
    time = pd.date_range("1985-11-01", periods=n, freq="h")
    observed_cm = 20 + 15 * np.sin(np.linspace(0, 2 * np.pi, n)) + np.linspace(0, 8, n)
    predicted_cm = observed_cm + np.linspace(-4, 5, n)
    scale = 100.0 if meters else 1.0
    data = {
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "observed": observed_cm / scale,
        "forecast" if rolling else "predicted": predicted_cm / scale,
    }
    if rolling:
        data["lead_step"] = np.arange(1, n + 1)
    if persistence:
        data["persistence"] = np.repeat(observed_cm[0], n) / scale
    pd.DataFrame(data).to_csv(path, index=False)


def _write_window_run(root: Path, name: str, input_steps: int, n_val: int = 1750) -> Path:
    run = root / name
    run.mkdir(parents=True)
    metrics = {
        "input_steps": input_steps,
        "model_name": "cnn",
        "pearson_r": 0.7 + input_steps / 100.0,
        "rmse": 0.12,
        "mae": 0.08,
        "rrmse": 0.3,
        "n_val": n_val,
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    _write_prediction_csv(run / "val_predictions.csv", n=min(n_val, 600), meters=True)
    return run


def _write_rolling_run(root: Path, name: str, n: int) -> Path:
    run = root / name
    run.mkdir(parents=True)
    _write_prediction_csv(run / "rolling_forecast.csv", n=n, meters=True, rolling=True, persistence=True)
    return run


def _assert_png(paths: dict[str, Path] | None) -> None:
    assert paths is not None
    assert set(paths) == {"png"}
    assert paths["png"].exists()
    assert paths["png"].stat().st_size > 0


def test_scan_results_and_unit_conversion(tmp_path: Path) -> None:
    _write_window_run(tmp_path, "ERA5_1985_1985_t8_h1", 8, n_val=72)
    _write_rolling_run(tmp_path, "ERA5_hourly_rolling_1985110100_n12", 12)

    experiments = io_utils.scan_results(tmp_path)
    names = {item.name for item in experiments}
    assert "ERA5_1985_1985_t8_h1" in names
    assert "ERA5_hourly_rolling_1985110100_n12" in names

    window = next(item for item in experiments if item.input_steps == 8)
    predictions = io_utils.load_predictions_for_experiment(window)
    assert predictions is not None
    assert predictions["observed"].max() > 1000 / 100  # converted to cm, not left as meters
    metrics = io_utils.compute_metrics_from_predictions(predictions)
    assert metrics["n"] == len(predictions)
    assert metrics["rmse"] > 0


def test_all_individual_plotters_generate_formats(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    figures = tmp_path / "figures"
    _write_window_run(results_root, "ERA5_1985_1985_t8_h1", 8, n_val=72)
    _write_window_run(results_root, "ERA5_1985_1985_hourly_t12_h1", 12)
    _write_window_run(results_root, "ERA5_1985_1985_hourly_t24_h1", 24)
    rolling_a = _write_rolling_run(results_root, "ERA5_hourly_rolling_1985110100_n12", 12)
    rolling_b = _write_rolling_run(results_root, "ERA5_hourly_rolling_1985110100_n72", 18)

    experiments = io_utils.scan_results(results_root)
    selected = select_window_experiments(experiments)
    assert 8 not in selected
    assert set(selected) == {12, 24}
    _assert_png(plot_window_metrics(experiments, figures))
    _assert_png(plot_window_timeseries(experiments, figures))
    _assert_png(plot_window_scatter(experiments, figures))

    rolling_csv = rolling_a / "rolling_forecast.csv"
    _assert_png(plot_rolling_forecast(rolling_csv, figures))
    _assert_png(plot_peak_analysis(rolling_csv, figures))
    _assert_png(plot_residual_diagnostics(rolling_csv, figures))
    _assert_png(plot_event_comparison([rolling_a / "rolling_forecast.csv", rolling_b / "rolling_forecast.csv"], figures))

    assert (figures / "peak_metrics.json").exists()
    assert (figures / "residual_statistics.json").exists()
    assert not list(figures.glob("*.pdf"))
    assert not list(figures.glob("*.svg"))


def test_make_all_journal_figures_manifest(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    figures = tmp_path / "journal_figures"
    _write_window_run(results_root, "ERA5_1985_1985_t8_h1", 8, n_val=72)
    _write_window_run(results_root, "ERA5_1985_1985_hourly_t12_h1", 12)
    _write_window_run(results_root, "ERA5_1985_1985_hourly_t24_h1", 24)
    _write_rolling_run(results_root, "ERA5_rolling_19851201_n7", 7)
    _write_rolling_run(results_root, "ERA5_rolling_19971201_n7", 7)

    manifest = make_all_journal_figures(results_root, figures)
    manifest_path = figures / "figure_manifest.csv"
    assert manifest_path.exists()
    assert not manifest.empty
    assert "figure_window_metrics" in set(manifest["figure_name"])
    assert (figures / "figure_window_metrics.png").exists()
    assert (figures / "figure_event_comparison.png").exists()
    assert "output_pdf" not in manifest.columns
    assert "output_svg" not in manifest.columns
    metric_sources = manifest.loc[manifest["figure_name"] == "figure_window_metrics", "source_experiments"].iloc[0]
    assert "ERA5_1985_1985_t8_h1" not in metric_sources
    assert not list(figures.glob("*.pdf"))
    assert not list(figures.glob("*.svg"))


def test_make_xiamen_journal_figures_from_formal_outputs(tmp_path: Path) -> None:
    project = tmp_path / "short_term_forecast"
    comparison = project / "outputs" / "experiments" / "xiamen" / "model_comparison"
    rolling = project / "outputs" / "experiments" / "xiamen" / "rolling_1996_seed42"
    baselines = project / "models" / "xiamen" / "baselines"
    formal = project / "models" / "xiamen" / "formal_seed42"
    for path in (comparison, rolling, baselines, formal):
        path.mkdir(parents=True, exist_ok=True)

    models = [
        "persistence", "ridge", "surge_mlp", "era5_cnn", "cnn",
        "cnn_lstm", "cnn_gru", "tcn", "transformer",
    ]
    pd.DataFrame(
        {
            "model": models,
            "n": 120,
            "pearson_r": np.linspace(0.82, 0.98, len(models)),
            "rmse_cm": np.linspace(12.0, 3.8, len(models)),
            "mae_cm": np.linspace(9.0, 2.8, len(models)),
            "bias_cm": np.linspace(-0.4, 0.4, len(models)),
        }
    ).to_csv(comparison / "validation_model_metrics.csv", index=False)

    leads = np.array([1, 3, 6, 12, 24, 48, 72])
    rolling_models = [*models, "cnn_gru_rollout6"]
    wide = {"lead_hours": leads, "valid_samples": 60}
    long_rows = []
    for model_index, model in enumerate(rolling_models):
        values = 3.0 + 0.12 * leads + model_index * 0.15
        if model == "cnn_gru_rollout6":
            values = 2.8 + 0.08 * leads
        wide[model] = values
        for lead, value in zip(leads, values):
            long_rows.append(
                {
                    "lead_hours": lead,
                    "model": model,
                    "rmse_cm": value,
                    "top5_rmse_cm": value * 1.4,
                    "rapid_rise_rmse_cm": value * 1.2,
                }
            )
    pd.DataFrame(wide).to_csv(rolling / "rolling_rmse_table.csv", index=False)
    pd.DataFrame(long_rows).to_csv(rolling / "rolling_metrics_long.csv", index=False)

    prediction_rows = []
    for lead in leads:
        observed = 0.25 + 0.12 * np.sin(np.linspace(0, 2 * np.pi, 60))
        predicted = observed + lead / 10000.0 + np.linspace(-0.01, 0.01, 60)
        for obs, pred in zip(observed, predicted):
            prediction_rows.append(
                {
                    "lead_hours": lead,
                    "observed_m": obs,
                    "cnn_gru_rollout6_m": pred,
                }
            )
    pd.DataFrame(prediction_rows).to_csv(
        rolling / "rolling_predictions_selected_leads.csv", index=False
    )

    times = pd.date_range("1996-01-01", periods=120, freq="h")
    observed = 0.2 + 0.1 * np.sin(np.linspace(0, 5 * np.pi, 120))
    pd.DataFrame(
        {
            "datetime": times,
            "observed_m": observed,
            "persistence_m": observed + 0.03,
            "ridge_m": observed + np.linspace(-0.02, 0.02, 120),
        }
    ).to_csv(baselines / "validation_baseline_predictions.csv", index=False)
    for model, amplitude in (("cnn_lstm", 0.014), ("cnn_gru", 0.012)):
        model_dir = formal / model
        model_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "datetime": times,
                "observed_m": observed,
                "predicted_m": observed + amplitude * np.cos(np.linspace(0, 4 * np.pi, 120)),
            }
        ).to_csv(model_dir / "validation_predictions.csv", index=False)

    rollout = formal / "cnn_gru_rollout6"
    rollout.mkdir(parents=True)
    pd.DataFrame(
        {
            "epoch": np.arange(6),
            "teacher_forcing_ratio": [0, 1, 0.75, 0.5, 0.25, 0],
            "train_scaled_mse": [np.nan, 0.08, 0.07, 0.06, 0.055, 0.052],
            "validation_recursive_scaled_mse": [0.09, 0.075, 0.068, 0.064, 0.066, 0.069],
            "validation_recursive_rmse_cm": [6.2, 5.8, 5.5, 5.3, 5.4, 5.6],
        }
    ).to_csv(rollout / "loss_history.csv", index=False)

    output = tmp_path / "figures"
    manifest = make_xiamen_journal_figures(project_root=project, output_dir=output)
    expected = {
        "fig01_one_step_model_comparison",
        "fig02_rmse_by_lead",
        "fig03_rollout_improvement",
        "fig04_extreme_and_rapid_rise_metrics",
        "fig05_rollout_training",
        "fig06_validation_scatter",
        "fig07_rollout_scatter_by_lead",
        "fig08_cnn_gru_residual_diagnostics",
        "fig09_cnn_gru_peak_analysis",
    }
    assert expected.issubset(set(manifest.loc[manifest.status == "ok", "figure_name"]))
    assert all((output / f"{name}.png").is_file() for name in expected)
    assert (output / "figure_manifest.csv").is_file()
    assert (output / "FIGURE_GUIDE.md").is_file()


def test_copy_journal_figures_excludes_prediction_tables(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "fig01_summary.png").write_bytes(b"png")
    (source / "figure_manifest.csv").write_text("status\nok\n", encoding="utf-8")
    (source / "FIGURE_GUIDE.md").write_text("guide", encoding="utf-8")
    (source / "rolling_predictions_selected_leads.csv").write_text("large", encoding="utf-8")

    copied = copy_journal_figures(source, destination)

    assert {path.name for path in copied} == {
        "fig01_summary.png",
        "figure_manifest.csv",
        "FIGURE_GUIDE.md",
    }
    assert not (destination / "rolling_predictions_selected_leads.csv").exists()
