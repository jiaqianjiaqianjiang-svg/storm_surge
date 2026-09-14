from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.short_term_forecast.journal_figures import io_utils
from src.short_term_forecast.journal_figures.make_all_journal_figures import make_all_journal_figures
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
