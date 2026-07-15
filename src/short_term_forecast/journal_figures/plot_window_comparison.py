"""Journal figures for input-window comparison experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from . import io_utils
from .style import (
    add_bar_labels,
    apply_axis_style,
    close_figure,
    format_time_axis,
    label_panels,
    metric_label,
    save_figure,
    setup_journal_style,
)


WINDOWS = (8, 12, 24)


def validation_sample_count(experiment: io_utils.ExperimentResult) -> int | None:
    """Return validation sample count from metrics, falling back to prediction CSV length."""

    for key in ("n_val", "val_size", "validation_samples", "n_validation"):
        value = experiment.metrics.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    frame = io_utils.load_predictions_for_experiment(experiment)
    return int(len(frame)) if frame is not None else None


def is_valid_window_experiment(experiment: io_utils.ExperimentResult) -> bool:
    """Filter out small debug/test runs from formal window-comparison figures."""

    n_val = validation_sample_count(experiment)
    if n_val is not None and n_val < config.MIN_VALIDATION_SAMPLES:
        print(f"Skipped experiment:\n{experiment.name}\nReason:\nvalidation sample number too small (n_val={n_val})")
        return False
    return True


def select_window_experiments(experiments: list[io_utils.ExperimentResult]) -> dict[int, io_utils.ExperimentResult]:
    """Pick one one-step experiment for each supported input window."""

    selected: dict[int, io_utils.ExperimentResult] = {}
    for experiment in experiments:
        if experiment.is_rolling:
            continue
        window = experiment.input_steps
        if window not in WINDOWS:
            continue
        if not is_valid_window_experiment(experiment):
            continue
        if window not in selected:
            selected[window] = experiment
            continue
        current = selected[window]
        if "val_predictions" not in current.files and "val_predictions" in experiment.files:
            selected[window] = experiment
    return selected


def _metrics_table(selected: dict[int, io_utils.ExperimentResult]) -> pd.DataFrame:
    rows = []
    for window in sorted(selected):
        experiment = selected[window]
        metrics = dict(experiment.metrics)
        pred = io_utils.load_predictions_for_experiment(experiment)
        if pred is not None:
            computed = io_utils.compute_metrics_from_predictions(pred)
            for key, value in computed.items():
                metrics.setdefault(key, value)
        rows.append({"input_steps": window, "experiment": experiment.name, **metrics})
    return pd.DataFrame(rows)


def plot_window_metrics(
    experiments: list[io_utils.ExperimentResult],
    output_dir: Path,
    figure_name: str = "figure_window_metrics",
    language: str = "en",
) -> dict[str, Path] | None:
    setup_journal_style(language=language)
    selected = select_window_experiments(experiments)
    table = _metrics_table(selected)
    required = ["pearson_r", "rmse", "mae", "rrmse"]
    if table.empty or not any(metric in table.columns for metric in required):
        io_utils.warn("No window metrics found; figure_window_metrics skipped")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    axes_flat = axes.ravel()
    labels = [f"t={int(value)} h" for value in table["input_steps"]]
    for ax, metric in zip(axes_flat, required):
        values = table[metric].to_numpy(float) if metric in table.columns else np.full(len(table), np.nan)
        ax.bar(labels, values, color="#6BAED6", edgecolor="#333333", linewidth=0.6)
        ax.set_ylabel(metric_label(metric))
        add_bar_labels(ax, values, fmt="{:.2f}" if metric != "pearson_r" else "{:.3f}")
        apply_axis_style(ax)
    label_panels(axes_flat)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, figure_name)
    close_figure(fig)
    return paths


def _prediction_frames(selected: dict[int, io_utils.ExperimentResult]) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for window, experiment in selected.items():
        frame = io_utils.load_predictions_for_experiment(experiment)
        if frame is not None:
            frames[window] = frame
    return frames


def plot_window_timeseries(
    experiments: list[io_utils.ExperimentResult],
    output_dir: Path,
    figure_name: str = "figure_window_timeseries",
    language: str = "en",
) -> dict[str, Path] | None:
    setup_journal_style(language=language)
    selected = select_window_experiments(experiments)
    frames = _prediction_frames(selected)
    frames = {window: df.dropna(subset=["datetime"]) for window, df in frames.items()}
    frames = {window: df for window, df in frames.items() if not df.empty}
    if len(frames) < 2:
        io_utils.warn("Need at least two window prediction CSVs for figure_window_timeseries; skipped")
        return None

    common_times = None
    for frame in frames.values():
        times = set(pd.to_datetime(frame["datetime"]).dropna())
        common_times = times if common_times is None else common_times.intersection(times)
    if not common_times:
        io_utils.warn("Window prediction CSVs have no overlapping validation times; skipped")
        return None
    common_index = pd.DatetimeIndex(sorted(common_times))

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    first = next(iter(frames.values())).set_index("datetime").sort_index()
    ax.plot(common_index, first.loc[common_index, "observed"], color="black", label="Observed", linewidth=1.5)
    colors = {8: "#74A9CF", 12: "#2B8CBE", 24: "#045A8D"}
    for window in sorted(frames):
        frame = frames[window].set_index("datetime").sort_index()
        ax.plot(common_index, frame.loc[common_index, "predicted"], label=f"t={window} h", color=colors.get(window), linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Storm surge (cm)")
    apply_axis_style(ax)
    ax.legend(loc="best")
    format_time_axis(ax)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, figure_name)
    close_figure(fig)
    return paths


def plot_window_scatter(
    experiments: list[io_utils.ExperimentResult],
    output_dir: Path,
    figure_name: str = "figure_window_scatter",
    language: str = "en",
) -> dict[str, Path] | None:
    setup_journal_style(language=language)
    selected = select_window_experiments(experiments)
    frames = _prediction_frames(selected)
    if not frames:
        io_utils.warn("No window prediction CSVs found for figure_window_scatter; skipped")
        return None

    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0), squeeze=False)
    axes_flat = axes.ravel()
    for ax, window in zip(axes_flat, sorted(frames)):
        frame = frames[window]
        obs = frame["observed"].to_numpy(float)
        pred = frame["predicted"].to_numpy(float)
        metric = io_utils.compute_metrics_from_predictions(frame)
        hb = ax.hexbin(obs, pred, gridsize=35, mincnt=1, cmap="Blues")
        low = float(np.nanmin([np.nanmin(obs), np.nanmin(pred)]))
        high = float(np.nanmax([np.nanmax(obs), np.nanmax(pred)]))
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=0.9)
        ax.set_title(f"t={window} h")
        ax.set_xlabel("Observed (cm)")
        ax.set_ylabel("Predicted (cm)")
        ax.text(
            0.04,
            0.96,
            f"r={metric['pearson_r']:.2f}\nRMSE={metric['rmse']:.1f} cm\nMAE={metric['mae']:.1f} cm\nn={metric['n']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )
        apply_axis_style(ax, grid=False)
        fig.colorbar(hb, ax=ax, label="Count", fraction=0.046, pad=0.03)
    label_panels(axes_flat)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, figure_name)
    close_figure(fig)
    return paths


def make_window_comparison_figures(results_root: Path, output_dir: Path, language: str = "en") -> dict[str, dict[str, Path] | None]:
    experiments = io_utils.scan_results(results_root)
    return {
        "figure_window_metrics": plot_window_metrics(experiments, output_dir, language=language),
        "figure_window_timeseries": plot_window_timeseries(experiments, output_dir, language=language),
        "figure_window_scatter": plot_window_scatter(experiments, output_dir, language=language),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate journal figures comparing input windows.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_window_comparison_figures(args.results_root, args.output_dir, language=args.language)


if __name__ == "__main__":
    main()
