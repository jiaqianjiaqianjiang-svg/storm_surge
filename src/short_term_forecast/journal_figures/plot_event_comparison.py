"""Compare multiple rolling forecast events in one journal figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import io_utils
from .style import apply_axis_style, close_figure, format_time_axis, save_figure, setup_journal_style


def _event_metrics(df: pd.DataFrame) -> dict[str, float]:
    obs = df["observed"].to_numpy(float)
    pred = df["predicted"].to_numpy(float)
    error = pred - obs
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def _event_name(path: Path) -> str:
    parent = path.parent.name
    return parent if parent else path.stem


def plot_event_comparison(
    forecast_csv: list[Path],
    output_dir: Path,
    figure_name: str = "figure_event_comparison",
    language: str = "en",
) -> dict[str, Path] | None:
    setup_journal_style(language=language)
    events: list[tuple[str, pd.DataFrame]] = []
    for path in forecast_csv:
        df, _ = io_utils.load_prediction_csv(path)
        if df is not None:
            events.append((_event_name(path), df))
    if not events:
        io_utils.warn("No readable rolling forecast CSVs for event comparison; skipped")
        return None

    all_values = np.concatenate([frame[["observed", "predicted"]].to_numpy(float).ravel() for _, frame in events])
    finite_values = all_values[np.isfinite(all_values)]
    y_pad = 0.08 * (np.nanmax(finite_values) - np.nanmin(finite_values)) if finite_values.size else 1.0
    y_low = float(np.nanmin(finite_values) - y_pad) if finite_values.size else -1.0
    y_high = float(np.nanmax(finite_values) + y_pad) if finite_values.size else 1.0
    all_errors = np.concatenate([(frame["predicted"] - frame["observed"]).to_numpy(float) for _, frame in events])
    finite_errors = all_errors[np.isfinite(all_errors)]
    e_abs = float(np.nanmax(np.abs(finite_errors))) if finite_errors.size else 1.0
    e_lim = e_abs * 1.12 if e_abs > 0 else 1.0

    fig, axes = plt.subplots(len(events), 2, figsize=(7.2, max(2.4, 2.2 * len(events))), squeeze=False, sharey="col")
    for row, (name, frame) in enumerate(events):
        ax_ts, ax_err = axes[row]
        dates = pd.to_datetime(frame["datetime"], errors="coerce")
        if dates.isna().all():
            x_values = np.arange(1, len(frame) + 1)
            x_label = "Lead time (h)"
        else:
            x_values = dates
            x_label = "Time"
        observed = frame["observed"].to_numpy(float)
        predicted = frame["predicted"].to_numpy(float)
        error = predicted - observed
        metrics = _event_metrics(frame)

        ax_ts.plot(x_values, observed, color="black", marker="o", markersize=2.6, label="Observed")
        ax_ts.plot(x_values, predicted, color="#0072B2", marker="s", markersize=2.6, label="Predicted")
        ax_ts.set_ylim(y_low, y_high)
        ax_ts.set_ylabel("Storm surge (cm)")
        ax_ts.set_title(name, loc="left", fontsize=8)
        ax_ts.text(
            0.98,
            0.04,
            f"RMSE={metrics['rmse']:.1f} cm\nMAE={metrics['mae']:.1f} cm\nBias={metrics['bias']:.1f} cm",
            transform=ax_ts.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
        )
        if row == 0:
            ax_ts.legend(loc="best")
        apply_axis_style(ax_ts)

        ax_err.axhline(0, color="black", linestyle="--", linewidth=0.8)
        colors = np.where(error >= 0, "#0072B2", "#D55E00")
        ax_err.bar(np.arange(1, len(error) + 1), error, color=colors, width=0.8)
        ax_err.set_ylim(-e_lim, e_lim)
        ax_err.set_ylabel("Error (cm)")
        apply_axis_style(ax_err)

        if row == len(events) - 1:
            ax_ts.set_xlabel(x_label)
            ax_err.set_xlabel("Lead time (h)")
        if not isinstance(x_values, np.ndarray):
            format_time_axis(ax_ts)

    fig.tight_layout()
    paths = save_figure(fig, output_dir, figure_name)
    close_figure(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate event comparison journal figure.")
    parser.add_argument("--forecast-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-name", default="figure_event_comparison")
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_event_comparison(args.forecast_csv, args.output_dir, args.figure_name, args.language)


if __name__ == "__main__":
    main()

