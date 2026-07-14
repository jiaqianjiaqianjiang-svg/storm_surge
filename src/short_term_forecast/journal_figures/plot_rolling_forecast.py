"""Four-panel journal figure for one rolling forecast run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import io_utils
from .style import apply_axis_style, close_figure, format_time_axis, label_panels, save_figure, setup_journal_style


def _cumulative_metrics(error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rmse = []
    mae = []
    for idx in range(1, len(error) + 1):
        partial = error[:idx]
        partial = partial[np.isfinite(partial)]
        if partial.size == 0:
            rmse.append(np.nan)
            mae.append(np.nan)
        else:
            rmse.append(float(np.sqrt(np.mean(partial**2))))
            mae.append(float(np.mean(np.abs(partial))))
    return np.asarray(rmse), np.asarray(mae)


def plot_rolling_forecast(
    forecast_csv: Path,
    output_dir: Path,
    figure_name: str = "figure_rolling_forecast",
    language: str = "en",
) -> dict[str, Path] | None:
    setup_journal_style(language=language)
    df, _ = io_utils.load_prediction_csv(forecast_csv)
    if df is None:
        io_utils.warn(f"Cannot build rolling forecast figure from {forecast_csv}")
        return None

    dates = pd.to_datetime(df["datetime"], errors="coerce")
    if dates.isna().all():
        dates = pd.RangeIndex(1, len(df) + 1)
        x_label = "Lead time (h)"
    else:
        x_label = "Time"
    lead_series = pd.to_numeric(df["lead_time"], errors="coerce").to_numpy(float)
    lead = np.where(np.isfinite(lead_series), lead_series, np.arange(1, len(df) + 1, dtype=float))
    observed = df["observed"].to_numpy(float)
    predicted = df["predicted"].to_numpy(float)
    error = predicted - observed
    abs_error = np.abs(error)
    cumulative_rmse, cumulative_mae = _cumulative_metrics(error)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax_ts, ax_err, ax_abs, ax_cum = axes.ravel()

    ax_ts.plot(dates, observed, color="black", marker="o", markersize=3, label="Observed")
    ax_ts.plot(dates, predicted, color="#0072B2", marker="s", markersize=3, label="Rolling forecast")
    if "persistence" in df.columns:
        ax_ts.plot(dates, df["persistence"], color="#E69F00", linestyle="--", marker="^", markersize=3, label="Persistence")
    ax_ts.set_xlabel(x_label)
    ax_ts.set_ylabel("Storm surge (cm)")
    ax_ts.legend(loc="best")
    apply_axis_style(ax_ts)

    ax_err.axhline(0, color="#333333", linewidth=0.8)
    colors = np.where(error >= 0, "#0072B2", "#D55E00")
    ax_err.bar(lead, error, color=colors, edgecolor="none", width=0.8)
    ax_err.set_xlabel("Lead time (h)")
    ax_err.set_ylabel("Forecast error (cm)")
    apply_axis_style(ax_err)

    ax_abs.plot(lead, abs_error, color="#6A3D9A", marker="o", markersize=3)
    ax_abs.set_xlabel("Lead time (h)")
    ax_abs.set_ylabel("Absolute error (cm)")
    apply_axis_style(ax_abs)

    ax_cum.plot(lead, cumulative_rmse, color="#D55E00", marker="o", markersize=3, label="Cumulative RMSE")
    ax_cum.plot(lead, cumulative_mae, color="#0072B2", marker="s", markersize=3, label="Cumulative MAE")
    ax_cum.set_xlabel("Lead time (h)")
    ax_cum.set_ylabel("Cumulative error (cm)")
    ax_cum.legend(loc="best")
    apply_axis_style(ax_cum)

    for axis in (ax_ts,):
        if not isinstance(dates, pd.RangeIndex):
            format_time_axis(axis)
    label_panels(axes.ravel())
    fig.tight_layout()
    paths = save_figure(fig, output_dir, figure_name)
    close_figure(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a four-panel rolling forecast journal figure.")
    parser.add_argument("--forecast-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-name", default="figure_rolling_forecast")
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_rolling_forecast(args.forecast_csv, args.output_dir, args.figure_name, args.language)


if __name__ == "__main__":
    main()
