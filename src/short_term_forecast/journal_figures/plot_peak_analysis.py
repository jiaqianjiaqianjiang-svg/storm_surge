"""Peak storm-surge analysis figures and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import io_utils
from .style import apply_axis_style, close_figure, label_panels, save_figure, setup_journal_style


DEFAULT_PERCENTILES = (90.0, 95.0, 99.0)


def compute_peak_metrics(df: pd.DataFrame, percentile: float) -> tuple[pd.DataFrame, dict[str, float]]:
    observed = df["observed"].to_numpy(float)
    threshold = float(np.nanpercentile(observed, percentile))
    peak_df = df.loc[df["observed"] >= threshold].copy()
    obs = peak_df["observed"].to_numpy(float)
    pred = peak_df["predicted"].to_numpy(float)
    error = pred - obs
    if len(error) == 0:
        metrics = {
            "percentile": percentile,
            "threshold_cm": threshold,
            "n_peak": 0,
            "peak_rmse": np.nan,
            "peak_mae": np.nan,
            "peak_bias": np.nan,
            "underestimation_ratio": np.nan,
            "peak_correlation": np.nan,
        }
    else:
        if len(obs) >= 2 and np.nanstd(obs) > 0 and np.nanstd(pred) > 0:
            corr = float(np.corrcoef(obs, pred)[0, 1])
        else:
            corr = float("nan")
        metrics = {
            "percentile": percentile,
            "threshold_cm": threshold,
            "n_peak": int(len(error)),
            "peak_rmse": float(np.sqrt(np.mean(error**2))),
            "peak_mae": float(np.mean(np.abs(error))),
            "peak_bias": float(np.mean(error)),
            "underestimation_ratio": float(np.mean(error < 0)),
            "peak_correlation": corr,
        }
    return peak_df, metrics


def plot_peak_analysis(
    prediction_csv: Path,
    output_dir: Path,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
    top_n: int = 10,
    figure_name: str = "figure_peak_analysis",
    language: str = "en",
) -> dict[str, Path] | None:
    setup_journal_style(language=language)
    df, _ = io_utils.load_prediction_csv(prediction_csv)
    if df is None:
        io_utils.warn(f"Cannot build peak analysis from {prediction_csv}")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    peak_tables: dict[float, pd.DataFrame] = {}
    metrics_rows = []
    for percentile in percentiles:
        peak_df, metrics = compute_peak_metrics(df, percentile)
        peak_tables[percentile] = peak_df
        metrics_rows.append(metrics)
    metrics_path = output_dir / "peak_metrics.json"
    metrics_path.write_text(json.dumps(metrics_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    primary_percentile = 95.0 if 95.0 in peak_tables else percentiles[0]
    peak_df = peak_tables[primary_percentile]
    metrics_df = pd.DataFrame(metrics_rows)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    ax_scatter, ax_error, ax_top, ax_ratio = axes.ravel()

    if peak_df.empty:
        ax_scatter.text(0.5, 0.5, "No peak samples", transform=ax_scatter.transAxes, ha="center", va="center")
    else:
        error = peak_df["predicted"] - peak_df["observed"]
        colors = np.where(error >= 0, "#0072B2", "#D55E00")
        ax_scatter.scatter(peak_df["observed"], peak_df["predicted"], c=colors, alpha=0.85, s=24)
        low = float(np.nanmin([peak_df["observed"].min(), peak_df["predicted"].min()]))
        high = float(np.nanmax([peak_df["observed"].max(), peak_df["predicted"].max()]))
        ax_scatter.plot([low, high], [low, high], color="black", linestyle="--", linewidth=0.9)
    ax_scatter.set_xlabel("Observed peak (cm)")
    ax_scatter.set_ylabel("Predicted peak (cm)")
    apply_axis_style(ax_scatter)

    if not peak_df.empty:
        error = peak_df["predicted"] - peak_df["observed"]
        ax_error.hist(error, bins=min(12, max(4, len(error))), color="#BDBDBD", edgecolor="#333333")
        ax_error.axvline(0, color="black", linestyle="--", linewidth=0.9)
        ax_error.axvline(np.nanmean(error), color="#D55E00", linewidth=1.2, label="Mean bias")
        ax_error.legend(loc="best")
    ax_error.set_xlabel("Peak error (cm)")
    ax_error.set_ylabel("Count")
    apply_axis_style(ax_error)

    top = df.sort_values("observed", ascending=False).head(top_n).copy()
    x = np.arange(len(top))
    width = 0.38
    ax_top.bar(x - width / 2, top["observed"], width, color="black", label="Observed")
    ax_top.bar(x + width / 2, top["predicted"], width, color="#0072B2", label="Predicted")
    ax_top.set_xlabel(f"Top-{len(top)} events")
    ax_top.set_ylabel("Storm surge (cm)")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([str(i + 1) for i in range(len(top))])
    ax_top.legend(loc="best")
    apply_axis_style(ax_top)

    ax_ratio.bar(
        [f"P{int(p)}" for p in metrics_df["percentile"]],
        metrics_df["underestimation_ratio"] * 100.0,
        color="#D55E00",
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax_ratio.set_ylim(0, 100)
    ax_ratio.set_ylabel("Underestimation ratio (%)")
    apply_axis_style(ax_ratio)

    label_panels(axes.ravel())
    fig.tight_layout()
    paths = save_figure(fig, output_dir, figure_name)
    close_figure(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate peak storm-surge analysis figures.")
    parser.add_argument("--prediction-csv", "--forecast-csv", dest="prediction_csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--percentiles", type=float, nargs="+", default=list(DEFAULT_PERCENTILES))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--figure-name", default="figure_peak_analysis")
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_peak_analysis(
        args.prediction_csv,
        args.output_dir,
        percentiles=tuple(args.percentiles),
        top_n=args.top_n,
        figure_name=args.figure_name,
        language=args.language,
    )


if __name__ == "__main__":
    main()

