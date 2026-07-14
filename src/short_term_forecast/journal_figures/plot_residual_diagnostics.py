"""Residual diagnostics for forecast predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import io_utils
from .style import apply_axis_style, close_figure, format_time_axis, label_panels, save_figure, setup_journal_style


def _skewness(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan")
    centered = values - np.mean(values)
    std = np.std(centered)
    return float(np.mean((centered / std) ** 3)) if std > 0 else float("nan")


def _kurtosis(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 4:
        return float("nan")
    centered = values - np.mean(values)
    std = np.std(centered)
    return float(np.mean((centered / std) ** 4) - 3.0) if std > 0 else float("nan")


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    values = values[np.isfinite(values)]
    if lag <= 0 or values.size <= lag:
        return float("nan")
    x = values[:-lag]
    y = values[lag:]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def residual_statistics(residual: np.ndarray) -> dict[str, float]:
    residual = residual[np.isfinite(residual)]
    return {
        "mean_error": float(np.mean(residual)) if residual.size else float("nan"),
        "median_error": float(np.median(residual)) if residual.size else float("nan"),
        "standard_deviation": float(np.std(residual, ddof=1)) if residual.size > 1 else float("nan"),
        "skewness": _skewness(residual),
        "kurtosis": _kurtosis(residual),
        "lag_1_autocorrelation": _autocorrelation(residual, 1),
        "lag_24_autocorrelation": _autocorrelation(residual, 24),
        "n": int(residual.size),
    }


def _normal_density(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if not np.isfinite(std) or std <= 0:
        return np.zeros_like(x)
    return 1.0 / (std * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def plot_residual_diagnostics(
    prediction_csv: Path,
    output_dir: Path,
    figure_name: str = "figure_residual_diagnostics",
    language: str = "en",
) -> dict[str, Path] | None:
    setup_journal_style(language=language)
    df, _ = io_utils.load_prediction_csv(prediction_csv)
    if df is None:
        io_utils.warn(f"Cannot build residual diagnostics from {prediction_csv}")
        return None

    residual = (df["predicted"] - df["observed"]).to_numpy(float)
    stats = residual_statistics(residual)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "residual_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    dates = pd.to_datetime(df["datetime"], errors="coerce")
    if dates.isna().all():
        x_values = np.arange(1, len(df) + 1)
        x_label = "Sample"
    else:
        x_values = dates
        x_label = "Time"

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    ax_ts, ax_hist, ax_qq, ax_acf = axes.ravel()

    ax_ts.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_ts.plot(x_values, residual, color="#0072B2", linewidth=1.1)
    ax_ts.set_xlabel(x_label)
    ax_ts.set_ylabel("Residual (cm)")
    apply_axis_style(ax_ts)
    if not isinstance(x_values, np.ndarray):
        format_time_axis(ax_ts)

    finite = residual[np.isfinite(residual)]
    ax_hist.hist(finite, bins=min(25, max(6, int(np.sqrt(max(len(finite), 1))))), density=True, color="#BDBDBD", edgecolor="#333333")
    if finite.size:
        xs = np.linspace(float(np.min(finite)), float(np.max(finite)), 200)
        ax_hist.plot(xs, _normal_density(xs, float(np.mean(finite)), float(np.std(finite))), color="#D55E00", linewidth=1.3)
    ax_hist.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax_hist.set_xlabel("Residual (cm)")
    ax_hist.set_ylabel("Density")
    apply_axis_style(ax_hist)

    if finite.size >= 2:
        sorted_resid = np.sort(finite)
        probs = (np.arange(1, finite.size + 1) - 0.5) / finite.size
        normal = NormalDist()
        theoretical = np.asarray([normal.inv_cdf(float(p)) for p in probs])
        theoretical = theoretical * np.std(finite) + np.mean(finite)
        ax_qq.scatter(theoretical, sorted_resid, s=14, color="#0072B2", alpha=0.8)
        low = float(np.nanmin([theoretical.min(), sorted_resid.min()]))
        high = float(np.nanmax([theoretical.max(), sorted_resid.max()]))
        ax_qq.plot([low, high], [low, high], color="black", linestyle="--", linewidth=0.9)
    ax_qq.set_xlabel("Theoretical quantiles (cm)")
    ax_qq.set_ylabel("Sample quantiles (cm)")
    apply_axis_style(ax_qq)

    lags = np.arange(1, 73)
    acf = np.asarray([_autocorrelation(residual, int(lag)) for lag in lags])
    ax_acf.axhline(0, color="black", linewidth=0.8)
    ax_acf.bar(lags, acf, color="#6A3D9A", edgecolor="none", width=0.8)
    ax_acf.set_xlabel("Lag (h)")
    ax_acf.set_ylabel("Autocorrelation")
    ax_acf.set_xlim(0, 73)
    ax_acf.set_ylim(-1, 1)
    apply_axis_style(ax_acf)

    label_panels(axes.ravel())
    fig.tight_layout()
    paths = save_figure(fig, output_dir, figure_name)
    close_figure(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate residual diagnostic journal figures.")
    parser.add_argument("--prediction-csv", "--forecast-csv", dest="prediction_csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-name", default="figure_residual_diagnostics")
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_residual_diagnostics(args.prediction_csv, args.output_dir, args.figure_name, args.language)


if __name__ == "__main__":
    main()

