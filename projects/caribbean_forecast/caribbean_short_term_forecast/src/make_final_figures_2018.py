"""Create publication-style Prickly Bay figures from existing 2018 results."""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parents[2]
LEADS = (1, 3, 6, 12, 24)
ROLLING_LEADS = (1, 3, 6, 12, 24, 48, 72)
COLORS = {
    "persistence": "#E69F00",
    "combined_ridge": "#4D4D4D",
    "ridge": "#4D4D4D",
    "xgboost": "#0072B2",
    "mlp": "#CC79A7",
    "gru": "#009E73",
    "dual_cnn_past": "#56B4E9",
    "dual_cnn_future": "#8C6BB1",
    "dual_cnn": "#D55E00",
    "dual_cnn_rollout_trained": "#009E73",
}
LABELS = {
    "persistence": "Persistence",
    "combined_ridge": "Combined-Ridge",
    "ridge": "Ridge",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "gru": "GRU",
    "dual_cnn_past": "Dual-CNN-Past",
    "dual_cnn_future": "Dual-CNN-Future",
    "dual_cnn": "Dual-CNN",
    "dual_cnn_rollout_trained": "Dual-CNN rollout-6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--evaluation-year", type=int, default=2018)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direct-dir", type=Path)
    parser.add_argument("--rolling-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.linewidth": 0.8,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.65,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def save(fig: plt.Figure, output: Path, dpi: int) -> None:
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_labels(axes: object) -> None:
    for index, ax in enumerate(np.asarray(axes).reshape(-1)):
        ax.text(-0.12, 1.04, f"({chr(97 + index)})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom")


def figure_direct_selected(direct: pd.DataFrame, output: Path, dpi: int) -> None:
    models = [name for name in direct.columns if name not in {"lead_hours", "valid_samples"}]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(direct))
    width = 0.82 / len(models)
    for index, model in enumerate(models):
        ax.bar(
            x - 0.41 + width / 2 + index * width,
            direct[model], width=width, color=COLORS.get(model, "#777777"),
            label=LABELS.get(model, model),
        )
    ax.set_xticks(x, [f"{int(value)} h" for value in direct.lead_hours])
    ax.set(xlabel="Forecast lead", ylabel="RMSE (cm)")
    ax.grid(axis="y")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, output, dpi)


def figure_direct_curve(metrics: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model in metrics.model.unique():
        subset = metrics[metrics.model == model].sort_values("lead_hours")
        ax.plot(subset.lead_hours, subset.rmse_cm, linewidth=1.7,
                color=COLORS.get(model, "#777777"), label=LABELS.get(model, model))
    ax.set(xlabel="Forecast lead (h)", ylabel="RMSE (cm)")
    ax.set_xticks([1, 3, 6, 12, 18, 24])
    ax.grid()
    ax.legend(ncol=3, frameon=False, loc="best")
    fig.tight_layout()
    save(fig, output, dpi)


def figure_rolling(rolling: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method in rolling.method.unique():
        subset = rolling[rolling.method == method].sort_values("lead_hours")
        ax.plot(subset.lead_hours, subset.rmse_cm, marker="o", markersize=3.5,
                linewidth=1.7, color=COLORS.get(method, "#777777"),
                label=LABELS.get(method, method))
    ax.set(xlabel="Lead time (h)", ylabel="RMSE (cm)")
    ax.set_xticks(ROLLING_LEADS)
    ax.grid()
    ax.legend(ncol=2, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, output, dpi)


def figure_rollout_improvement(rolling: pd.DataFrame, output: Path, dpi: int) -> None:
    base = rolling[rolling.method == "dual_cnn"].set_index("lead_hours")
    tuned = rolling[rolling.method == "dual_cnn_rollout_trained"].set_index("lead_hours")
    leads = np.asarray(ROLLING_LEADS)
    base_rmse = base.loc[leads, "rmse_cm"].to_numpy()
    tuned_rmse = tuned.loc[leads, "rmse_cm"].to_numpy()
    improvement = 100 * (base_rmse - tuned_rmse) / base_rmse
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    axes[0].plot(leads, base_rmse, marker="o", color=COLORS["dual_cnn"], label="Dual-CNN")
    axes[0].plot(leads, tuned_rmse, marker="o", color=COLORS["dual_cnn_rollout_trained"], label="Rollout-6")
    axes[0].set(xlabel="Lead time (h)", ylabel="RMSE (cm)", xticks=ROLLING_LEADS)
    axes[0].grid(); axes[0].legend(frameon=False)
    bars = axes[1].bar(leads.astype(str), improvement, color=COLORS["dual_cnn_rollout_trained"])
    axes[1].set(xlabel="Lead time (h)", ylabel="RMSE reduction (%)")
    axes[1].grid(axis="y")
    for bar, value in zip(bars, improvement):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.1f}%",
                     ha="center", va="bottom", fontsize=7)
    panel_labels(axes)
    fig.tight_layout()
    save(fig, output, dpi)


def figure_extremes(rolling: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for method in rolling.method.unique():
        subset = rolling[rolling.method == method].sort_values("lead_hours")
        color = COLORS.get(method, "#777777")
        label = LABELS.get(method, method)
        axes[0].plot(subset.lead_hours, subset.top5_rmse_cm, marker="o", markersize=3,
                     color=color, label=label)
        axes[1].plot(subset.lead_hours, subset.rapid_rise_rmse_cm, marker="o", markersize=3,
                     color=color, label=label)
    axes[0].set(xlabel="Lead time (h)", ylabel="Top 5% RMSE (cm)", xticks=ROLLING_LEADS)
    axes[1].set(xlabel="Lead time (h)", ylabel="Rapid-rise RMSE (cm)", xticks=ROLLING_LEADS)
    for ax in axes: ax.grid()
    axes[0].legend(ncol=2, frameon=False, fontsize=7)
    panel_labels(axes)
    fig.tight_layout()
    save(fig, output, dpi)


def figure_scatter(predictions: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.2))
    for ax, lead in zip(axes.flat, (1, 6, 12, 24)):
        observed = predictions[f"observed_{lead:02d}h_m"].to_numpy(float) * 100
        predicted = predictions[f"combined_ridge_{lead:02d}h_m"].to_numpy(float) * 100
        valid = np.isfinite(observed) & np.isfinite(predicted)
        observed, predicted = observed[valid], predicted[valid]
        bounds = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
        ax.hexbin(observed, predicted, gridsize=42, mincnt=1, cmap="Blues")
        ax.plot(bounds, bounds, color="black", linestyle="--", linewidth=1)
        rmse = np.sqrt(np.mean((predicted - observed) ** 2))
        correlation = np.corrcoef(observed, predicted)[0, 1]
        ax.text(0.04, 0.95, f"r = {correlation:.3f}\nRMSE = {rmse:.2f} cm\nn = {len(observed)}",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set(xlabel="Observed (cm)", ylabel="Predicted (cm)", title=f"Lead {lead} h")
        ax.grid(alpha=0.25)
    panel_labels(axes)
    fig.tight_layout()
    save(fig, output, dpi)


def autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    centered = values - np.mean(values)
    denominator = np.dot(centered, centered)
    return np.asarray([
        np.dot(centered[:-lag], centered[lag:]) / denominator
        if lag < len(centered) and denominator > 0 else np.nan
        for lag in range(1, max_lag + 1)
    ])


def figure_residuals(predictions: pd.DataFrame, output: Path, stats_output: Path, dpi: int) -> None:
    observed = predictions["observed_01h_m"].to_numpy(float) * 100
    predicted = predictions["combined_ridge_01h_m"].to_numpy(float) * 100
    residual = predicted - observed
    valid = np.isfinite(residual)
    residual = residual[valid]
    times = pd.to_datetime(predictions.loc[valid, "valid_time_01h"])
    acf = autocorrelation(residual, 72)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.8))
    axes[0, 0].plot(times, residual, color=COLORS["combined_ridge"], linewidth=0.6)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set(xlabel="Valid time", ylabel="Residual (cm)")
    axes[0, 1].hist(residual, bins=35, density=True, color="#8FBBD9", edgecolor="white")
    if len(residual) > 2 and np.std(residual) > 0:
        grid = np.linspace(residual.min(), residual.max(), 250)
        axes[0, 1].plot(grid, stats.gaussian_kde(residual)(grid), color="#0072B2")
    axes[0, 1].set(xlabel="Residual (cm)", ylabel="Density")
    stats.probplot(residual, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("")
    axes[1, 0].set(xlabel="Theoretical quantiles", ylabel="Ordered residuals (cm)")
    axes[1, 1].bar(np.arange(1, 73), acf, color="#56B4E9", width=0.8)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set(xlabel="Lag (h)", ylabel="Autocorrelation", xlim=(0, 73))
    axes[1, 1].xaxis.set_major_locator(MaxNLocator(7, integer=True))
    for ax in axes.flat: ax.grid(alpha=0.35)
    panel_labels(axes)
    fig.tight_layout()
    save(fig, output, dpi)
    statistics = {
        "n": int(len(residual)),
        "mean_error_cm": float(np.mean(residual)),
        "median_error_cm": float(np.median(residual)),
        "standard_deviation_cm": float(np.std(residual, ddof=1)),
        "skewness": float(stats.skew(residual, bias=False)),
        "kurtosis": float(stats.kurtosis(residual, bias=False)),
        "lag_1_autocorrelation": float(acf[0]),
        "lag_24_autocorrelation": float(acf[23]),
    }
    stats_output.write_text(json.dumps(statistics, indent=2), encoding="utf-8")


def copy_events(rolling_dir: Path, output: Path) -> list[str]:
    names: list[str] = []
    for index, source in enumerate(sorted(rolling_dir.glob("strong_event_*.png")), start=1):
        destination = output / f"fig08_strong_event_{index}.png"
        shutil.copy2(source, destination)
        names.append(destination.name)
    return names


def main() -> None:
    args = parse_args()
    apply_style()
    experiments = MODULE_ROOT / "outputs" / "experiments" / args.station
    direct_dir = args.direct_dir or experiments / f"direct_multimodel_{args.evaluation_year}_final_seed{args.seed}"
    rolling_dir = args.rolling_dir or experiments / f"rolling_72_with_rollout6_{args.evaluation_year}_seed{args.seed}"
    output = args.output_dir or experiments / f"journal_figures_{args.evaluation_year}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    direct_metrics = pd.read_csv(direct_dir / "metrics_all_leads.csv")
    direct_selected = pd.read_csv(direct_dir / "rmse_primary_leads.csv")
    direct_predictions = pd.read_csv(direct_dir / "test_predictions.csv")
    rolling_metrics = pd.read_csv(rolling_dir / "metrics_selected_leads.csv")
    figures = []
    jobs = [
        ("fig01_direct_selected_leads.png", lambda p: figure_direct_selected(direct_selected, p, args.dpi)),
        ("fig02_direct_rmse_by_lead.png", lambda p: figure_direct_curve(direct_metrics, p, args.dpi)),
        ("fig03_rolling_rmse_by_lead.png", lambda p: figure_rolling(rolling_metrics, p, args.dpi)),
        ("fig04_rollout_improvement.png", lambda p: figure_rollout_improvement(rolling_metrics, p, args.dpi)),
        ("fig05_extreme_metrics.png", lambda p: figure_extremes(rolling_metrics, p, args.dpi)),
        ("fig06_direct_scatter.png", lambda p: figure_scatter(direct_predictions, p, args.dpi)),
        ("fig07_residual_diagnostics.png", lambda p: figure_residuals(direct_predictions, p, output / "residual_statistics.json", args.dpi)),
    ]
    for name, job in jobs:
        destination = output / name
        try:
            job(destination)
            figures.append(name)
            print(f"created: {destination}")
        except Exception as error:
            warnings.warn(f"Skipped {name}: {error}")
    figures.extend(copy_events(rolling_dir, output))
    pd.DataFrame({"figure_name": figures, "status": "created"}).to_csv(
        output / "figure_manifest.csv", index=False
    )
    package = REPOSITORY_ROOT / "reports" / "experiment_results" / f"{args.station}_short_term_{args.evaluation_year}_seed{args.seed}" / "journal_figures"
    package.mkdir(parents=True, exist_ok=True)
    for path in output.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".csv", ".json"}:
            shutil.copy2(path, package / path.name)
    print(f"journal figures: {output}")
    print(f"Git-safe figure package: {package}")


if __name__ == "__main__":
    main()
