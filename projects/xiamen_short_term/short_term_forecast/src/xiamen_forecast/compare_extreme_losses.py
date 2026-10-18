"""在共同时间戳上汇总1996年一步极值损失消融实验。"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluate import calculate_detailed_metrics
from .station_config import STATIONS, get_station_config, validate_output_location


MODULE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = (
    ("mse", "cnn_gru", "MSE"),
    ("wmse", "cnn_gru_extreme_wmse", "Extreme-WMSE"),
    ("awmse", "cnn_gru_extreme_awmse", "Extreme-AWMSE"),
    ("whuber", "cnn_gru_extreme_whuber", "Extreme-WHuber"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", choices=tuple(STATIONS), default="lianyungang")
    parser.add_argument("--validation-year", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _load_predictions(path: Path, expected_year: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"datetime", "observed_m", "predicted_m"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[list(required)].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.dropna().drop_duplicates("datetime").sort_values("datetime")
    years = set(frame.datetime.dt.year.unique())
    if years != {expected_year}:
        raise ValueError(
            f"Expected validation year {expected_year} only in {path}, found {sorted(years)}"
        )
    return frame


def collect_common_predictions(
    model_root: Path,
    validation_year: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """读取已有实验，并仅保留各实验完全一致的验证时间。"""

    available: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    for key, directory, _ in EXPERIMENTS:
        path = model_root / directory / "validation_predictions.csv"
        if not path.is_file():
            warnings.warn(f"Missing validation predictions, skipping {key}: {path}")
            continue
        available[key] = _load_predictions(path, validation_year)
        sources[key] = str(path)
    if "mse" not in available:
        raise FileNotFoundError(
            f"Baseline predictions are required: {model_root / 'cnn_gru' / 'validation_predictions.csv'}"
        )
    common = set(available["mse"].datetime)
    for frame in available.values():
        common &= set(frame.datetime)
    common_index = pd.DatetimeIndex(sorted(common))
    if not len(common_index):
        raise ValueError("No common validation timestamps exist across the experiments")
    combined = pd.DataFrame({"datetime": common_index})
    reference_observed: np.ndarray | None = None
    for key, frame in available.items():
        aligned = frame.set_index("datetime").loc[common_index]
        observed = aligned.observed_m.to_numpy(dtype=float)
        if reference_observed is None:
            reference_observed = observed
            combined["observed_m"] = observed
        elif not np.allclose(reference_observed, observed, rtol=0, atol=1e-7):
            raise ValueError(f"Observed targets disagree for experiment {key}")
        combined[f"{key}_predicted_m"] = aligned.predicted_m.to_numpy(dtype=float)
    return combined, sources


def summarise_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = {key: label for key, _, label in EXPERIMENTS}
    observed = frame.observed_m.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(frame.datetime)
    details: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for key, _, _ in EXPERIMENTS:
        column = f"{key}_predicted_m"
        if column not in frame:
            continue
        metrics = calculate_detailed_metrics(
            observed, frame[column].to_numpy(dtype=float), dates
        )
        details[key] = metrics
        overall = metrics["overall"]
        p99 = metrics["positive_extreme_p99"]
        peak = metrics["positive_peak_events"]
        rows.append(
            {
                "experiment": key,
                "display_name": labels[key],
                "common_samples": int(len(frame)),
                "rmse_cm": overall["rmse_cm"],
                "mae_cm": overall["mae_cm"],
                "bias_cm": overall["bias_cm"],
                "pearson_r": overall["pearson_r"],
                "positive_p99_n": p99["n"],
                "positive_p99_threshold_cm": p99.get("threshold_cm", np.nan),
                "positive_p99_rmse_cm": p99.get("rmse_cm", np.nan),
                "positive_p99_bias_cm": p99.get("bias_cm", np.nan),
                "positive_p99_underestimation_ratio": p99.get(
                    "underestimation_ratio", np.nan
                ),
                "positive_p99_mean_underestimation_cm": p99.get(
                    "mean_underestimation_cm", np.nan
                ),
                "positive_p99_mean_overestimation_cm": p99.get(
                    "mean_overestimation_cm", np.nan
                ),
                "positive_event_count": peak.get("event_count", 0),
                "positive_event_peak_rmse_cm": peak.get("peak_height_rmse_cm", np.nan),
                "positive_event_peak_bias_cm": peak.get(
                    "mean_peak_height_bias_cm", np.nan
                ),
                "positive_event_timing_mae_hours": peak.get(
                    "mean_absolute_timing_error_hours", np.nan
                ),
            }
        )
        for event in peak.get("events", []):
            events.append({"experiment": key, "display_name": labels[key], **event})
    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary.experiment == "mse"].iloc[0]
    summary["rmse_change_vs_mse_percent"] = 100 * (
        summary.rmse_cm / baseline.rmse_cm - 1
    )
    summary["positive_p99_rmse_reduction_vs_mse_percent"] = 100 * (
        1 - summary.positive_p99_rmse_cm / baseline.positive_p99_rmse_cm
    )
    summary["practical_overall_rmse_within_5_percent"] = (
        summary.rmse_change_vs_mse_percent <= 5.0
    )
    summary["practical_p99_rmse_reduction_at_least_10_percent"] = (
        summary.positive_p99_rmse_reduction_vs_mse_percent >= 10.0
    )
    details["positive_peak_event_rows"] = events
    return summary, details


def plot_summary(frame: pd.DataFrame, destination: Path, station_name: str) -> None:
    x = np.arange(len(frame))
    labels = frame.display_name.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4))
    panels = (
        ("rmse_cm", "Overall RMSE (cm)"),
        ("positive_p99_rmse_cm", "Positive P99 RMSE (cm)"),
        ("positive_p99_bias_cm", "Positive P99 bias (cm)"),
        ("positive_p99_underestimation_ratio", "P99 underestimation ratio"),
        ("positive_event_peak_rmse_cm", "Positive-event peak RMSE (cm)"),
    )
    colors = ["#4D4D4D", "#2166AC", "#B2182B", "#6A3D9A"][: len(frame)]
    for panel, (ax, (column, ylabel)) in enumerate(zip(axes.flat, panels)):
        values = frame[column].to_numpy(dtype=float)
        bars = ax.bar(x, values, color=colors, width=0.68)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.22)
        ax.text(0.01, 0.98, f"({chr(97 + panel)})", transform=ax.transAxes, va="top")
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3 if value >= 0 else -12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                )
    axes.flat[-1].axis("off")
    fig.suptitle(f"{station_name} 1996 validation: extreme-loss ablation", fontsize=12)
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    station = get_station_config(args.station)
    validation_year = args.validation_year or station.validation_year
    if validation_year != station.validation_year:
        raise ValueError(
            "Extreme-loss method selection is restricted to the configured validation year"
        )
    model_root = args.model_root or (
        MODULE_ROOT / "models" / args.station / f"formal_seed{args.seed}"
    )
    output = args.output_dir or (
        MODULE_ROOT
        / "outputs"
        / "experiments"
        / args.station
        / f"extreme_loss_ablation_{validation_year}_seed{args.seed}"
    )
    validate_output_location(output, station.experiment_root, "Extreme-loss comparison")
    output.mkdir(parents=True, exist_ok=True)
    predictions, sources = collect_common_predictions(model_root, validation_year)
    summary, details = summarise_predictions(predictions)
    summary.to_csv(output / "validation_extreme_loss_summary.csv", index=False)
    pd.DataFrame(details.pop("positive_peak_event_rows")).to_csv(
        output / "validation_positive_peak_events.csv", index=False
    )
    (output / "validation_extreme_loss_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "station": station.station_id,
        "validation_year": validation_year,
        "test_year_loaded": False,
        "common_timestamp_count": len(predictions),
        "sources": sources,
        "selection_targets": {
            "positive_p99_rmse_reduction_percent": 10.0,
            "maximum_overall_rmse_degradation_percent": 5.0,
            "long_lead_rule": "assess separately with rolling_diagnostics",
        },
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_summary(
        summary,
        output / "validation_extreme_loss_comparison.png",
        station.name,
    )
    print(summary.to_string(index=False), flush=True)
    print(f"outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
