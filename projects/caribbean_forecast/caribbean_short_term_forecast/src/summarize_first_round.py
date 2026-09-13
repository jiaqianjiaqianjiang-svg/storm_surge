"""Create a compact comparison table for baselines and seed-42 neural models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]


def _row(model: str, split: str, metrics: dict[str, Any]) -> dict[str, Any]:
    overall = metrics["overall"]
    peaks = metrics["peak_events"]
    return {
        "model": model,
        "split": split,
        "n": overall["n"],
        "rmse_cm": overall["rmse_cm"],
        "mae_cm": overall["mae_cm"],
        "pearson_r": overall["pearson_r"],
        "r2": overall["r2"],
        "bias_cm": overall["bias_cm"],
        "skill_vs_ridge": metrics.get("skill_score_vs_ridge"),
        "top10_abs_rmse_cm": metrics["top_absolute_10_percent"]["rmse_cm"],
        "top5_abs_rmse_cm": metrics["top_absolute_5_percent"]["rmse_cm"],
        "rapid_rise_rmse_cm": metrics["rapid_rise_top_10_percent"]["rmse_cm"],
        "mean_abs_peak_error_cm": peaks["mean_absolute_peak_error_cm"],
        "max_peak_error_cm": peaks["maximum_peak_error_cm"],
        "mean_abs_timing_error_h": peaks["mean_absolute_timing_error_hours"],
        "underestimated_max_abs_peak": peaks["underestimated_maximum_absolute_peak"],
    }


def summarise(
    baseline_metrics_path: str | Path,
    formal_root: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    baseline = json.loads(Path(baseline_metrics_path).read_text(encoding="utf-8"))
    rows = []
    for split in ("validation", "test"):
        for model in ("zero", "persistence", "ridge"):
            rows.append(_row(model, split, baseline["metrics"][split][model]))
    for model in ("surge_mlp", "era5_cnn", "dual"):
        metrics_path = Path(formal_root) / model / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for split in ("validation", "test"):
            rows.append(_row(model, split, metrics[split]))
    frame = pd.DataFrame(rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


def plot_comparison(frame: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_order = ["zero", "persistence", "ridge", "surge_mlp", "era5_cnn", "dual"]
    labels = ["Zero", "Persistence", "Ridge", "Surge MLP", "ERA5 CNN", "Dual"]
    x = np.arange(len(model_order))
    validation = frame.set_index(["model", "split"]).loc[
        [(model, "validation") for model in model_order], "rmse_cm"
    ].to_numpy()
    test = frame.set_index(["model", "split"]).loc[
        [(model, "test") for model in model_order], "rmse_cm"
    ].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.38
    ax.bar(x - width / 2, validation, width, label="2017 validation")
    ax.bar(x + width / 2, test, width, label="2018 test")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("RMSE (cm)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    rmse_path = destination / "seed42_overall_rmse_comparison.png"
    fig.savefig(rmse_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    test_frame = frame.loc[frame.split == "test"].set_index("model").loc[model_order]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    ax.bar(x - width, test_frame.rmse_cm, width, label="All hours")
    ax.bar(x, test_frame.top5_abs_rmse_cm, width, label="Top 5% |surge|")
    ax.bar(x + width, test_frame.rapid_rise_rmse_cm, width, label="Rapid rise")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("2018 RMSE (cm)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    extreme_path = destination / "seed42_test_extreme_rmse_comparison.png"
    fig.savefig(extreme_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return [rmse_path, extreme_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=MODULE_ROOT / "models" / "prickly_bay" / "baselines" / "baseline_metrics.json",
    )
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=MODULE_ROOT / "models" / "prickly_bay" / "formal_seed42",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODULE_ROOT / "outputs" / "experiments" / "prickly_bay"
        / "seed42_model_comparison.csv",
    )
    parser.add_argument("--plot-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = summarise(arguments.baseline_metrics, arguments.formal_root, arguments.output)
    plot_comparison(result, arguments.plot_dir or arguments.output.parent)
    print(result.to_string(index=False))
