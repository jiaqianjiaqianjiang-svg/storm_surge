"""Collect and plot one-step metrics from the formal Xiamen model runs."""

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


MODULE_ROOT = Path(__file__).resolve().parents[2]
MODEL_ORDER = (
    "persistence",
    "ridge",
    "surge_mlp",
    "era5_cnn",
    "cnn",
    "cnn_lstm",
    "cnn_gru",
    "tcn",
    "transformer",
)
DISPLAY_NAMES = {
    "persistence": "Persistence",
    "ridge": "Ridge",
    "surge_mlp": "Surge MLP",
    "era5_cnn": "ERA5 CNN",
    "cnn": "CNN",
    "cnn_lstm": "CNN-LSTM",
    "cnn_gru": "CNN-GRU",
    "tcn": "TCN",
    "transformer": "Transformer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="xiamen")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    return parser.parse_args()


def _overall(details: dict[str, Any]) -> dict[str, Any]:
    overall = dict(details.get("overall", {}))
    if "skill_score_vs_ridge" in details:
        overall["skill_score_vs_ridge"] = details["skill_score_vs_ridge"]
    return overall


def collect_model_metrics(
    model_root: Path,
    baseline_dir: Path,
    split: str = "validation",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_path = baseline_dir / "baseline_metrics.json"
    if baseline_path.is_file():
        report = json.loads(baseline_path.read_text(encoding="utf-8"))
        for name in ("persistence", "ridge"):
            details = report.get("metrics", {}).get(split, {}).get(name)
            if details:
                rows.append({"model": name, **_overall(details)})
    else:
        warnings.warn(f"Missing baseline metrics: {baseline_path}")

    for name in MODEL_ORDER[2:]:
        path = model_root / name / "metrics.json"
        if name == "cnn" and not path.is_file():
            legacy = model_root / "dual" / "metrics.json"
            if legacy.is_file():
                path = legacy
        if not path.is_file():
            warnings.warn(f"Missing model metrics: {path}")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        details = report.get(split)
        if not details:
            warnings.warn(f"Missing {split} metrics in {path}")
            continue
        rows.append({"model": name, **_overall(details)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    order = {name: index for index, name in enumerate(MODEL_ORDER)}
    frame["_order"] = frame.model.map(order)
    return frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def plot_model_comparison(frame: pd.DataFrame, destination: Path, split: str) -> None:
    if frame.empty:
        raise ValueError("No model metrics are available for comparison")
    labels = [DISPLAY_NAMES[name] for name in frame.model]
    x = np.arange(len(frame))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = (
        ("pearson_r", "Pearson r", "#4477AA"),
        ("rmse_cm", "RMSE (cm)", "#CC6677"),
        ("mae_cm", "MAE (cm)", "#228833"),
        ("bias_cm", "Bias (cm)", "#AA3377"),
    )
    for label, ax, (column, ylabel, color) in zip("abcd", axes.flat, panels):
        values = frame[column].to_numpy(dtype=float)
        bars = ax.bar(x, values, color=color, width=0.72)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.22)
        ax.text(0.01, 0.98, f"({label})", transform=ax.transAxes, va="top")
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                offset = 3 if value >= 0 else -12
                ax.annotate(
                    f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                )
    fig.suptitle(f"Xiamen one-hour forecast: {split} set", fontsize=12)
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_root = args.model_root or (
        MODULE_ROOT / "models" / args.station / "formal_seed42"
    )
    baseline_dir = args.baseline_dir or (
        MODULE_ROOT / "models" / args.station / "baselines"
    )
    output = args.output_dir or (
        MODULE_ROOT / "outputs" / "experiments" / args.station / "model_comparison"
    )
    output.mkdir(parents=True, exist_ok=True)
    frame = collect_model_metrics(model_root, baseline_dir, args.split)
    frame.to_csv(output / f"{args.split}_model_metrics.csv", index=False)
    plot_model_comparison(
        frame, output / f"{args.split}_model_comparison.png", args.split
    )
    print(frame.to_string(index=False))
    print(f"outputs: {output}")


if __name__ == "__main__":
    main()
