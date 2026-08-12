"""Unified 2017 72-hour recursive comparison for four representative models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .evaluate import calculate_metrics
    from .evaluate_baselines import summarise_atmosphere
    from .rolling_24_comparison import recursive_dual_predictions, recursive_tabular_predictions
    from .rolling_diagnostics import atmospheric_validity, find_common_origins
    from .train_station import load_prepared
except ImportError:
    from evaluate import calculate_metrics
    from evaluate_baselines import summarise_atmosphere
    from rolling_24_comparison import recursive_dual_predictions, recursive_tabular_predictions
    from rolling_diagnostics import atmospheric_validity, find_common_origins
    from train_station import load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[1]
LEADS = (1, 3, 6, 12, 24, 48, 72)
METHODS = ("persistence", "ridge", "xgboost", "dual_cnn")
DISPLAY = {
    "persistence": "Persistence",
    "ridge": "Ridge",
    "xgboost": "XGBoost",
    "dual_cnn": "Dual-CNN",
    "dual_cnn_rollout_trained": "Dual-CNN rollout-trained",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--validation-year", type=int, default=2017)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--ridge-model", type=Path)
    parser.add_argument("--xgboost-model", type=Path)
    parser.add_argument("--dual-checkpoint", type=Path)
    parser.add_argument("--rollout-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def build_metrics(
    surge: np.ndarray,
    origins: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lead in LEADS:
        target_positions = origins + lead - 1
        observed = np.asarray(surge[target_positions], dtype=float)
        prior = np.asarray(surge[target_positions - 1], dtype=float)
        threshold = float(np.quantile(np.abs(observed), 0.95))
        top_mask = np.abs(observed) >= threshold
        rises = observed - prior
        positive = rises[rises > 0]
        rise_threshold = float(np.quantile(positive, 0.90)) if len(positive) else np.nan
        rapid_mask = rises >= rise_threshold if np.isfinite(rise_threshold) else np.zeros(len(observed), bool)
        ridge_mse = float(np.mean((predictions["ridge"][:, lead - 1] - observed) ** 2))
        for method in predictions:
            predicted = predictions[method][:, lead - 1]
            overall = calculate_metrics(observed, predicted)
            top = calculate_metrics(observed[top_mask], predicted[top_mask])
            rapid = calculate_metrics(observed[rapid_mask], predicted[rapid_mask])
            mse = float(np.mean((predicted - observed) ** 2))
            rows.append({
                "lead_hours": lead,
                "method": method,
                **overall,
                "skill_vs_same_lead_ridge": 1 - mse / ridge_mse if ridge_mse > 0 else np.nan,
                "top5_n": int(top["n"]),
                "top5_threshold_cm": threshold * 100,
                "top5_rmse_cm": top["rmse_cm"],
                "rapid_rise_n": int(rapid["n"]),
                "rapid_rise_threshold_cm_per_hour": rise_threshold * 100,
                "rapid_rise_rmse_cm": rapid["rmse_cm"],
            })
    return pd.DataFrame(rows)


def plot_metric(metrics: pd.DataFrame, column: str, ylabel: str, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ordered = [method for method in (*METHODS, "dual_cnn_rollout_trained") if method in set(metrics.method)]
    for method in ordered:
        subset = metrics[metrics.method == method]
        ax.plot(subset.lead_hours, subset[column], marker="o", linewidth=2, label=DISPLAY[method])
    ax.axhline(0, color="grey", linewidth=0.8) if column == "bias_cm" else None
    ax.set(xlabel="Lead time (hours)", ylabel=ylabel, xticks=LEADS)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Prickly Bay 2017 unified 72-hour recursive hindcast")
    fig.tight_layout()
    fig.savefig(destination, dpi=400)
    plt.close(fig)


def plot_events(
    output: Path,
    times: pd.DatetimeIndex,
    surge: np.ndarray,
    origins: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    complete = np.asarray([np.isfinite(surge[o:o + 72]).all() for o in origins])
    candidates = np.flatnonzero(complete)
    scores = np.asarray([np.max(np.abs(surge[origins[i]:origins[i] + 72])) for i in candidates])
    selected: list[int] = []
    for position in candidates[np.argsort(scores)[::-1]]:
        if all(abs(int(origins[position]) - int(origins[other])) >= 72 for other in selected):
            selected.append(int(position))
        if len(selected) == 3:
            break
    details = []
    for number, position in enumerate(selected, 1):
        origin = int(origins[position])
        valid_times = times[origin:origin + 72]
        fig, ax = plt.subplots(figsize=(12, 5.8))
        ax.plot(valid_times, surge[origin:origin + 72] * 100, color="black", linewidth=2.3, label="Observed")
        for method in predictions:
            ax.plot(valid_times, predictions[method][position] * 100, linewidth=1.35, label=DISPLAY[method])
        ax.set(xlabel="Valid time", ylabel="Storm surge (cm)")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3)
        ax.set_title(f"72 h recursive hindcast from {times[origin]:%Y-%m-%d %H:%M}\nKnown future ERA5 forcing")
        fig.autofmt_xdate(); fig.tight_layout()
        name = f"strong_event_{number}_{times[origin]:%Y%m%d_%H%M}.png"
        fig.savefig(output / name, dpi=400); plt.close(fig)
        details.append({"forecast_origin": times[origin].isoformat(), "plot": name})
    return details


def main() -> None:
    args = parse_args()
    dataset = args.dataset_path or MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    ridge_path = args.ridge_model or MODULE_ROOT / "models" / args.station / "baselines" / "ridge_pipeline.joblib"
    xgb_path = args.xgboost_model or MODULE_ROOT / "outputs" / "experiments" / args.station / "rolling_24_comparison_2017_seed42" / "xgboost_one_step.joblib"
    dual_path = args.dual_checkpoint or MODULE_ROOT / "models" / args.station / "formal_seed42" / "dual" / "best_model.pth"
    output = args.output_dir or MODULE_ROOT / "outputs" / "experiments" / args.station / "rolling_72_comparison_2017_seed42"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    atmosphere, surge, times = load_prepared(dataset, 2011, args.validation_year)
    if times.max().year != args.validation_year:
        raise AssertionError("Data beyond validation year entered the 72-hour experiment")
    origins = find_common_origins(times, atmospheric_validity(atmosphere), surge, args.validation_year, 24, 72, LEADS)
    histories = np.stack([np.asarray(surge[o - 24:o], dtype=np.float32) for o in origins])
    summaries = summarise_atmosphere(atmosphere)
    ridge = joblib.load(ridge_path)
    xgboost = joblib.load(xgb_path)
    checkpoint = torch.load(dual_path, map_location=device, weights_only=False)
    predictions = {
        "persistence": np.repeat(np.asarray(surge[origins - 1], dtype=np.float32)[:, None], 72, axis=1),
        "ridge": recursive_tabular_predictions(ridge, summaries, histories, origins - 1, output_steps=72),
        "xgboost": recursive_tabular_predictions(xgboost, summaries, histories, origins - 1, output_steps=72),
        "dual_cnn": recursive_dual_predictions(atmosphere, origins - 1, histories, checkpoint, device, args.batch_size, output_steps=72),
    }
    if args.rollout_checkpoint:
        rollout_checkpoint = torch.load(
            args.rollout_checkpoint, map_location=device, weights_only=False
        )
        predictions["dual_cnn_rollout_trained"] = recursive_dual_predictions(
            atmosphere, origins - 1, histories, rollout_checkpoint,
            device, args.batch_size, output_steps=72,
        )
    metrics = build_metrics(surge, origins, predictions)
    metrics.to_csv(output / "metrics_selected_leads.csv", index=False)
    wide = metrics.pivot(index="lead_hours", columns="method", values="rmse_cm").reset_index()
    wide.insert(1, "valid_samples", len(origins))
    wide.to_csv(output / "rmse_table.csv", index=False)
    rows = []
    for lead in LEADS:
        targets = origins + lead - 1
        data = {"forecast_origin": times[origins], "valid_time": times[targets], "lead_hours": lead, "observed_m": surge[targets]}
        data.update({f"{name}_m": values[:, lead - 1] for name, values in predictions.items()})
        rows.append(pd.DataFrame(data))
    pd.concat(rows, ignore_index=True).to_csv(output / "predictions_selected_leads.csv", index=False)
    plot_metric(metrics, "rmse_cm", "RMSE (cm)", output / "rmse_vs_lead.png")
    plot_metric(metrics, "bias_cm", "Bias (cm)", output / "bias_vs_lead.png")
    plot_metric(metrics, "top5_rmse_cm", "Top 5% RMSE (cm)", output / "top5_rmse_vs_lead.png")
    events = plot_events(output, times, surge, origins, predictions)
    metadata = {
        "experiment": "known-future-ERA5 72-hour recursive hindcast",
        "validation_year": args.validation_year,
        "2018_loaded": False,
        "common_origins": int(len(origins)),
        "leads": list(LEADS),
        "models": list(predictions),
        "events": events,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(wide.to_string(index=False))
    print(metrics[["lead_hours", "method", "rmse_cm", "mae_cm", "bias_cm", "pearson_r", "top5_rmse_cm", "rapid_rise_rmse_cm"]].to_string(index=False))
    print(f"outputs: {output}")


if __name__ == "__main__":
    main()
