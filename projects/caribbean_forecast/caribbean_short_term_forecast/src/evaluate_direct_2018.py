"""Evaluate locked direct 24-hour models on the 2018 independent test year."""

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
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .direct_forecast_models import (
        ArrayDirectDataset,
        DirectDualCNN,
        DirectGRU,
        DirectMLP,
        GridDirectDataset,
    )
    from .direct_multistep_baselines import (
        PRIMARY_LEADS,
        build_feature_matrix,
        direct_labels,
        direct_origins,
        hourly_atmosphere_valid,
        predict_selected_models,
        summarise_atmosphere,
    )
    from .train_direct_multimodel import (
        DISPLAY_NAMES,
        MODEL_NAMES,
        build_gru_sequences,
        calculate_all_metrics,
    )
    from .train_station import load_prepared
except ImportError:
    from direct_forecast_models import (
        ArrayDirectDataset,
        DirectDualCNN,
        DirectGRU,
        DirectMLP,
        GridDirectDataset,
    )
    from direct_multistep_baselines import (
        PRIMARY_LEADS,
        build_feature_matrix,
        direct_labels,
        direct_origins,
        hourly_atmosphere_valid,
        predict_selected_models,
        summarise_atmosphere,
    )
    from train_direct_multimodel import (
        DISPLAY_NAMES,
        MODEL_NAMES,
        build_gru_sequences,
        calculate_all_metrics,
    )
    from train_station import load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--evaluation-year", type=int, default=2018)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--development-dir", type=Path)
    parser.add_argument("--ridge-bundle", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grid-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def model_from_checkpoint(checkpoint: dict[str, Any]) -> torch.nn.Module:
    name = checkpoint.get("model_name")
    if name == "DirectMLP":
        return DirectMLP(checkpoint["input_features"], checkpoint.get("output_steps", 24))
    if name == "DirectGRU":
        return DirectGRU(
            checkpoint.get("input_features", 14), checkpoint.get("hidden_size", 96),
            checkpoint.get("num_layers", 2), checkpoint.get("output_steps", 24),
        )
    if name == "DirectDualCNN":
        return DirectDualCNN(
            checkpoint["atmosphere_steps"], checkpoint.get("input_steps", 24),
            checkpoint.get("variables", 3), checkpoint.get("grid_size", 40),
            checkpoint.get("output_steps", 24),
        )
    raise ValueError(f"Unsupported direct checkpoint model_name: {name!r}")


def predict_loader(
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 2:
                values, _ = batch
                predicted = model(values.to(device, non_blocking=True))
            else:
                atmosphere, history, _ = batch
                predicted = model(
                    atmosphere.to(device, non_blocking=True),
                    history.to(device, non_blocking=True),
                )
            batches.append(predicted.float().cpu().numpy())
    scaled = np.concatenate(batches, axis=0)
    return scaled * float(checkpoint["surge_scale"]) + float(checkpoint["surge_mean"])


def plot_rmse(metrics: pd.DataFrame, destination: Path, year: int) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name in MODEL_NAMES:
        subset = metrics[metrics.model == name].sort_values("lead_hours")
        ax.plot(subset.lead_hours, subset.rmse_cm, linewidth=1.7, label=DISPLAY_NAMES[name])
    ax.set(
        xlabel="Forecast lead (h)", ylabel="RMSE (cm)",
        title=f"Prickly Bay {year} independent direct forecast test",
    )
    ax.set_xticks([1, 3, 6, 12, 18, 24])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.evaluation_year <= 2017:
        raise ValueError("Final evaluation must use a year after the 2017 development year")
    dataset = args.dataset_path or MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    development = args.development_dir or MODULE_ROOT / "outputs" / "experiments" / args.station / "direct_multimodel_2017_seed42"
    ridge_path = args.ridge_bundle or MODULE_ROOT / "outputs" / "experiments" / args.station / "direct_multistep_baselines_2017" / "direct_ridge_models.joblib"
    output = args.output_dir or MODULE_ROOT / "outputs" / "experiments" / args.station / f"direct_multimodel_{args.evaluation_year}_final_seed42"
    output.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    atmosphere, surge, times = load_prepared(dataset, 2011, args.evaluation_year)
    if times.max().year != args.evaluation_year:
        raise AssertionError("Prepared data do not reach the requested evaluation year")
    validity = hourly_atmosphere_valid(atmosphere)
    origins = direct_origins(times, validity, surge, {args.evaluation_year})
    if not len(origins):
        raise ValueError(f"No complete direct 24-hour samples found in {args.evaluation_year}")
    labels = direct_labels(surge, origins)
    summaries = summarise_atmosphere(atmosphere)
    features = build_feature_matrix(summaries, surge, origins, True, "past_future")
    predictions: dict[str, np.ndarray] = {
        "persistence": np.repeat(np.asarray(surge[origins])[:, None], 24, axis=1),
    }

    ridge_bundle = joblib.load(required_file(ridge_path, "direct Ridge bundle"))
    ridge = ridge_bundle["models"]["combined_ridge_past_future"]
    predictions["combined_ridge"] = predict_selected_models(
        ridge["scaler"], ridge["models_by_alpha"],
        np.asarray(ridge["selected_alpha_by_lead"]), features,
    )

    xgboost_models = joblib.load(required_file(development / "xgboost_models.joblib", "direct XGBoost models"))
    predictions["xgboost"] = np.column_stack(
        [model.predict(features) for model in xgboost_models]
    ).astype(np.float32)

    dummy = np.zeros_like(labels, dtype=np.float32)
    mlp_checkpoint = torch.load(required_file(development / "mlp" / "best_model.pth", "MLP checkpoint"), map_location="cpu", weights_only=False)
    mlp_features = mlp_checkpoint["feature_scaler"].transform(features).astype(np.float32)
    predictions["mlp"] = predict_loader(
        development / "mlp" / "best_model.pth",
        DataLoader(ArrayDirectDataset(mlp_features, dummy), batch_size=args.batch_size),
        device,
    )

    gru_checkpoint = torch.load(required_file(development / "gru" / "best_model.pth", "GRU checkpoint"), map_location="cpu", weights_only=False)
    gru_features = build_gru_sequences(
        summaries, surge, origins, gru_checkpoint["weather_scaler"],
        float(gru_checkpoint["surge_mean"]), float(gru_checkpoint["surge_scale"]),
    )
    predictions["gru"] = predict_loader(
        development / "gru" / "best_model.pth",
        DataLoader(ArrayDirectDataset(gru_features, dummy), batch_size=args.batch_size),
        device,
    )

    for name, include_future in (("dual_cnn_past", False), ("dual_cnn_future", True)):
        checkpoint_path = required_file(development / name / "best_model.pth", f"{name} checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        scaler = checkpoint["atmosphere_scaler"]
        dataset_view = GridDirectDataset(
            atmosphere, surge, origins, labels,
            np.asarray(scaler["mean"], dtype=np.float32),
            np.asarray(scaler["scale"], dtype=np.float32),
            float(checkpoint["surge_mean"]), float(checkpoint["surge_scale"]),
            include_future,
        )
        predictions[name] = predict_loader(
            checkpoint_path,
            DataLoader(dataset_view, batch_size=args.grid_batch_size, num_workers=0),
            device,
        )

    metrics = calculate_all_metrics(labels, predictions, surge, origins)
    metrics.to_csv(output / "metrics_all_leads.csv", index=False)
    selected = metrics[metrics.lead_hours.isin(PRIMARY_LEADS)].pivot(
        index="lead_hours", columns="model", values="rmse_cm"
    ).reset_index()
    selected.insert(1, "valid_samples", len(origins))
    selected = selected[["lead_hours", "valid_samples", *MODEL_NAMES]]
    selected.to_csv(output / "rmse_primary_leads.csv", index=False)

    data: dict[str, Any] = {"forecast_origin": times[origins]}
    for lead in range(1, 25):
        data[f"valid_time_{lead:02d}h"] = times[origins + lead]
        data[f"observed_{lead:02d}h_m"] = labels[:, lead - 1]
        for name in MODEL_NAMES:
            data[f"{name}_{lead:02d}h_m"] = predictions[name][:, lead - 1]
    pd.DataFrame(data).to_csv(output / "test_predictions.csv", index=False)
    plot_rmse(metrics, output / "rmse_vs_lead.png", args.evaluation_year)
    metadata = {
        "experiment": "locked direct 24-hour independent test",
        "evaluation_year": args.evaluation_year,
        "development_year": 2017,
        "training_years": [2011, 2016],
        "samples": int(len(origins)),
        "models": list(MODEL_NAMES),
        "device": str(device),
        "retrained": False,
        "checkpoint_selection_used_test_data": False,
        "future_era5_definition": "known future ERA5 reanalysis; historical hindcast only",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(selected.to_string(index=False), flush=True)
    print(f"outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
