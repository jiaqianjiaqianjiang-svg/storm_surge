"""Evaluate locked one-step checkpoints without retraining or refitting them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset_builder import SchemeBDataset, Standardisation, valid_targets
from .evaluate import calculate_detailed_metrics
from .forecast_model import MODEL_TYPES, model_from_checkpoint
from .plotting import observed_vs_predicted, validation_scatter
from .rolling_diagnostics import resolve_checkpoint_path
from .station_config import (
    STATIONS,
    get_station_config,
    validate_dataset_identity,
)
from .train_xiamen import amp_context, load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_MODELS = tuple(
    name for name in MODEL_TYPES if name != "dual"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", choices=tuple(STATIONS), default="xiamen")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=EVALUATION_MODELS,
        default=list(EVALUATION_MODELS),
    )
    parser.add_argument("--year", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _scalers(checkpoint: dict[str, object]) -> dict[str, Standardisation]:
    states = checkpoint["scalers"]
    return {
        "atmosphere": Standardisation.from_state_dict(
            states["atmosphere"], (1, 3, 1, 1)
        ),
        "surge": Standardisation.from_state_dict(states["surge"], (1,)),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    model_type: str,
    station_id: str,
    atmosphere: np.ndarray,
    surge: np.ndarray,
    times: pd.DatetimeIndex,
    evaluation_year: int,
    split: str,
    baseline_dir: Path,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> dict[str, object]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    checkpoint_station = checkpoint.get("station_id")
    if checkpoint_station not in (None, station_id):
        raise ValueError(
            f"Checkpoint belongs to {checkpoint_station}, not {station_id}: "
            f"{checkpoint_path}"
        )
    input_steps = int(checkpoint["input_steps"])
    targets, _ = valid_targets(times, atmosphere, surge, input_steps)
    targets = [target for target in targets if times[target].year == evaluation_year]
    if not targets:
        raise ValueError(
            f"No valid {evaluation_year} one-step targets for {station_id}"
        )
    dataset = SchemeBDataset(
        atmosphere,
        surge,
        times,
        targets,
        input_steps,
        _scalers(checkpoint),
        model_type,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model = model_from_checkpoint(checkpoint).to(device).eval()
    scale = float(checkpoint["scalers"]["surge"]["scale"][0])
    mean = float(checkpoint["scalers"]["surge"]["mean"][0])
    predicted_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for weather, history, _ in loader:
            weather = weather.to(device, non_blocking=device.type == "cuda")
            history = history.to(device, non_blocking=device.type == "cuda")
            with amp_context(use_amp):
                predicted_batches.append(model(weather, history).float().cpu().numpy())
    predicted = np.concatenate(predicted_batches) * scale + mean
    observed = np.asarray(surge[targets], dtype=np.float32)
    dates = times[targets]
    reference_path = baseline_dir / f"{split}_baseline_predictions.csv"
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Run evaluate_baselines for split '{split}' first: {reference_path}"
        )
    reference = pd.read_csv(reference_path, parse_dates=["datetime"])
    ridge = (
        reference.set_index("datetime")
        .reindex(dates)
        .ridge_m.to_numpy(dtype=float)
    )
    if not np.isfinite(ridge).all():
        raise ValueError(
            f"Ridge baseline does not cover every {split} checkpoint timestamp"
        )
    metrics = calculate_detailed_metrics(observed, predicted, dates, ridge)
    destination = checkpoint_path.parent
    pd.DataFrame(
        {
            "datetime": dates,
            "observed_m": observed,
            "predicted_m": predicted,
        }
    ).to_csv(destination / f"{split}_predictions.csv", index=False)
    observed_vs_predicted(
        dates,
        observed,
        predicted,
        destination / f"{split}_observed_vs_predicted.png",
    )
    validation_scatter(
        observed, predicted, destination / f"{split}_scatter.png"
    )
    metrics_path = destination / "metrics.json"
    all_metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else {}
    )
    all_metrics[split] = metrics
    metrics_path.write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    args = parse_args()
    station = get_station_config(args.station)
    configured_year = (
        station.validation_year if args.split == "validation" else station.test_year
    )
    evaluation_year = configured_year if args.year is None else args.year
    if evaluation_year != configured_year:
        raise ValueError(
            f"{station.name} {args.split} split is fixed to {configured_year}"
        )
    dataset_path = args.dataset_path or station.dataset_dir
    validate_dataset_identity(dataset_path, station.station_id)
    checkpoint_root = args.checkpoint_root or (
        station.model_root / f"formal_seed{args.seed}"
    )
    baseline_dir = args.baseline_dir or station.model_root / "baselines"
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_amp = device.type == "cuda" and not args.no_amp
    atmosphere, surge, times = load_prepared(
        dataset_path, evaluation_year - 1, evaluation_year
    )
    if (times.year > evaluation_year).any():
        raise AssertionError("Future-year data entered checkpoint evaluation")
    results: dict[str, object] = {}
    for model_type in args.models:
        checkpoint_path = resolve_checkpoint_path(checkpoint_root, model_type)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing locked {model_type} checkpoint: {checkpoint_path}"
            )
        results[model_type] = evaluate_checkpoint(
            checkpoint_path,
            model_type,
            station.station_id,
            atmosphere,
            surge,
            times,
            evaluation_year,
            args.split,
            baseline_dir,
            args.batch_size,
            device,
            use_amp,
        )
        print(
            f"{model_type}: RMSE={results[model_type]['overall']['rmse_cm']:.4f} cm",
            flush=True,
        )


if __name__ == "__main__":
    main()
