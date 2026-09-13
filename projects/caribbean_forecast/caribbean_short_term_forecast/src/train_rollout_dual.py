"""Fine-tune a one-step Dual-CNN for stable recursive forecasts via 6-step unrolling."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .forecast_model import model_from_checkpoint
    from .rolling_diagnostics import atmospheric_validity
    from .train_station import load_prepared
except ImportError:
    from forecast_model import model_from_checkpoint
    from rolling_diagnostics import atmospheric_validity
    from train_station import load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rollout_origins(
    times: pd.DatetimeIndex,
    atmosphere_valid: np.ndarray,
    surge: np.ndarray,
    years: set[int],
    rollout_steps: int,
) -> np.ndarray:
    values = np.asarray(surge, dtype=float)
    origins = []
    for origin in range(24, len(times) - rollout_steps + 1):
        if times[origin].year not in years or times[origin + rollout_steps - 1].year not in years:
            continue
        if not np.all(np.diff(times[origin - 24:origin + rollout_steps].values) == np.timedelta64(1, "h")):
            continue
        if not atmosphere_valid[origin - 24:origin + rollout_steps - 1].all():
            continue
        if not np.isfinite(values[origin - 24:origin + rollout_steps]).all():
            continue
        origins.append(origin)
    return np.asarray(origins, dtype=np.int64)


class RolloutDataset(Dataset):
    def __init__(
        self,
        origins: np.ndarray,
        embeddings: np.ndarray,
        surge: np.ndarray,
        surge_mean: float,
        surge_scale: float,
        rollout_steps: int,
    ) -> None:
        self.origins = origins
        self.embeddings = embeddings
        self.surge = surge
        self.mean = surge_mean
        self.scale = surge_scale
        self.steps = rollout_steps

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        origin = int(self.origins[item])
        weather = self.embeddings[origin:origin + self.steps]
        history = (np.asarray(self.surge[origin - 24:origin], dtype=np.float32) - self.mean) / self.scale
        targets = (np.asarray(self.surge[origin:origin + self.steps], dtype=np.float32) - self.mean) / self.scale
        return torch.from_numpy(weather.copy()), torch.from_numpy(history), torch.from_numpy(targets)


def precompute_weather_embeddings(
    model: nn.Module,
    atmosphere: np.ndarray,
    positions: np.ndarray,
    checkpoint: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dimension = int(model.atmosphere_head[1].out_features)
    embeddings = np.full((len(atmosphere), dimension), np.nan, dtype=np.float32)
    mean = np.asarray(checkpoint["scalers"]["atmosphere"]["mean"], dtype=np.float32).reshape(1, 1, -1, 1, 1)
    scale = np.asarray(checkpoint["scalers"]["atmosphere"]["scale"], dtype=np.float32).reshape(1, 1, -1, 1, 1)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start:start + batch_size]
            raw = np.stack([np.asarray(atmosphere[p - 24:p], dtype=np.float32) for p in batch_positions])
            scaled = (raw - mean) / scale
            weather = torch.from_numpy(scaled.reshape(len(batch_positions), -1, 40, 40)).to(device)
            encoded = model.atmosphere_head(model.atmosphere_branch(weather))
            embeddings[batch_positions] = encoded.float().cpu().numpy()
        print(f"weather embeddings complete: {len(positions)} target hours", flush=True)
    return embeddings


def rollout_forward(
    model: nn.Module,
    weather: torch.Tensor,
    history: torch.Tensor,
    targets: torch.Tensor,
    teacher_ratio: float,
) -> torch.Tensor:
    outputs = []
    current = history
    for step in range(weather.shape[1]):
        history_features = model.surge_branch(current)
        predicted = model.fusion(torch.cat([weather[:, step], history_features], dim=1)).squeeze(-1)
        outputs.append(predicted)
        if step + 1 < weather.shape[1]:
            if teacher_ratio <= 0:
                next_value = predicted
            elif teacher_ratio >= 1:
                next_value = targets[:, step]
            else:
                mask = torch.rand(len(predicted), device=predicted.device) < teacher_ratio
                next_value = torch.where(mask, targets[:, step], predicted)
            current = torch.cat([current[:, 1:], next_value[:, None]], dim=1)
    return torch.stack(outputs, dim=1)


def main() -> None:
    args = parse_args(); set_seed(args.seed)
    dataset_path = args.dataset_path or MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    base_path = args.base_checkpoint or MODULE_ROOT / "models" / args.station / "formal_seed42" / "dual" / "best_model.pth"
    output = args.output_dir or MODULE_ROOT / "models" / args.station / "rollout6_seed42"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    atmosphere, surge, times = load_prepared(dataset_path, 2011, 2017)
    if times.max().year != 2017:
        raise AssertionError("2018 entered rollout training")
    checkpoint = torch.load(base_path, map_location=device, weights_only=False)
    model = model_from_checkpoint(checkpoint).to(device)
    valid = atmospheric_validity(atmosphere)
    train_origins = rollout_origins(times, valid, surge, set(range(2011, 2017)), args.rollout_steps)
    validation_origins = rollout_origins(times, valid, surge, {2017}, args.rollout_steps)
    if args.smoke_test:
        train_origins = train_origins[:512]; validation_origins = validation_origins[:256]
    positions = np.unique(np.concatenate([
        train_origins[:, None] + np.arange(args.rollout_steps),
        validation_origins[:, None] + np.arange(args.rollout_steps),
    ]))
    embeddings = precompute_weather_embeddings(model, atmosphere, positions, checkpoint, device, args.embedding_batch_size)
    surge_mean = float(checkpoint["scalers"]["surge"]["mean"][0])
    surge_scale = float(checkpoint["scalers"]["surge"]["scale"][0])
    train_data = RolloutDataset(train_origins, embeddings, surge, surge_mean, surge_scale, args.rollout_steps)
    validation_data = RolloutDataset(validation_origins, embeddings, surge, surge_mean, surge_scale, args.rollout_steps)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)
    for parameter in model.atmosphere_branch.parameters(): parameter.requires_grad = False
    for parameter in model.atmosphere_head.parameters(): parameter.requires_grad = False
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=1e-5
    )
    criterion = nn.MSELoss()
    best = float("inf"); stale = 0; history_rows = []
    best_path = output / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        model.train(); model.atmosphere_branch.eval(); model.atmosphere_head.eval()
        teacher_ratio = max(0.0, 1.0 - (epoch - 1) / max(1, args.epochs - 1))
        total = count = 0
        for weather, history, targets in train_loader:
            weather, history, targets = weather.to(device), history.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = rollout_forward(model, weather, history, targets, teacher_ratio)
            loss = criterion(predicted, targets)
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(targets); count += len(targets)
        model.eval(); validation_total = validation_count = 0
        with torch.inference_mode():
            for weather, history, targets in validation_loader:
                weather, history, targets = weather.to(device), history.to(device), targets.to(device)
                predicted = rollout_forward(model, weather, history, targets, 0.0)
                loss = criterion(predicted, targets)
                validation_total += float(loss) * len(targets); validation_count += len(targets)
        validation_mse = validation_total / validation_count
        row = {
            "epoch": epoch, "teacher_forcing_ratio": teacher_ratio,
            "train_scaled_mse": total / count,
            "validation_recursive_scaled_mse": validation_mse,
            "validation_recursive_rmse_cm": np.sqrt(validation_mse) * surge_scale * 100,
        }
        history_rows.append(row); print(row, flush=True)
        if validation_mse < best - 1e-8:
            best = validation_mse; stale = 0
            torch.save({
                **checkpoint,
                "model_state_dict": model.state_dict(),
                "rollout_training": {
                    "rollout_steps": args.rollout_steps,
                    "teacher_forcing_schedule": "linear 1 to 0",
                    "frozen_atmosphere_encoder": True,
                    "selection_metric": "2017 recursive rollout MSE",
                    "train_samples": len(train_origins),
                    "validation_samples": len(validation_origins),
                    "seed": args.seed,
                },
            }, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True); break
    pd.DataFrame(history_rows).to_csv(output / "loss_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8)); frame = pd.DataFrame(history_rows)
    ax.plot(frame.epoch, frame.validation_recursive_rmse_cm, marker="o")
    ax.set(xlabel="Epoch", ylabel="2017 recursive RMSE (cm)", title=f"{args.rollout_steps}-step rollout fine-tuning")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "loss_curve.png", dpi=300); plt.close(fig)
    metadata = {
        "2018_loaded": False, "base_checkpoint": str(base_path), "device": str(device),
        "rollout_steps": args.rollout_steps, "train_samples": len(train_origins),
        "validation_samples": len(validation_origins), "best_validation_scaled_mse": best,
    }
    (output / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"checkpoint: {best_path}")


if __name__ == "__main__":
    main()
