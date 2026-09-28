"""Fine-tune Prickly Bay CNN-GRU with a six-step recursive loss."""

from __future__ import annotations

import argparse
import json
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
    from .train_rollout_dual import rollout_origins, set_seed
    from .train_station import amp_context, load_prepared, make_grad_scaler
except ImportError:
    from forecast_model import model_from_checkpoint
    from rolling_diagnostics import atmospheric_validity
    from train_rollout_dual import rollout_origins, set_seed
    from train_station import amp_context, load_prepared, make_grad_scaler


MODULE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--train-start-year", type=int, default=2011)
    parser.add_argument("--train-end-year", type=int, default=2016)
    parser.add_argument("--validation-year", type=int, default=2017)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--teacher-forcing-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


class TemporalRolloutDataset(Dataset):
    def __init__(
        self, origins: np.ndarray, embeddings: np.ndarray, surge: np.ndarray,
        surge_mean: float, surge_scale: float, input_steps: int,
        rollout_steps: int,
    ) -> None:
        self.origins = origins
        self.embeddings = embeddings
        self.surge = surge
        self.mean = surge_mean
        self.scale = surge_scale
        self.input_steps = input_steps
        self.steps = rollout_steps

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, item: int):
        origin = int(self.origins[item])
        weather = np.stack([
            self.embeddings[origin + step - self.input_steps:origin + step]
            for step in range(self.steps)
        ]).astype(np.float32, copy=False)
        history = np.asarray(
            self.surge[origin - self.input_steps:origin], dtype=np.float32
        )
        targets = np.asarray(
            self.surge[origin:origin + self.steps], dtype=np.float32
        )
        return (
            torch.from_numpy(weather.copy()),
            torch.from_numpy((history - self.mean) / self.scale),
            torch.from_numpy((targets - self.mean) / self.scale),
        )


def precompute_hourly_embeddings(
    model: nn.Module, atmosphere: np.ndarray, positions: np.ndarray,
    checkpoint: dict[str, Any], device: torch.device, batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    if not hasattr(model, "encoder") or not hasattr(model, "forward_encoded"):
        raise TypeError("Checkpoint is not a temporal CNN-GRU model")
    dimension = int(model.encoder.head[1].out_features)
    embeddings = np.full((len(atmosphere), dimension), np.nan, dtype=np.float32)
    mean = np.asarray(checkpoint["scalers"]["atmosphere"]["mean"], dtype=np.float32).reshape(1, -1, 1, 1)
    scale = np.asarray(checkpoint["scalers"]["atmosphere"]["scale"], dtype=np.float32).reshape(1, -1, 1, 1)
    model.encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            selected = positions[start:start + batch_size]
            raw = np.asarray(atmosphere[selected], dtype=np.float32)
            weather = torch.from_numpy((raw - mean) / scale).to(device)
            with amp_context(use_amp):
                encoded = model.encoder.head(model.encoder.network(weather))
            embeddings[selected] = encoded.float().cpu().numpy()
            print(
                f"hourly weather embeddings {min(start + batch_size, len(positions))}/"
                f"{len(positions)}", flush=True,
            )
    return embeddings


def temporal_rollout_forward(
    model: nn.Module, weather: torch.Tensor, history: torch.Tensor,
    targets: torch.Tensor, teacher_ratio: float,
) -> torch.Tensor:
    outputs = []
    current = history
    for step in range(weather.shape[1]):
        prediction = model.forward_encoded(weather[:, step], current)
        outputs.append(prediction)
        if step + 1 < weather.shape[1]:
            if teacher_ratio <= 0:
                next_value = prediction
            elif teacher_ratio >= 1:
                next_value = targets[:, step]
            else:
                use_truth = torch.rand(len(prediction), device=prediction.device) < teacher_ratio
                next_value = torch.where(use_truth, targets[:, step], prediction)
            current = torch.cat([current[:, 1:], next_value[:, None]], dim=1)
    return torch.stack(outputs, dim=1)


def scheduled_teacher_ratio(epoch: int, decay_epochs: int) -> float:
    return max(0.0, 1.0 - (epoch - 1) / max(1, decay_epochs - 1))


def evaluate_recursive_mse(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    device: torch.device, use_amp: bool,
) -> float:
    model.eval(); total = 0.0; count = 0
    with torch.inference_mode():
        for weather, history, targets in loader:
            weather, history, targets = weather.to(device), history.to(device), targets.to(device)
            with amp_context(use_amp):
                loss = criterion(
                    temporal_rollout_forward(model, weather, history, targets, 0.0),
                    targets,
                )
            total += float(loss) * len(targets); count += len(targets)
    return total / count


def main() -> None:
    args = parse_args(); set_seed(args.seed)
    dataset_path = args.dataset_path or MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    base_path = args.base_checkpoint or MODULE_ROOT / "models" / args.station / "formal_seed42" / "cnn_gru" / "best_model.pth"
    output = args.output_dir or MODULE_ROOT / "models" / args.station / "formal_seed42" / f"cnn_gru_rollout{args.rollout_steps}"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_amp = device.type == "cuda" and not args.no_amp
    atmosphere, surge, times = load_prepared(
        dataset_path, args.train_start_year, args.validation_year
    )
    if (times.year > args.validation_year).any():
        raise AssertionError("Independent-test data entered rollout training")
    checkpoint = torch.load(base_path, map_location=device, weights_only=False)
    model = model_from_checkpoint(checkpoint).to(device)
    input_steps = int(checkpoint["input_steps"])
    valid = atmospheric_validity(atmosphere)
    train_origins = rollout_origins(
        times, valid, surge, set(range(args.train_start_year, args.train_end_year + 1)),
        args.rollout_steps,
    )
    validation_origins = rollout_origins(
        times, valid, surge, {args.validation_year}, args.rollout_steps
    )
    if args.smoke_test:
        train_origins = train_origins[:512]; validation_origins = validation_origins[:256]
        args.epochs = min(args.epochs, 2)
    if not len(train_origins) or not len(validation_origins):
        raise ValueError("No valid rollout samples were found")
    first = int(min(train_origins.min(), validation_origins.min())) - input_steps
    last = int(max(train_origins.max(), validation_origins.max())) + args.rollout_steps
    embeddings = precompute_hourly_embeddings(
        model, atmosphere, np.arange(first, last), checkpoint, device,
        args.embedding_batch_size, use_amp,
    )
    surge_mean = float(checkpoint["scalers"]["surge"]["mean"][0])
    surge_scale = float(checkpoint["scalers"]["surge"]["scale"][0])
    train_data, validation_data = [
        TemporalRolloutDataset(
            origins, embeddings, surge, surge_mean, surge_scale,
            input_steps, args.rollout_steps,
        ) for origins in (train_origins, validation_origins)
    ]
    loader_args = {
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_data, **loader_args)
    model.encoder.eval()
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=1e-5,
    )
    criterion = nn.MSELoss(); scaler = make_grad_scaler(use_amp)
    base_mse = evaluate_recursive_mse(model, validation_loader, criterion, device, use_amp)
    best = base_mse; best_epoch = 0; stale = 0
    rows = [{
        "epoch": 0, "teacher_forcing_ratio": 0.0,
        "train_scaled_mse": np.nan,
        "validation_recursive_scaled_mse": base_mse,
        "validation_recursive_rmse_cm": np.sqrt(base_mse) * surge_scale * 100,
    }]
    best_path = output / "best_model.pth"

    def save_best(improved: bool) -> None:
        torch.save({
            **checkpoint, "model_state_dict": model.state_dict(),
            "rollout_training": {
                "base_model_type": "cnn_gru", "rollout_steps": args.rollout_steps,
                "teacher_forcing_schedule": f"linear 1 to 0 over {args.teacher_forcing_epochs} epochs",
                "frozen_atmosphere_encoder": True,
                "selection_metric": f"{args.validation_year} recursive rollout MSE",
                "base_validation_scaled_mse": base_mse,
                "best_validation_scaled_mse": best, "best_epoch": best_epoch,
                "fine_tuned_improved_over_base": improved,
                "train_samples": len(train_origins),
                "validation_samples": len(validation_origins), "seed": args.seed,
            },
        }, best_path)

    save_best(False)
    print(f"base_recursive_validation_rmse_cm={rows[0]['validation_recursive_rmse_cm']:.4f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); model.encoder.eval()
        ratio = scheduled_teacher_ratio(epoch, args.teacher_forcing_epochs)
        total = 0.0; count = 0
        for weather, history, targets in train_loader:
            weather, history, targets = weather.to(device), history.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(use_amp):
                predicted = temporal_rollout_forward(model, weather, history, targets, ratio)
                loss = criterion(predicted, targets)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += float(loss.detach()) * len(targets); count += len(targets)
        validation_mse = evaluate_recursive_mse(
            model, validation_loader, criterion, device, use_amp
        )
        row = {
            "epoch": epoch, "teacher_forcing_ratio": ratio,
            "train_scaled_mse": total / count,
            "validation_recursive_scaled_mse": validation_mse,
            "validation_recursive_rmse_cm": np.sqrt(validation_mse) * surge_scale * 100,
        }
        rows.append(row); print(row, flush=True)
        if validation_mse < best - 1e-8:
            best = validation_mse; best_epoch = epoch; stale = 0; save_best(True)
        elif ratio <= 0:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True); break
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "loss_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(frame.epoch, frame.validation_recursive_rmse_cm, marker="o")
    ax.set(xlabel="Epoch", ylabel="2017 recursive RMSE (cm)", title="CNN-GRU 6-step rollout fine-tuning")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "loss_curve.png", dpi=300); plt.close(fig)
    metadata = {
        "independent_test_loaded": False, "base_checkpoint": str(base_path),
        "device": str(device), "amp": use_amp, "rollout_steps": args.rollout_steps,
        "train_samples": len(train_origins), "validation_samples": len(validation_origins),
        "base_validation_rmse_cm": float(np.sqrt(base_mse) * surge_scale * 100),
        "best_validation_rmse_cm": float(np.sqrt(best) * surge_scale * 100),
        "best_epoch": best_epoch,
    }
    (output / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2)); print(f"checkpoint: {best_path}")


if __name__ == "__main__":
    main()
