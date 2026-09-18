"""Fine-tune a temporal Xiamen model with recursive multi-step loss."""

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

from .forecast_model import model_from_checkpoint
from .rolling_diagnostics import atmospheric_validity, resolve_checkpoint_path
from .train_rollout_cnn import rollout_origins, set_seed
from .train_xiamen import amp_context, load_prepared, make_grad_scaler


MODULE_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_MODELS = ("cnn_gru", "cnn_lstm", "tcn", "transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="xiamen")
    parser.add_argument("--model-type", choices=SUPPORTED_MODELS, default="cnn_gru")
    parser.add_argument("--train-start-year", type=int, default=1970)
    parser.add_argument("--train-end-year", type=int, default=1995)
    parser.add_argument("--validation-year", type=int, default=1996)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--teacher-forcing-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


class TemporalRolloutDataset(Dataset):
    """Return pre-encoded weather windows for every recursive target."""

    def __init__(
        self,
        origins: np.ndarray,
        hourly_embeddings: np.ndarray,
        surge: np.ndarray,
        surge_mean: float,
        surge_scale: float,
        input_steps: int,
        rollout_steps: int,
    ) -> None:
        self.origins = origins
        self.embeddings = hourly_embeddings
        self.surge = surge
        self.mean = surge_mean
        self.scale = surge_scale
        self.input_steps = input_steps
        self.steps = rollout_steps

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        origin = int(self.origins[item])
        weather = np.stack(
            [
                self.embeddings[
                    origin + step - self.input_steps : origin + step
                ]
                for step in range(self.steps)
            ]
        ).astype(np.float32, copy=False)
        history = np.asarray(
            self.surge[origin - self.input_steps : origin], dtype=np.float32
        )
        targets = np.asarray(
            self.surge[origin : origin + self.steps], dtype=np.float32
        )
        history = (history - self.mean) / self.scale
        targets = (targets - self.mean) / self.scale
        return (
            torch.from_numpy(weather.copy()),
            torch.from_numpy(history),
            torch.from_numpy(targets),
        )


def precompute_hourly_embeddings(
    model: nn.Module,
    atmosphere: np.ndarray,
    positions: np.ndarray,
    checkpoint: dict[str, Any],
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    if not hasattr(model, "encoder") or not hasattr(model, "forward_encoded"):
        raise TypeError("The checkpoint is not a supported temporal forecast model")
    dimension = int(model.encoder.head[1].out_features)
    embeddings = np.full((len(atmosphere), dimension), np.nan, dtype=np.float32)
    mean = np.asarray(
        checkpoint["scalers"]["atmosphere"]["mean"], dtype=np.float32
    ).reshape(1, -1, 1, 1)
    scale = np.asarray(
        checkpoint["scalers"]["atmosphere"]["scale"], dtype=np.float32
    ).reshape(1, -1, 1, 1)
    model.encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start : start + batch_size]
            raw = np.asarray(atmosphere[batch_positions], dtype=np.float32)
            weather = torch.from_numpy((raw - mean) / scale).to(
                device, non_blocking=True
            )
            with amp_context(use_amp):
                encoded = model.encoder.head(model.encoder.network(weather))
            embeddings[batch_positions] = encoded.float().cpu().numpy()
            print(
                f"hourly weather embeddings "
                f"{min(start + batch_size, len(positions))}/{len(positions)}",
                flush=True,
            )
    return embeddings


def temporal_rollout_forward(
    model: nn.Module,
    weather: torch.Tensor,
    history: torch.Tensor,
    targets: torch.Tensor,
    teacher_ratio: float,
) -> torch.Tensor:
    """Unroll with each model's own predictions when teacher_ratio is zero."""
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
                use_truth = (
                    torch.rand(len(prediction), device=prediction.device)
                    < teacher_ratio
                )
                next_value = torch.where(use_truth, targets[:, step], prediction)
            current = torch.cat([current[:, 1:], next_value[:, None]], dim=1)
    return torch.stack(outputs, dim=1)


def scheduled_teacher_ratio(epoch: int, decay_epochs: int) -> float:
    if epoch < 1 or decay_epochs < 1:
        raise ValueError("epoch and decay_epochs must be positive")
    return max(0.0, 1.0 - (epoch - 1) / max(1, decay_epochs - 1))


def evaluate_recursive_mse(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    non_blocking: bool,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for weather, history, targets in loader:
            weather = weather.to(device, non_blocking=non_blocking)
            history = history.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            with amp_context(use_amp):
                predicted = temporal_rollout_forward(
                    model, weather, history, targets, 0.0
                )
                loss = criterion(predicted, targets)
            total += float(loss) * len(targets)
            count += len(targets)
    return total / count


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dataset_path = args.dataset_path or (
        MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    )
    checkpoint_root = MODULE_ROOT / "models" / args.station / "formal_seed42"
    base_path = args.base_checkpoint or resolve_checkpoint_path(
        checkpoint_root, args.model_type
    )
    output = args.output_dir or (
        checkpoint_root / f"{args.model_type}_rollout{args.rollout_steps}"
    )
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = device.type == "cuda" and not args.no_amp

    atmosphere, surge, times = load_prepared(
        dataset_path, args.train_start_year, args.validation_year
    )
    if (times.year > args.validation_year).any():
        raise AssertionError("Test-year data entered rollout training")
    checkpoint = torch.load(base_path, map_location=device, weights_only=False)
    model = model_from_checkpoint(checkpoint).to(device)
    if not hasattr(model, "forward_encoded"):
        raise TypeError(f"{args.model_type} does not support encoded rollout training")
    input_steps = int(checkpoint["input_steps"])
    valid = atmospheric_validity(atmosphere)
    train_origins = rollout_origins(
        times,
        valid,
        surge,
        set(range(args.train_start_year, args.train_end_year + 1)),
        input_steps,
        args.rollout_steps,
    )
    validation_origins = rollout_origins(
        times,
        valid,
        surge,
        {args.validation_year},
        input_steps,
        args.rollout_steps,
    )
    if not len(train_origins) or not len(validation_origins):
        raise ValueError("No valid rollout samples were found")
    if args.smoke_test:
        train_origins = train_origins[:512]
        validation_origins = validation_origins[:256]
        args.epochs = min(args.epochs, 2)

    first_position = int(min(train_origins.min(), validation_origins.min())) - input_steps
    last_position = int(max(train_origins.max(), validation_origins.max())) + args.rollout_steps - 1
    positions = np.arange(first_position, last_position, dtype=np.int64)
    embeddings = precompute_hourly_embeddings(
        model,
        atmosphere,
        positions,
        checkpoint,
        device,
        args.embedding_batch_size,
        use_amp,
    )
    surge_mean = float(checkpoint["scalers"]["surge"]["mean"][0])
    surge_scale = float(checkpoint["scalers"]["surge"]["scale"][0])
    datasets = [
        TemporalRolloutDataset(
            origins,
            embeddings,
            surge,
            surge_mean,
            surge_scale,
            input_steps,
            args.rollout_steps,
        )
        for origins in (train_origins, validation_origins)
    ]
    pin_memory = device.type == "cuda"
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(datasets[0], shuffle=True, **loader_options)
    validation_loader = DataLoader(datasets[1], **loader_options)
    model.encoder.eval()
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    criterion = nn.MSELoss()
    grad_scaler = make_grad_scaler(use_amp)
    base_validation_mse = evaluate_recursive_mse(
        model,
        validation_loader,
        criterion,
        device,
        use_amp,
        pin_memory,
    )
    base_validation_rmse_cm = float(
        np.sqrt(base_validation_mse) * surge_scale * 100
    )
    best = base_validation_mse
    best_epoch = 0
    stale = 0
    rows: list[dict[str, float]] = [
        {
            "epoch": 0.0,
            "teacher_forcing_ratio": 0.0,
            "train_scaled_mse": float("nan"),
            "validation_recursive_scaled_mse": base_validation_mse,
            "validation_recursive_rmse_cm": base_validation_rmse_cm,
        }
    ]
    best_path = output / "best_model.pth"

    def save_best_checkpoint(fine_tuned: bool) -> None:
        torch.save(
            {
                **checkpoint,
                "model_state_dict": model.state_dict(),
                "rollout_training": {
                    "base_model_type": args.model_type,
                    "rollout_steps": args.rollout_steps,
                    "teacher_forcing_schedule": (
                        f"linear 1 to 0 over {args.teacher_forcing_epochs} epochs"
                    ),
                    "frozen_atmosphere_encoder": True,
                    "selection_metric": (
                        f"{args.validation_year} recursive rollout MSE"
                    ),
                    "base_validation_scaled_mse": base_validation_mse,
                    "best_validation_scaled_mse": best,
                    "best_epoch": best_epoch,
                    "fine_tuned_improved_over_base": fine_tuned,
                    "train_samples": len(train_origins),
                    "validation_samples": len(validation_origins),
                    "seed": args.seed,
                },
            },
            best_path,
        )

    save_best_checkpoint(fine_tuned=False)
    print(
        f"model={args.model_type} device={device} amp={use_amp} "
        f"train_origins={len(train_origins)} validation_origins={len(validation_origins)}",
        flush=True,
    )
    print(
        f"base_recursive_validation_rmse_cm={base_validation_rmse_cm:.4f}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.encoder.eval()
        teacher_ratio = scheduled_teacher_ratio(
            epoch, args.teacher_forcing_epochs
        )
        train_total = 0.0
        train_count = 0
        for weather, history, targets in train_loader:
            weather = weather.to(device, non_blocking=pin_memory)
            history = history.to(device, non_blocking=pin_memory)
            targets = targets.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(use_amp):
                predicted = temporal_rollout_forward(
                    model, weather, history, targets, teacher_ratio
                )
                loss = criterion(predicted, targets)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_total += float(loss.detach()) * len(targets)
            train_count += len(targets)

        validation_mse = evaluate_recursive_mse(
            model,
            validation_loader,
            criterion,
            device,
            use_amp,
            pin_memory,
        )
        row = {
            "epoch": float(epoch),
            "teacher_forcing_ratio": teacher_ratio,
            "train_scaled_mse": train_total / train_count,
            "validation_recursive_scaled_mse": validation_mse,
            "validation_recursive_rmse_cm": (
                np.sqrt(validation_mse) * surge_scale * 100
            ),
        }
        rows.append(row)
        print(row, flush=True)
        if validation_mse < best - 1e-8:
            best = validation_mse
            best_epoch = epoch
            stale = 0
            save_best_checkpoint(fine_tuned=True)
        else:
            if teacher_ratio <= 0:
                stale += 1
            else:
                stale = 0
            if teacher_ratio <= 0 and stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "loss_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(frame.epoch, frame.validation_recursive_rmse_cm, marker="o")
    ax.set(
        xlabel="Epoch",
        ylabel=f"{args.validation_year} recursive RMSE (cm)",
        title=f"{args.model_type} {args.rollout_steps}-step rollout fine-tuning",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "loss_curve.png", dpi=300)
    plt.close(fig)
    metadata = {
        "test_year_loaded": False,
        "base_model_type": args.model_type,
        "base_checkpoint": str(base_path),
        "device": str(device),
        "amp": use_amp,
        "rollout_steps": args.rollout_steps,
        "teacher_forcing_epochs": args.teacher_forcing_epochs,
        "train_samples": len(train_origins),
        "validation_samples": len(validation_origins),
        "base_validation_scaled_mse": base_validation_mse,
        "base_validation_rmse_cm": base_validation_rmse_cm,
        "best_validation_scaled_mse": best,
        "best_validation_rmse_cm": float(np.sqrt(best) * surge_scale * 100),
        "best_epoch": best_epoch,
        "fine_tuned_improved_over_base": best_epoch > 0,
    }
    (output / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    print(f"checkpoint: {best_path}", flush=True)


if __name__ == "__main__":
    main()
