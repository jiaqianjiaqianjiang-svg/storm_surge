"""训练厦门站短时风暴潮预报模型。

当前默认实现 hourly forecast：
使用前 t 个小时 ERA5 的 U10、V10、MSL 40×40 网格，
再加前 t 个小时 storm surge，预测下一小时 storm surge。

建议 ERA5 小时级实验先用 --frequency hourly --input-steps 24，
即“前 24 小时预测下一小时”。ERA20C 原始为 3 小时，可用 --frequency 3hourly。
daily 频率仍保留，用于前期流程验证和 daily maximum surge 对比。

示例：
python src/train_forecast_xiamen.py --data-source ERA5 --start-year 1985 --end-year 1997 --input-steps 8 --epochs 50
python src/train_forecast_xiamen.py --data-source ERA20C --start-year 1985 --end-year 1997 --input-steps 16 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config
from forecast_cnn_model import ForecastCNN
from forecast_dataset import build_forecast_arrays, chronological_split, standardize_train_val


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except OSError:
                pass


configure_console_encoding()


class ForecastArrayDataset(Dataset):
    """内存数组版 Dataset，适合厦门站 daily forecast 小到中等规模实验。"""

    def __init__(self, atmosphere: np.ndarray, surge_history: np.ndarray, target: np.ndarray) -> None:
        self.atmosphere = atmosphere
        self.surge_history = surge_history
        self.target = target

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.atmosphere[index].astype("float32", copy=False)),
            torch.from_numpy(self.surge_history[index].astype("float32", copy=False)),
            torch.tensor(float(self.target[index]), dtype=torch.float32),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练厦门站短时风暴潮 forecast CNN。")
    parser.add_argument("--data-source", choices=["ERA5", "ERA20C"], default=config.FORECAST_DATA_SOURCE)
    parser.add_argument("--frequency", choices=["hourly", "3hourly", "daily"], default=config.FORECAST_FREQUENCY)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--input-steps", type=int, default=config.FORECAST_INPUT_STEPS)
    parser.add_argument("--horizon", type=int, default=config.FORECAST_HORIZON)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="显式保存 X_atmosphere.npy、X_surge_history.npy、y.npy；小时级数据很大，默认不保存。",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境没有可用 CUDA")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    for atmosphere, surge_history, target in loader:
        atmosphere = atmosphere.to(device)
        surge_history = surge_history.to(device)
        target = target.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pred = model(atmosphere, surge_history)
            loss = criterion(pred, target)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * int(target.shape[0])
        total_count += int(target.shape[0])
    return total_loss / max(total_count, 1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for atmosphere, surge_history, _ in loader:
            out = model(atmosphere.to(device), surge_history.to(device)).cpu().numpy()
            preds.append(out)
    return np.concatenate(preds, axis=0)


def compute_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    mae = float(np.mean(np.abs(pred - obs)))
    denominator = float(np.sqrt(np.mean(obs**2)))
    rrmse = float(rmse / denominator) if denominator > 0 else float("nan")
    if len(obs) >= 2 and np.std(obs) > 0 and np.std(pred) > 0:
        r = float(pearsonr(obs, pred).statistic)
    else:
        r = float("nan")
    return {"pearson_r": r, "rmse": rmse, "mae": mae, "rrmse": rrmse}


def plot_loss_curve(history: list[dict[str, float]], path: Path) -> None:
    """保存训练集和验证集 MSE loss 曲线。"""

    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = [item["epoch"] for item in history]
    ax.plot(epochs, [item["train_loss"] for item in history], label="train")
    ax.plot(epochs, [item["val_loss"] for item in history], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_validation_timeseries(dates: pd.DatetimeIndex, obs: np.ndarray, pred: np.ndarray, path: Path) -> None:
    """保存验证集 observed vs predicted 折线图，便于直接放入 PPT。"""

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(dates, obs, label="observed", linewidth=1.5)
    ax.plot(dates, pred, label="predicted", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Storm surge")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year 不能晚于 --end-year")
    if args.horizon != 1:
        raise ValueError("当前方案B短时滚动预报训练阶段请使用 horizon=1，多步预报请用 rolling_forecast.py")
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[TRAIN] 使用设备: {device}")

    run_name = (
        f"{args.data_source}_{args.start_year}_{args.end_year}"
        f"_{args.frequency}_t{args.input_steps}_h{args.horizon}"
    )
    output_dir = config.FORECAST_OUTPUT_ROOT / run_name
    model_dir = config.FORECAST_MODEL_ROOT / run_name
    figure_dir = config.FORECAST_FIGURE_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    x_atm, x_surge, y, dates_np, target_dates = build_forecast_arrays(
        data_source=args.data_source,
        start_year=args.start_year,
        end_year=args.end_year,
        input_steps=args.input_steps,
        horizon=args.horizon,
        frequency=args.frequency,
    )
    train_idx, val_idx = chronological_split(len(y), args.train_ratio)
    print(f"[DATASET] X_atmosphere shape: {x_atm.shape}")
    print(f"[DATASET] X_surge_history shape: {x_surge.shape}")
    print(f"[DATASET] y shape: {y.shape}")
    print(f"[DATASET] train size: {len(train_idx):,}")
    print(f"[DATASET] val size: {len(val_idx):,}")
    x_atm_std, x_surge_std, y_std, scalers = standardize_train_val(x_atm, x_surge, y, train_idx)

    if args.save_arrays:
        np.save(output_dir / "X_atmosphere.npy", x_atm_std)
        np.save(output_dir / "X_surge_history.npy", x_surge_std)
        np.save(output_dir / "y.npy", y_std)
        np.save(output_dir / "dates.npy", dates_np)
        print(f"[OUTPUT] 已保存标准化数组: {output_dir}")
    else:
        print("[OUTPUT] 默认不保存 X_atmosphere/X_surge_history/y 大数组；如需保存请加 --save-arrays")
    (output_dir / "scalers.json").write_text(json.dumps(scalers, ensure_ascii=False, indent=2), encoding="utf-8")

    train_ds = ForecastArrayDataset(x_atm_std[train_idx], x_surge_std[train_idx], y_std[train_idx])
    val_ds = ForecastArrayDataset(x_atm_std[val_idx], x_surge_std[val_idx], y_std[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ForecastCNN(input_steps=args.input_steps, n_variables=len(config.VARIABLES), grid_size=config.GRID_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = make_optimizer(args, model)
    best_val = float("inf")
    best_path = model_dir / "best_forecast_cnn.pth"
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[TRAIN] epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "scalers": scalers,
                    "variables": config.VARIABLES,
                    "grid_size": config.GRID_SIZE,
                },
                best_path,
            )
            print(f"[TRAIN] 保存 best model: {best_path}")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_pred_std = predict(model, val_loader, device)
    target_mean = scalers["target_mean"]
    target_std = scalers["target_std"]
    val_pred = val_pred_std * target_std + target_mean
    val_obs = y[val_idx]

    metrics = compute_metrics(val_obs, val_pred)
    metrics.update({"best_val_loss": best_val, "n_train": int(len(train_idx)), "n_val": int(len(val_idx))})
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    val_dates = pd.DatetimeIndex(target_dates[val_idx])
    pd.DataFrame(
        {
            "datetime": val_dates.strftime("%Y-%m-%d %H:%M:%S"),
            "observed": val_obs,
            "predicted": val_pred,
        }
    ).to_csv(output_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(output_dir / "loss_history.csv", index=False, encoding="utf-8-sig")
    plot_loss_curve(history, figure_dir / "loss_curve.png")
    plot_validation_timeseries(val_dates, val_obs, val_pred, figure_dir / "val_observed_vs_predicted.png")

    print("[METRICS] " + json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OUTPUT] 验证集预测: {output_dir / 'val_predictions.csv'}")
    print(f"[OUTPUT] 指标: {output_dir / 'metrics.json'}")
    print(f"[OUTPUT] loss 曲线: {figure_dir / 'loss_curve.png'}")
    print(f"[OUTPUT] 验证集 observed vs predicted 图: {figure_dir / 'val_observed_vs_predicted.png'}")


if __name__ == "__main__":
    main()
