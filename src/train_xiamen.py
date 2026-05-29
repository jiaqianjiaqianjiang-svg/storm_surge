"""训练厦门站 CNN 风暴潮重建模型。

使用预处理阶段生成的 outputs/xiamen/*.npy 文件训练 5 个不同随机种子的 CNN，
并对验证集进行 5-model averaging ensemble。

示例：
python src/train_xiamen.py --epochs 100 --batch-size 32 --lr 0.001
python src/train_xiamen.py --epochs 2 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Windows + conda 环境中，PyTorch、numpy、matplotlib 可能同时加载不同来源的
# OpenMP 运行库。这里在导入这些数值库之前设置兼容开关，避免训练结束绘图时
# 出现 libiomp5md.dll already initialized 崩溃。若你的环境没有这个问题，该设置
# 不会影响正常运行。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    nn = None
    DataLoader = None
    Dataset = object

from config import PROJECT_ROOT, XIAMEN_OUTPUT_DIR
from metrics import compute_metrics
from plot_results import ensure_figure_dir, plot_loss_curve, plot_pred_vs_obs, plot_scatter


def configure_console_encoding() -> None:
    """避免 Windows 终端输出中文时出现编码错误。"""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except OSError:
                pass


configure_console_encoding()


class NpyStormSurgeDataset(Dataset):
    """从 .npy 文件读取 CNN 输入和标签。

    X 文件使用 mmap_mode='r'，不会一次性把全部样本载入内存；
    每次 __getitem__ 只取一个样本，再转为 torch.Tensor。
    """

    def __init__(self, x_path: Path, y_path: Path) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path)
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"X/y 样本数不一致: {self.x.shape[0]} vs {self.y.shape[0]}")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # mmap_mode='r' 读出的数组是只读视图。PyTorch 会警告 non-writable tensor，
        # 因此这里显式 copy 一份可写数组，避免潜在未定义行为。
        x_array = np.array(self.x[index], dtype=np.float32, copy=True)
        x = torch.from_numpy(x_array)
        y = torch.tensor(self.y[index], dtype=torch.float32)
        return x, y


def parse_args() -> argparse.Namespace:
    """解析训练命令行参数。"""

    parser = argparse.ArgumentParser(description="训练厦门站 CNN storm surge reconstruction 模型。")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数，论文复现实验可先用 100")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="SGD 学习率")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="SGD weight decay")
    parser.add_argument("--num-workers", type=int, default=0, help="Windows 上建议先用 0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="5-model ensemble 随机种子")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="训练设备")
    parser.add_argument(
        "--surge-unit-scale-to-cm",
        type=float,
        default=100.0,
        help=(
            "把预处理输出的原始 storm surge 单位转换为厘米的倍率。"
            "GESLA 常见单位为米，因此默认 100。最终论文对比指标使用厘米计算。"
        ),
    )
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=1e-5, help="Minimum validation loss improvement for early stopping")
    parser.add_argument("--extreme-percentile", type=float, default=95.0, help="Percentile threshold for extreme surge validation")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置随机种子，使每个模型可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(choice: str) -> torch.device:
    """根据命令行选择 CPU 或 GPU。"""

    if torch is None:
        raise ModuleNotFoundError(
            "当前 Python 环境没有安装 PyTorch。请先在 jjq 环境中安装 torch，"
            "或确认已经激活正确的 conda 环境。"
        )
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境没有可用 CUDA")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_y_scaler(output_dir: Path) -> tuple[float, float]:
    """读取预处理阶段保存的 y 标准化参数。"""

    scaler_path = output_dir / "y_scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError(f"缺少 y_scaler.json，请先运行预处理: {scaler_path}")
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    return float(scaler["mean"]), float(scaler["std"])


def inverse_transform_y(y_standardized: np.ndarray, mean: float, std: float) -> np.ndarray:
    """将标准化后的 y 还原到原始 storm surge 单位。"""

    return y_standardized * std + mean


def summarize_array_file(path: Path, mmap: bool = True) -> dict[str, object]:
    """检查 .npy 文件的 shape、dtype、NaN 和 Inf。

    训练前先做这个检查，可以尽早发现预处理输出中是否存在坏值。
    对 X 这种大文件使用 mmap，避免一次性读入内存。
    """

    arr = np.load(path, mmap_mode="r" if mmap else None)
    total = int(arr.size)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    finite_count = total - nan_count - inf_count

    summary = {
        "path": str(path),
        "shape": tuple(int(v) for v in arr.shape),
        "dtype": str(arr.dtype),
        "total": total,
        "finite": finite_count,
        "nan": nan_count,
        "inf": inf_count,
    }

    finite_values = arr[np.isfinite(arr)]
    if finite_values.size > 0:
        summary["min"] = float(np.min(finite_values))
        summary["max"] = float(np.max(finite_values))
        summary["mean"] = float(np.mean(finite_values))
    else:
        summary["min"] = None
        summary["max"] = None
        summary["mean"] = None
    return summary


def validate_training_arrays(output_dir: Path) -> None:
    """训练前检查 X/y 是否含 NaN/Inf。

    如果发现坏值，直接停止训练，并提示重新运行预处理。继续用坏值训练只会得到
    NaN loss 和无效模型。
    """

    print("[CHECK] 开始检查预处理输出数组...")
    files = [
        ("X_train", output_dir / "X_train.npy", True),
        ("y_train", output_dir / "y_train.npy", False),
        ("X_val", output_dir / "X_val.npy", True),
        ("y_val", output_dir / "y_val.npy", False),
    ]

    bad_files: list[dict[str, object]] = []
    for name, path, mmap in files:
        summary = summarize_array_file(path, mmap=mmap)
        print(
            f"[CHECK] {name}: shape={summary['shape']}, dtype={summary['dtype']}, "
            f"nan={summary['nan']}, inf={summary['inf']}, "
            f"min={summary['min']}, max={summary['max']}, mean={summary['mean']}"
        )
        if summary["nan"] or summary["inf"]:
            bad_files.append(summary)

    if bad_files:
        details = "\n".join(
            f"- {item['path']}: nan={item['nan']}, inf={item['inf']}, shape={item['shape']}"
            for item in bad_files
        )
        raise ValueError(
            "训练数据中存在 NaN/Inf，已停止训练。\n"
            f"{details}\n"
            "请重新运行预处理代码。新版预处理会跳过 ERA 时间片不足或插值后含 NaN 的日期。"
        )


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """运行一个训练或验证 epoch。

    optimizer 为 None 时表示验证模式，不反向传播。
    返回该 epoch 的平均 MSE loss。
    """

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            loss = criterion(pred, y)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """对一个 DataLoader 进行预测，返回 numpy 数组。"""

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).detach().cpu().numpy()
            outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def main() -> None:
    """训练 5 个模型、做 ensemble、保存指标和图。"""

    args = parse_args()
    if torch is None:
        raise ModuleNotFoundError(
            "训练阶段需要 PyTorch，但当前环境没有安装 torch。\n"
            "请先运行: pip install torch\n"
            "如果你需要 GPU 版本，请按 PyTorch 官网选择与你 CUDA 匹配的安装命令。"
        )

    from cnn_model import StormSurgeCNN

    device = resolve_device(args.device)

    output_dir = XIAMEN_OUTPUT_DIR
    model_dir = PROJECT_ROOT / "models" / "xiamen"
    figure_dir = ensure_figure_dir(PROJECT_ROOT / "figures" / "xiamen")
    model_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "dates_val.npy",
        "y_scaler.json",
    ]
    for name in required_files:
        path = output_dir / name
        if not path.exists():
            raise FileNotFoundError(f"缺少预处理输出文件: {path}")

    validate_training_arrays(output_dir)

    train_dataset = NpyStormSurgeDataset(output_dir / "X_train.npy", output_dir / "y_train.npy")
    val_dataset = NpyStormSurgeDataset(output_dir / "X_val.npy", output_dir / "y_val.npy")
    y_mean, y_std = load_y_scaler(output_dir)

    print("=" * 80)
    print(f"[TRAIN] device: {device}")
    print(f"[TRAIN] X_train shape: {train_dataset.x.shape}")
    print(f"[TRAIN] y_train shape: {train_dataset.y.shape}")
    print(f"[TRAIN] X_val shape: {val_dataset.x.shape}")
    print(f"[TRAIN] y_val shape: {val_dataset.y.shape}")
    print(f"[TRAIN] seeds: {args.seeds}")
    print("=" * 80)

    criterion = nn.MSELoss()
    history: list[dict] = []
    val_predictions_standardized: list[np.ndarray] = []

    for seed in args.seeds:
        print(f"[TRAIN] 开始训练 seed={seed}")
        set_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        model = StormSurgeCNN(in_channels=int(train_dataset.x.shape[1])).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0
        for epoch in range(1, args.epochs + 1):
            train_loss = run_one_epoch(model, train_loader, criterion, device, optimizer)
            val_loss = run_one_epoch(model, val_loader, criterion, device, optimizer=None)
            history.append({"seed": seed, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"[TRAIN] seed={seed} epoch={epoch:03d}/{args.epochs:03d} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"best_epoch={best_epoch:03d} no_improve={epochs_without_improvement}/{args.patience}"
            )
            if epochs_without_improvement >= args.patience:
                print(f"[TRAIN] seed={seed} early stopping at epoch {epoch}, best_epoch={best_epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model_path = model_dir / f"model_seed_{seed}.pth"
        torch.save(
            {
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "y_scaler": {"mean": y_mean, "std": y_std},
            },
            model_path,
        )
        print(f"[SAVE] 模型已保存: {model_path}")

        val_pred = predict(model, val_loader, device)
        val_predictions_standardized.append(val_pred)

    pred_matrix_standardized = np.vstack(val_predictions_standardized)
    pred_ensemble_standardized = pred_matrix_standardized.mean(axis=0)

    y_val_standardized = np.load(output_dir / "y_val.npy")
    # y_val.npy 和模型输出都是标准化后的数值；必须先用训练集 mean/std 反标准化。
    # 论文报告 RMSE/MAE 的单位是厘米，因此再乘以 --surge-unit-scale-to-cm。
    observed_raw = inverse_transform_y(y_val_standardized, y_mean, y_std)
    pred_matrix_raw = inverse_transform_y(pred_matrix_standardized, y_mean, y_std)
    pred_ensemble_raw = inverse_transform_y(pred_ensemble_standardized, y_mean, y_std)
    observed_cm = observed_raw * args.surge_unit_scale_to_cm
    pred_matrix_cm = pred_matrix_raw * args.surge_unit_scale_to_cm
    pred_ensemble_cm = pred_ensemble_raw * args.surge_unit_scale_to_cm

    dates_val = np.load(output_dir / "dates_val.npy").astype("datetime64[D]").astype(str)
    extreme_threshold_cm = float(np.percentile(observed_cm, args.extreme_percentile))
    extreme_mask = observed_cm >= extreme_threshold_cm
    predictions = pd.DataFrame(
        {
            "date": dates_val,
            "y_true_scaled": y_val_standardized,
            "y_pred_scaled": pred_ensemble_standardized,
            "y_true_raw": observed_raw,
            "y_pred_raw": pred_ensemble_raw,
            "y_true_cm": observed_cm,
            "y_pred_cm": pred_ensemble_cm,
            "is_extreme_top_percentile": extreme_mask,
            # 兼容旧版绘图/查看习惯：observed 和 pred_ensemble 现在明确保存为厘米。
            "observed": observed_cm,
            "pred_ensemble": pred_ensemble_cm,
        }
    )
    for row_index, seed in enumerate(args.seeds):
        predictions[f"pred_seed_{seed}_scaled"] = pred_matrix_standardized[row_index]
        predictions[f"pred_seed_{seed}_raw"] = pred_matrix_raw[row_index]
        predictions[f"pred_seed_{seed}_cm"] = pred_matrix_cm[row_index]

    metrics = compute_metrics(observed_cm, pred_ensemble_cm, unit_suffix="_cm")
    extreme_metrics = compute_metrics(observed_cm[extreme_mask], pred_ensemble_cm[extreme_mask], unit_suffix="_cm")
    extreme_metrics = {f"extreme_top_{100 - args.extreme_percentile:.0f}pct_{key}": value for key, value in extreme_metrics.items()}
    metrics.update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "seeds": args.seeds,
            "n_train": len(train_dataset),
            "n_val": len(val_dataset),
            "extreme_percentile": args.extreme_percentile,
            "extreme_threshold_cm": extreme_threshold_cm,
            "n_extreme_val": int(extreme_mask.sum()),
            "metric_unit": "cm",
            "surge_unit_scale_to_cm": args.surge_unit_scale_to_cm,
            "y_scaler_mean_raw": y_mean,
            "y_scaler_std_raw": y_std,
        }
    )
    metrics.update(extreme_metrics)

    predictions_path = output_dir / "validation_predictions.csv"
    metrics_path = output_dir / "metrics.json"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_loss_curve(history, figure_dir / "loss_curve.png")
    plot_pred_vs_obs(predictions, figure_dir / "pred_vs_obs.png")
    plot_scatter(predictions, figure_dir / "scatter.png")

    print("=" * 80)
    print("[RESULT] Ensemble validation metrics")
    for key, value in metrics.items():
        print(f"[RESULT] {key}: {value}")
    print(f"[SAVE] validation_predictions.csv: {predictions_path}")
    print(f"[SAVE] metrics.json: {metrics_path}")
    print(f"[SAVE] figures: {figure_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
