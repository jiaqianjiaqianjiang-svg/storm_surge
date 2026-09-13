"""训练过程和验证结果绘图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_figure_dir(path: str | Path) -> Path:
    """创建图片输出目录。"""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_loss_curve(history: list[dict], output_path: str | Path) -> None:
    """绘制 5 个模型平均训练/验证 loss 曲线。"""

    output_path = Path(output_path)
    history_df = pd.DataFrame(history)
    grouped = history_df.groupby("epoch", as_index=False)[["train_loss", "val_loss"]].mean()

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["epoch"], grouped["train_loss"], label="Train loss")
    plt.plot(grouped["epoch"], grouped["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Xiamen CNN loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pred_vs_obs(predictions: pd.DataFrame, output_path: str | Path) -> None:
    """绘制验证集时间序列：观测 daily maximum surge vs ensemble 预测。"""

    output_path = Path(output_path)
    dates = pd.to_datetime(predictions["date"])

    plt.figure(figsize=(11, 5))
    plt.plot(dates, predictions["observed"], label="Observed", linewidth=1.4)
    plt.plot(dates, predictions["pred_ensemble"], label="CNN ensemble", linewidth=1.4)
    plt.xlabel("Date")
    plt.ylabel("Daily maximum storm surge")
    plt.title("Xiamen validation: predicted vs observed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_scatter(predictions: pd.DataFrame, output_path: str | Path) -> None:
    """绘制验证集散点图。"""

    output_path = Path(output_path)
    observed = predictions["observed"].to_numpy(dtype=float)
    predicted = predictions["pred_ensemble"].to_numpy(dtype=float)
    low = float(np.nanmin([observed.min(), predicted.min()]))
    high = float(np.nanmax([observed.max(), predicted.max()]))

    plt.figure(figsize=(6, 6))
    plt.scatter(observed, predicted, s=18, alpha=0.75)
    plt.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title("Xiamen validation scatter")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
