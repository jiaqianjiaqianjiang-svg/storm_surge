"""模型验证指标。

所有指标都在反标准化后的原始 storm surge 数值上计算，便于和论文中的
相关系数、RMSE、MAE、RRMSE 对照。
"""

from __future__ import annotations

import numpy as np


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 Pearson correlation coefficient。

    如果输入长度小于 2 或任一序列方差为 0，相关系数没有意义，返回 NaN。
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 root mean squared error。"""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 mean absolute error。"""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def rrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 relative RMSE。

    这里采用论文常见的相对误差写法：RMSE / mean(abs(y_true)) × 100%。
    若分母为 0，则返回 NaN。
    """

    denominator = float(np.mean(np.abs(y_true)))
    if denominator == 0:
        return float("nan")
    return rmse(y_true, y_pred) / denominator * 100.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """一次性计算验证集指标。"""

    return {
        "pearson_r": pearson_r(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rrmse_percent": rrmse(y_true, y_pred),
    }
