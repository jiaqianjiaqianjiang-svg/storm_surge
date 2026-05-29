"""模型验证指标。

训练阶段的 y 是标准化后的数值；真正和论文对比时，必须先反标准化，
再把 storm surge 转成厘米。这里的函数只负责数学计算，单位由调用者保证。
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


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算决定系数 R²。

    论文图中常用 R² 描述拟合优度。R²=1 表示完全一致；如果模型比直接预测均值还差，
    R² 可能为负数。
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


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

    这里采用常见相对误差写法：RMSE / mean(abs(y_true)) × 100%。
    若分母为 0，则返回 NaN。
    """

    denominator = float(np.mean(np.abs(y_true)))
    if denominator == 0:
        return float("nan")
    return rmse(y_true, y_pred) / denominator * 100.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, unit_suffix: str = "") -> dict[str, float]:
    """一次性计算验证集指标。

    unit_suffix 可用于把有单位的指标写清楚。例如调用者传入厘米数据时，
    使用 unit_suffix="_cm" 会得到 rmse_cm、mae_cm。
    """

    return {
        "pearson_r": pearson_r(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        f"rmse{unit_suffix}": rmse(y_true, y_pred),
        f"mae{unit_suffix}": mae(y_true, y_pred),
        "rrmse_percent": rrmse(y_true, y_pred),
    }
