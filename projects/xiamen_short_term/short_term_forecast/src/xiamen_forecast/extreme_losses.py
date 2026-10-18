"""正向极端风暴增水训练损失。

输入损失函数的目标已经标准化；极值权重必须先把目标反变换为米，确保阈值
始终具有明确的物理意义。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


LOSS_TYPES = ("mse", "wmse", "awmse", "whuber")
EXTREME_MODEL_DIRECTORIES = {
    "wmse": "cnn_gru_extreme_wmse",
    "awmse": "cnn_gru_extreme_awmse",
    "whuber": "cnn_gru_extreme_whuber",
}


@dataclass(frozen=True)
class ExtremeLossConfig:
    """可序列化的训练目标定义。"""

    loss_type: str
    alpha: float
    beta: float
    huber_delta_cm: float
    huber_delta_scaled: float
    surge_mean_m: float
    surge_scale_m: float
    q90_probability: float
    q99_probability: float
    raw_q90_m: float | None
    raw_q99_m: float | None
    q90_m: float | None
    q99_m: float | None
    threshold_unit: str
    threshold_sample_count: int
    positive_threshold_protected: bool
    positive_threshold_strategy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_targets(values_m: object) -> np.ndarray:
    values = np.asarray(values_m, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def fit_positive_extreme_thresholds(
    training_targets_m: object,
    q90_probability: float = 0.90,
    q99_probability: float = 0.99,
) -> dict[str, Any]:
    """只使用训练目标拟合物理单位阈值。

    原始P90不为正时，下阈值保护为0 m，使零及负增水保持单位权重，只强化
    正向增水；同时要求P99必须大于保护后的P90。
    """

    if not 0 < q90_probability < q99_probability < 1:
        raise ValueError("Expected 0 < q90_probability < q99_probability < 1")
    values = _finite_targets(training_targets_m)
    if not len(values):
        raise ValueError("No finite training targets are available for thresholds")
    raw_q90 = float(np.quantile(values, q90_probability))
    raw_q99 = float(np.quantile(values, q99_probability))
    protected = raw_q90 <= 0.0
    q90 = max(raw_q90, 0.0)
    if not np.isfinite(raw_q99) or raw_q99 <= q90:
        raise ValueError(
            "Positive extreme thresholds are invalid: training P99 must be "
            f"greater than protected P90 (P90={q90:.6g} m, P99={raw_q99:.6g} m)"
        )
    return {
        "q90_probability": q90_probability,
        "q99_probability": q99_probability,
        "raw_q90_m": raw_q90,
        "raw_q99_m": raw_q99,
        "q90_m": q90,
        "q99_m": raw_q99,
        "threshold_unit": "m",
        "threshold_sample_count": int(len(values)),
        "positive_threshold_protected": protected,
        "positive_threshold_strategy": (
            "q90=max(raw_training_p90, 0 m); targets <= 0 m retain unit weight"
        ),
    }


def extreme_ratio_from_scaled_targets(
    target_scaled: torch.Tensor,
    surge_mean_m: float,
    surge_scale_m: float,
    q90_m: float,
    q99_m: float,
) -> torch.Tensor:
    """根据标准化目标返回[0, 1]范围的正向极值强度。"""

    if not np.isfinite(surge_scale_m) or surge_scale_m <= 0:
        raise ValueError("surge_scale_m must be finite and positive")
    if not q99_m > q90_m:
        raise ValueError("q99_m must be greater than q90_m")
    target_m = target_scaled * surge_scale_m + surge_mean_m
    return torch.clamp((target_m - q90_m) / (q99_m - q90_m), 0.0, 1.0)


class ExtremeRegressionLoss(nn.Module):
    """普通MSE、加权MSE、不对称加权MSE或加权Huber损失。"""

    def __init__(self, config: ExtremeLossConfig) -> None:
        super().__init__()
        if config.loss_type not in LOSS_TYPES:
            raise ValueError(f"Unsupported loss type: {config.loss_type}")
        if config.alpha < 0 or config.beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if config.huber_delta_scaled <= 0:
            raise ValueError("Huber delta must be positive")
        if config.loss_type != "mse":
            if config.q90_m is None or config.q99_m is None:
                raise ValueError("Extreme losses require q90_m and q99_m")
            if config.q99_m <= config.q90_m:
                raise ValueError("q99_m must be greater than q90_m")
        self.config = config

    def _ratio(self, target: torch.Tensor) -> torch.Tensor:
        if self.config.loss_type == "mse":
            return torch.zeros_like(target)
        return extreme_ratio_from_scaled_targets(
            target,
            self.config.surge_mean_m,
            self.config.surge_scale_m,
            float(self.config.q90_m),
            float(self.config.q99_m),
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
            )
        error = prediction - target
        if self.config.loss_type == "mse":
            return torch.mean(error.square())
        ratio = self._ratio(target)
        weights = 1.0 + self.config.alpha * ratio
        denominator = torch.clamp(weights.sum(), min=torch.finfo(weights.dtype).eps)
        if self.config.loss_type == "wmse":
            point_loss = error.square()
        elif self.config.loss_type == "awmse":
            under_error = torch.relu(target - prediction)
            point_loss = error.square() + self.config.beta * ratio * under_error.square()
        else:
            absolute = error.abs()
            delta = self.config.huber_delta_scaled
            point_loss = torch.where(
                absolute <= delta,
                0.5 * error.square(),
                delta * (absolute - 0.5 * delta),
            )
        return torch.sum(weights * point_loss) / denominator


def build_training_loss(
    loss_type: str,
    training_targets_m: object,
    surge_mean_m: float,
    surge_scale_m: float,
    alpha: float = 2.0,
    beta: float = 1.0,
    huber_delta_cm: float = 5.0,
) -> tuple[ExtremeRegressionLoss, dict[str, Any]]:
    """仅根据训练目标建立损失函数及完整元数据。"""

    loss_type = str(loss_type).lower()
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {LOSS_TYPES}")
    if not np.isfinite(surge_scale_m) or surge_scale_m <= 0:
        raise ValueError("surge_scale_m must be finite and positive")
    if not np.isfinite(huber_delta_cm) or huber_delta_cm <= 0:
        raise ValueError("huber_delta_cm must be finite and positive")
    if loss_type == "mse":
        thresholds = {
            "q90_probability": 0.90,
            "q99_probability": 0.99,
            "raw_q90_m": None,
            "raw_q99_m": None,
            "q90_m": None,
            "q99_m": None,
            "threshold_unit": "m",
            "threshold_sample_count": int(len(_finite_targets(training_targets_m))),
            "positive_threshold_protected": False,
            "positive_threshold_strategy": "not applicable to ordinary MSE",
        }
    else:
        thresholds = fit_positive_extreme_thresholds(training_targets_m)
    config = ExtremeLossConfig(
        loss_type=loss_type,
        alpha=float(alpha),
        beta=float(beta),
        huber_delta_cm=float(huber_delta_cm),
        huber_delta_scaled=float((huber_delta_cm / 100.0) / surge_scale_m),
        surge_mean_m=float(surge_mean_m),
        surge_scale_m=float(surge_scale_m),
        **thresholds,
    )
    criterion = ExtremeRegressionLoss(config)
    return criterion, config.to_dict()


def one_step_experiment_directory(model_type: str, loss_type: str) -> str:
    """保留历史MSE目录，并隔离每个新训练目标。"""

    if loss_type == "mse":
        return model_type
    if model_type != "cnn_gru":
        raise ValueError("Extreme-loss ablations are fixed to the CNN-GRU architecture")
    return EXTREME_MODEL_DIRECTORIES[loss_type]


def rollout_experiment_directory(
    base_experiment: str,
    loss_type: str,
    rollout_steps: int,
) -> str:
    """返回互不覆盖的rollout checkpoint目录名。"""

    if base_experiment == "cnn_gru" and loss_type == "mse":
        return f"cnn_gru_rollout{rollout_steps}"
    return f"{base_experiment}_rollout{rollout_steps}_{loss_type}"
