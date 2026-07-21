"""短时预报模型共用组件。"""

from __future__ import annotations

import torch
from torch import nn


class StepCNNEncoder(nn.Module):
    """对单个时刻的 U10/V10/SLP 40x40 网格提取空间特征。"""

    def __init__(self, n_variables: int = 3, grid_size: int = 40, feature_dim: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_variables, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        pooled = grid_size // 8
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pooled * pooled, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, atmosphere: torch.Tensor, input_steps: int, n_variables: int) -> torch.Tensor:
        """返回 shape=(batch, t, feature_dim) 的逐时刻空间特征。"""

        batch_size = atmosphere.shape[0]
        x = atmosphere.view(batch_size, input_steps, n_variables, atmosphere.shape[-2], atmosphere.shape[-1])
        x = x.reshape(batch_size * input_steps, n_variables, atmosphere.shape[-2], atmosphere.shape[-1])
        features = self.head(self.cnn(x))
        return features.view(batch_size, input_steps, -1)


class TemporalConvBlock(nn.Module):
    """简化版 TCN block，使用 causal padding 后裁掉未来部分。"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.trim = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(x)
        if self.trim > 0:
            out = out[..., :-self.trim]
        out = self.dropout(self.activation(out))
        return out + residual
