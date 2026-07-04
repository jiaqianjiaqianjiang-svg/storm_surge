"""短时风暴潮预报 CNN 模型。

模型包含两个分支：
1. 气象场 CNN 分支，输入为 (batch, t*3, 40, 40)。
2. 历史增水 MLP 分支，输入为 (batch, t)。

两个分支的特征拼接后输出下一步 storm surge 标量。
"""

from __future__ import annotations

import torch
from torch import nn


class ForecastCNN(nn.Module):
    """融合大气场和历史增水的短时预报模型。"""

    def __init__(self, input_steps: int, n_variables: int = 3, grid_size: int = 40) -> None:
        super().__init__()
        self.input_steps = input_steps
        self.n_variables = n_variables
        self.grid_size = grid_size
        in_channels = input_steps * n_variables

        # CNN 分支：从 t 个时间步的 U10/V10/SLP 网格中提取空间天气形势特征。
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        pooled_size = grid_size // 8
        cnn_feature_dim = 128 * pooled_size * pooled_size
        self.cnn_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        # 历史增水分支：学习最近 t 个 storm surge 的自回归状态。
        self.surge_mlp = nn.Sequential(
            nn.Linear(input_steps, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # 融合两类特征，输出下一步风暴潮。
        self.regressor = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, atmosphere: torch.Tensor, surge_history: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        atmosphere:
            shape = (batch, t*3, 40, 40)，通道顺序为每个时间步依次拼接
            U10、V10、SLP/MSL。
        surge_history:
            shape = (batch, t)，最近 t 个 storm surge 历史值。
        """

        weather_features = self.cnn_head(self.cnn(atmosphere))
        surge_features = self.surge_mlp(surge_history)
        fused = torch.cat([weather_features, surge_features], dim=1)
        return self.regressor(fused).squeeze(1)
