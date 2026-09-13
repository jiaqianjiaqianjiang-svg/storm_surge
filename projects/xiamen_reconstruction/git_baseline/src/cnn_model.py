"""论文 CNN 风暴潮重建模型。

输入张量形状为 ``(batch, 48, 40, 40)``：
- 48 个通道 = 16 个 3 小时时间片 × 3 个 ERA 变量；
- 40×40 是厦门站周围 10°×10° 区域插值后的网格。

模型结构对应论文描述：
3 个卷积层，每个卷积层后接 BatchNorm + ReLU + MaxPool，
随后接 3 个全连接层，输出单个 daily maximum storm surge。
"""

from __future__ import annotations

import torch
from torch import nn


class StormSurgeCNN(nn.Module):
    """用于厦门站 daily maximum storm surge 重建的 CNN。

    这里使用 padding=2，让 5×5 卷积不改变空间大小。每次 2×2 池化后：
    40×40 -> 20×20 -> 10×10 -> 5×5。
    最后将 128 个 5×5 特征图展平，送入 3 个全连接层。
    """

    def __init__(self, in_channels: int = 48) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回 shape=(batch,) 的预测值。"""

        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(-1)
