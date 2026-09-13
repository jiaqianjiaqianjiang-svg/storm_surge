"""作者 notebook 风格的 CNN 风暴潮重建模型。

输入张量形状为 ``(batch, 3, 160, 160)``：
- 3 个通道分别是 U10、V10、SLP；
- 每个通道内部把 16 个 40×40 的 3 小时时间片拼成 4×4 的 160×160 大图。

模型结构贴近作者 notebook：
Conv2d(3,6,5) -> BN -> ReLU -> MaxPool
Conv2d(6,12,5) -> BN -> ReLU -> MaxPool
Conv2d(12,6,5) -> BN -> ReLU -> MaxPool
Linear(6*20*20,100) -> Linear(100,10) -> Linear(10,1)
"""

from __future__ import annotations

import torch
from torch import nn


class StormSurgeCNN(nn.Module):
    """用于厦门站 daily maximum storm surge 重建的 CNN。"""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.out1 = nn.Linear(6 * 20 * 20, 100)
        self.out2 = nn.Linear(100, 10)
        self.out3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回 shape=(batch,) 的预测值。"""

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.out1(x))
        x = self.relu(self.out2(x))
        x = self.out3(x)
        return x.squeeze(-1)
