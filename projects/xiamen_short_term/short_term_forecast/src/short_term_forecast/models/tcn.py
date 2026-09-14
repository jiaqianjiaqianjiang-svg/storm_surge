"""简化 TCN 短时风暴潮预报模型。"""

from __future__ import annotations

import torch
from torch import nn

from models.common import StepCNNEncoder, TemporalConvBlock


class TCNForecastModel(nn.Module):
    """用一维 temporal convolution 处理逐小时特征。"""

    def __init__(self, input_steps: int, n_variables: int = 3, grid_size: int = 40, channels: int = 96) -> None:
        super().__init__()
        self.input_steps = input_steps
        self.n_variables = n_variables
        self.encoder = StepCNNEncoder(n_variables=n_variables, grid_size=grid_size, feature_dim=64)
        self.proj = nn.Conv1d(65, channels, kernel_size=1)
        self.tcn = nn.Sequential(
            TemporalConvBlock(channels, dilation=1),
            TemporalConvBlock(channels, dilation=2),
            TemporalConvBlock(channels, dilation=4),
        )
        self.regressor = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, atmosphere: torch.Tensor, surge_history: torch.Tensor) -> torch.Tensor:
        weather_seq = self.encoder(atmosphere, self.input_steps, self.n_variables)
        seq = torch.cat([weather_seq, surge_history.unsqueeze(-1)], dim=-1)
        x = seq.transpose(1, 2)
        features = self.tcn(self.proj(x))
        return self.regressor(features[:, :, -1]).squeeze(1)
