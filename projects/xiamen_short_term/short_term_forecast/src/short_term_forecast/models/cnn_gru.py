"""CNN-GRU 短时风暴潮预报模型。"""

from __future__ import annotations

import torch
from torch import nn

from models.common import StepCNNEncoder


class CNNGRUForecastModel(nn.Module):
    """CNN 提取逐时刻空间特征，GRU 建模时间演变。"""

    def __init__(self, input_steps: int, n_variables: int = 3, grid_size: int = 40, hidden_size: int = 96) -> None:
        super().__init__()
        self.input_steps = input_steps
        self.n_variables = n_variables
        self.encoder = StepCNNEncoder(n_variables=n_variables, grid_size=grid_size, feature_dim=64)
        self.gru = nn.GRU(input_size=65, hidden_size=hidden_size, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, atmosphere: torch.Tensor, surge_history: torch.Tensor) -> torch.Tensor:
        weather_seq = self.encoder(atmosphere, self.input_steps, self.n_variables)
        seq = torch.cat([weather_seq, surge_history.unsqueeze(-1)], dim=-1)
        out, _ = self.gru(seq)
        return self.regressor(out[:, -1]).squeeze(1)
