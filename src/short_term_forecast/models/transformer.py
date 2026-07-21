"""轻量 Transformer 短时风暴潮预报模型。"""

from __future__ import annotations

import torch
from torch import nn

from models.common import StepCNNEncoder


class SimpleTransformerForecastModel(nn.Module):
    """TransformerEncoder 处理逐小时气象-增水序列特征。"""

    def __init__(
        self,
        input_steps: int,
        n_variables: int = 3,
        grid_size: int = 40,
        hidden_size: int = 96,
        nhead: int = 4,
    ) -> None:
        super().__init__()
        self.input_steps = input_steps
        self.n_variables = n_variables
        self.encoder = StepCNNEncoder(n_variables=n_variables, grid_size=grid_size, feature_dim=64)
        self.input_proj = nn.Linear(65, hidden_size)
        self.positional = nn.Parameter(torch.zeros(1, input_steps, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, atmosphere: torch.Tensor, surge_history: torch.Tensor) -> torch.Tensor:
        weather_seq = self.encoder(atmosphere, self.input_steps, self.n_variables)
        seq = torch.cat([weather_seq, surge_history.unsqueeze(-1)], dim=-1)
        hidden = self.input_proj(seq) + self.positional
        encoded = self.transformer(hidden)
        return self.regressor(encoded[:, -1]).squeeze(1)
