"""Spatiotemporal model variants for the formal Xiamen experiment."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn


class StepCNNEncoder(nn.Module):
    """Encode each hourly ERA5 field before temporal modelling."""

    def __init__(
        self,
        variables: int,
        grid_size: int,
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.variables = int(variables)
        self.grid_size = int(grid_size)
        self.network = nn.Sequential(
            nn.Conv2d(self.variables, 16, kernel_size=3, padding=1),
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
        with torch.no_grad():
            dummy = torch.zeros(1, self.variables, self.grid_size, self.grid_size)
            flattened = self.network(dummy).numel()
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(flattened, feature_dim), nn.ReLU(inplace=True)
        )

    def forward(self, atmosphere: torch.Tensor, input_steps: int) -> torch.Tensor:
        expected_channels = input_steps * self.variables
        if atmosphere.ndim != 4 or atmosphere.shape[1] != expected_channels:
            raise ValueError(
                f"Atmosphere must have shape (batch, {expected_channels}, "
                f"{self.grid_size}, {self.grid_size})"
            )
        batch = atmosphere.shape[0]
        hourly = atmosphere.reshape(
            batch * input_steps,
            self.variables,
            self.grid_size,
            self.grid_size,
        )
        encoded = self.head(self.network(hourly))
        return encoded.reshape(batch, input_steps, -1)


class TemporalConvBlock(nn.Module):
    """Residual causal temporal convolution."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.trim = padding

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        output = self.convolution(values)
        if self.trim:
            output = output[..., :-self.trim]
        return residual + self.dropout(self.activation(output))


class _TemporalForecastBase(nn.Module):
    def __init__(
        self,
        input_steps: int,
        variables: Sequence[str],
        grid_size: int,
    ) -> None:
        super().__init__()
        self.input_steps = int(input_steps)
        self.variables = tuple(variables)
        self.grid_size = int(grid_size)
        self.encoder = StepCNNEncoder(len(self.variables), self.grid_size)

    def sequence_features(
        self, atmosphere: torch.Tensor, surge_history: torch.Tensor
    ) -> torch.Tensor:
        if surge_history.ndim != 2 or surge_history.shape[1] != self.input_steps:
            raise ValueError(
                f"Surge history must have shape (batch, {self.input_steps})"
            )
        weather = self.encoder(atmosphere, self.input_steps)
        return torch.cat([weather, surge_history.unsqueeze(-1)], dim=-1)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "model_name": type(self).__name__,
            "input_steps": self.input_steps,
            "variables": list(self.variables),
            "grid_size": self.grid_size,
        }


class CNNLSTMForecastModel(_TemporalForecastBase):
    """Hourly spatial CNN encoder followed by an LSTM."""

    def __init__(
        self,
        input_steps: int = 24,
        variables: Sequence[str] = ("U10", "V10", "MSL"),
        grid_size: int = 40,
        hidden_size: int = 96,
    ) -> None:
        super().__init__(input_steps, variables, grid_size)
        self.lstm = nn.LSTM(65, hidden_size, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self, atmosphere: torch.Tensor, surge_history: torch.Tensor
    ) -> torch.Tensor:
        output, _ = self.lstm(self.sequence_features(atmosphere, surge_history))
        return self.regressor(output[:, -1]).squeeze(-1)


class CNNGRUForecastModel(_TemporalForecastBase):
    """Hourly spatial CNN encoder followed by a GRU."""

    def __init__(
        self,
        input_steps: int = 24,
        variables: Sequence[str] = ("U10", "V10", "MSL"),
        grid_size: int = 40,
        hidden_size: int = 96,
    ) -> None:
        super().__init__(input_steps, variables, grid_size)
        self.gru = nn.GRU(65, hidden_size, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self, atmosphere: torch.Tensor, surge_history: torch.Tensor
    ) -> torch.Tensor:
        output, _ = self.gru(self.sequence_features(atmosphere, surge_history))
        return self.regressor(output[:, -1]).squeeze(-1)


class TCNForecastModel(_TemporalForecastBase):
    """Hourly spatial CNN encoder followed by causal temporal convolutions."""

    def __init__(
        self,
        input_steps: int = 24,
        variables: Sequence[str] = ("U10", "V10", "MSL"),
        grid_size: int = 40,
        channels: int = 96,
    ) -> None:
        super().__init__(input_steps, variables, grid_size)
        self.projection = nn.Conv1d(65, channels, kernel_size=1)
        self.temporal = nn.Sequential(
            TemporalConvBlock(channels, dilation=1),
            TemporalConvBlock(channels, dilation=2),
            TemporalConvBlock(channels, dilation=4),
        )
        self.regressor = nn.Sequential(
            nn.Linear(channels, 64), nn.ReLU(inplace=True), nn.Linear(64, 1)
        )

    def forward(
        self, atmosphere: torch.Tensor, surge_history: torch.Tensor
    ) -> torch.Tensor:
        sequence = self.sequence_features(atmosphere, surge_history).transpose(1, 2)
        features = self.temporal(self.projection(sequence))
        return self.regressor(features[:, :, -1]).squeeze(-1)


class TransformerForecastModel(_TemporalForecastBase):
    """Hourly spatial CNN encoder followed by a compact Transformer."""

    def __init__(
        self,
        input_steps: int = 24,
        variables: Sequence[str] = ("U10", "V10", "MSL"),
        grid_size: int = 40,
        hidden_size: int = 96,
        heads: int = 4,
    ) -> None:
        super().__init__(input_steps, variables, grid_size)
        self.input_projection = nn.Linear(65, hidden_size)
        self.positional = nn.Parameter(torch.zeros(1, self.input_steps, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(inplace=True), nn.Linear(64, 1)
        )

    def forward(
        self, atmosphere: torch.Tensor, surge_history: torch.Tensor
    ) -> torch.Tensor:
        sequence = self.sequence_features(atmosphere, surge_history)
        hidden = self.input_projection(sequence) + self.positional
        encoded = self.transformer(hidden)
        return self.regressor(encoded[:, -1]).squeeze(-1)
