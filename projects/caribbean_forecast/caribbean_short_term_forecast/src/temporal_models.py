"""Spatiotemporal models shared with the Xiamen short-term workflow."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn


class StepCNNEncoder(nn.Module):
    """Encode each hourly ERA5 field independently."""

    def __init__(self, variables: int, grid_size: int, feature_dim: int = 64) -> None:
        super().__init__()
        self.variables = int(variables)
        self.grid_size = int(grid_size)
        self.network = nn.Sequential(
            nn.Conv2d(self.variables, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
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
            batch * input_steps, self.variables, self.grid_size, self.grid_size
        )
        encoded = self.head(self.network(hourly))
        return encoded.reshape(batch, input_steps, -1)


class CNNGRUForecastModel(nn.Module):
    """Hourly spatial CNN encoder followed by a GRU temporal encoder."""

    def __init__(
        self,
        input_steps: int = 24,
        variables: Sequence[str] = ("U10", "V10", "MSL"),
        grid_size: int = 40,
        hidden_size: int = 96,
    ) -> None:
        super().__init__()
        self.input_steps = int(input_steps)
        self.variables = tuple(variables)
        self.grid_size = int(grid_size)
        self.encoder = StepCNNEncoder(len(self.variables), self.grid_size)
        self.gru = nn.GRU(65, hidden_size, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(64, 1),
        )

    def weather_features(self, atmosphere: torch.Tensor) -> torch.Tensor:
        return self.encoder(atmosphere, self.input_steps)

    def sequence_from_embeddings(
        self, weather: torch.Tensor, surge_history: torch.Tensor
    ) -> torch.Tensor:
        if surge_history.ndim != 2 or surge_history.shape[1] != self.input_steps:
            raise ValueError(
                f"Surge history must have shape (batch, {self.input_steps})"
            )
        if weather.ndim != 3 or weather.shape[1:] != (self.input_steps, 64):
            raise ValueError(
                f"Weather embeddings must have shape (batch, {self.input_steps}, 64)"
            )
        return torch.cat([weather, surge_history.unsqueeze(-1)], dim=-1)

    def forward(
        self, atmosphere: torch.Tensor, surge_history: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_encoded(self.weather_features(atmosphere), surge_history)

    def forward_encoded(
        self, weather: torch.Tensor, surge_history: torch.Tensor
    ) -> torch.Tensor:
        output, _ = self.gru(self.sequence_from_embeddings(weather, surge_history))
        return self.regressor(output[:, -1]).squeeze(-1)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "model_name": type(self).__name__,
            "input_steps": self.input_steps,
            "variables": list(self.variables),
            "grid_size": self.grid_size,
        }
