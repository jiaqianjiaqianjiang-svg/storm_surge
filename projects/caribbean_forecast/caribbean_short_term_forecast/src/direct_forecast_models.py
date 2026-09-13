"""Models and datasets for direct continuous 24-hour surge prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class DirectMLP(nn.Module):
    def __init__(self, input_features: int, output_steps: int = 24) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.output_steps = int(output_steps)
        self.network = nn.Sequential(
            nn.Linear(self.input_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.output_steps),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "model_name": type(self).__name__,
            "input_features": self.input_features,
            "output_steps": self.output_steps,
        }


class DirectGRU(nn.Module):
    def __init__(
        self,
        input_features: int = 14,
        hidden_size: int = 96,
        num_layers: int = 2,
        output_steps: int = 24,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.output_steps = int(output_steps)
        self.gru = nn.GRU(
            input_size=self.input_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2 if self.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, self.output_steps),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        return self.head(hidden[-1])

    def architecture_config(self) -> dict[str, Any]:
        return {
            "model_name": type(self).__name__,
            "input_features": self.input_features,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_steps": self.output_steps,
        }


class DirectDualCNN(nn.Module):
    def __init__(
        self,
        atmosphere_steps: int,
        input_steps: int = 24,
        variables: int = 3,
        grid_size: int = 40,
        output_steps: int = 24,
    ) -> None:
        super().__init__()
        self.atmosphere_steps = int(atmosphere_steps)
        self.input_steps = int(input_steps)
        self.variables = int(variables)
        self.grid_size = int(grid_size)
        self.output_steps = int(output_steps)
        channels = self.atmosphere_steps * self.variables
        self.atmosphere_branch = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flattened = self.atmosphere_branch(
                torch.zeros(1, channels, self.grid_size, self.grid_size)
            ).numel()
        self.atmosphere_head = nn.Sequential(
            nn.Flatten(), nn.Linear(flattened, 128), nn.ReLU(inplace=True)
        )
        self.surge_branch = nn.Sequential(
            nn.Linear(self.input_steps, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(160, 96), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(96, self.output_steps),
        )

    def forward(self, atmosphere: torch.Tensor, surge_history: torch.Tensor) -> torch.Tensor:
        weather = self.atmosphere_head(self.atmosphere_branch(atmosphere))
        history = self.surge_branch(surge_history)
        return self.fusion(torch.cat([weather, history], dim=1))

    def architecture_config(self) -> dict[str, Any]:
        return {
            "model_name": type(self).__name__,
            "atmosphere_steps": self.atmosphere_steps,
            "input_steps": self.input_steps,
            "variables": self.variables,
            "grid_size": self.grid_size,
            "output_steps": self.output_steps,
        }


class MatchedSurgeAblation(nn.Module):
    """Surge-only ablation retaining the Dual-CNN surge branch and fusion head."""

    def __init__(self, input_steps: int = 24, output_steps: int = 24) -> None:
        super().__init__()
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.surge_branch = nn.Sequential(
            nn.Linear(self.input_steps, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(160, 96), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(96, self.output_steps),
        )

    def forward(self, surge_history: torch.Tensor) -> torch.Tensor:
        history = self.surge_branch(surge_history)
        # The 128 weather features are explicitly absent, while the fusion
        # head remains dimensionally identical to DirectDualCNN.
        absent_weather = torch.zeros(
            (len(surge_history), 128),
            dtype=history.dtype,
            device=history.device,
        )
        return self.fusion(torch.cat([absent_weather, history], dim=1))

    def architecture_config(self) -> dict[str, Any]:
        return {
            "model_name": type(self).__name__,
            "input_steps": self.input_steps,
            "output_steps": self.output_steps,
            "ablation": "Dual-CNN surge branch and fusion head with ERA5 branch removed",
        }


class ArrayDirectDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[index]), torch.from_numpy(self.labels[index])


class GridDirectDataset(Dataset):
    """Load overlapping ERA5 windows on demand from the prepared memory map."""

    def __init__(
        self,
        atmosphere: np.ndarray,
        surge: np.ndarray,
        origins: np.ndarray,
        labels: np.ndarray,
        atmosphere_mean: np.ndarray,
        atmosphere_scale: np.ndarray,
        surge_mean: float,
        surge_scale: float,
        include_future: bool,
        input_steps: int = 24,
        output_steps: int = 24,
    ) -> None:
        self.atmosphere = atmosphere
        self.surge = np.asarray(surge, dtype=np.float32)
        self.origins = np.asarray(origins, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.atmosphere_mean = np.asarray(atmosphere_mean, dtype=np.float32).reshape(1, -1, 1, 1)
        self.atmosphere_scale = np.asarray(atmosphere_scale, dtype=np.float32).reshape(1, -1, 1, 1)
        self.surge_mean = float(surge_mean)
        self.surge_scale = float(surge_scale)
        self.include_future = bool(include_future)
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        origin = int(self.origins[index])
        start = origin - self.input_steps + 1
        stop = origin + self.output_steps + 1 if self.include_future else origin + 1
        atmosphere = np.asarray(self.atmosphere[start:stop], dtype=np.float32)
        atmosphere = (atmosphere - self.atmosphere_mean) / self.atmosphere_scale
        channels = atmosphere.reshape(-1, atmosphere.shape[-2], atmosphere.shape[-1])
        history = np.asarray(self.surge[start : origin + 1], dtype=np.float32)
        history = (history - self.surge_mean) / self.surge_scale
        labels = (self.labels[index] - self.surge_mean) / self.surge_scale
        return (
            torch.from_numpy(channels.copy()),
            torch.from_numpy(history.copy()),
            torch.from_numpy(labels.copy()),
        )
