"""Two-branch CNN/MLP model for one-hour storm-surge prediction."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn


class CaribbeanSurgeCNN(nn.Module):
    def __init__(
        self,
        input_steps: int = 24,
        variables: Sequence[str] = ("U10", "V10", "MSL"),
        grid_size: int = 40,
    ) -> None:
        super().__init__()
        self.input_steps = int(input_steps)
        self.variables = tuple(variables)
        self.grid_size = int(grid_size)
        in_channels = self.input_steps * len(self.variables)
        self.atmosphere_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, self.grid_size, self.grid_size)
            flattened = self.atmosphere_branch(dummy).numel()
        self.atmosphere_head = nn.Sequential(nn.Flatten(), nn.Linear(flattened, 128), nn.ReLU(inplace=True))
        self.surge_branch = nn.Sequential(
            nn.Linear(self.input_steps, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(160, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def forward(self, atmosphere: torch.Tensor, surge_history: torch.Tensor) -> torch.Tensor:
        expected_channels = self.input_steps * len(self.variables)
        if atmosphere.ndim != 4 or atmosphere.shape[1] != expected_channels:
            raise ValueError(
                f"Atmosphere must have shape (batch, {expected_channels}, {self.grid_size}, {self.grid_size})"
            )
        if surge_history.ndim != 2 or surge_history.shape[1] != self.input_steps:
            raise ValueError(f"Surge history must have shape (batch, {self.input_steps})")
        weather = self.atmosphere_head(self.atmosphere_branch(atmosphere))
        history = self.surge_branch(surge_history)
        return self.fusion(torch.cat([weather, history], dim=1)).squeeze(-1)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "model_name": type(self).__name__,
            "input_steps": self.input_steps,
            "variables": list(self.variables),
            "grid_size": self.grid_size,
        }


class ERA5OnlyCNN(nn.Module):
    """Atmospheric ablation: predict without observed surge history."""

    def __init__(
        self,
        input_steps: int = 24,
        variables: Sequence[str] = ("U10", "V10", "MSL"),
        grid_size: int = 40,
    ) -> None:
        super().__init__()
        self.input_steps = int(input_steps)
        self.variables = tuple(variables)
        self.grid_size = int(grid_size)
        in_channels = self.input_steps * len(self.variables)
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flattened = self.network(
                torch.zeros(1, in_channels, self.grid_size, self.grid_size)
            ).numel()
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(flattened, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 1),
        )

    def forward(self, atmosphere: torch.Tensor, surge_history: torch.Tensor) -> torch.Tensor:
        expected_channels = self.input_steps * len(self.variables)
        if atmosphere.ndim != 4 or atmosphere.shape[1] != expected_channels:
            raise ValueError(
                f"Atmosphere must have shape (batch, {expected_channels}, "
                f"{self.grid_size}, {self.grid_size})"
            )
        return self.head(self.network(atmosphere)).squeeze(-1)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "model_name": type(self).__name__,
            "input_steps": self.input_steps,
            "variables": list(self.variables),
            "grid_size": self.grid_size,
        }


class SurgeHistoryMLP(nn.Module):
    """History-only ablation: predict without ERA5 fields."""

    def __init__(
        self,
        input_steps: int = 24,
        variables: Sequence[str] = ("U10", "V10", "MSL"),
        grid_size: int = 40,
    ) -> None:
        super().__init__()
        self.input_steps = int(input_steps)
        self.variables = tuple(variables)
        self.grid_size = int(grid_size)
        self.network = nn.Sequential(
            nn.Linear(self.input_steps, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(32, 1),
        )

    def forward(self, atmosphere: torch.Tensor, surge_history: torch.Tensor) -> torch.Tensor:
        if surge_history.ndim != 2 or surge_history.shape[1] != self.input_steps:
            raise ValueError(f"Surge history must have shape (batch, {self.input_steps})")
        return self.network(surge_history).squeeze(-1)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "model_name": type(self).__name__,
            "input_steps": self.input_steps,
            "variables": list(self.variables),
            "grid_size": self.grid_size,
        }


MODEL_TYPES: dict[str, type[nn.Module]] = {
    "dual": CaribbeanSurgeCNN,
    "era5_cnn": ERA5OnlyCNN,
    "surge_mlp": SurgeHistoryMLP,
}


def create_model(
    model_type: str,
    input_steps: int,
    variables: Sequence[str],
    grid_size: int,
) -> nn.Module:
    try:
        model_class = MODEL_TYPES[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown model type {model_type!r}; expected {sorted(MODEL_TYPES)}") from exc
    return model_class(input_steps, variables, grid_size)


def model_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    names = {model_class.__name__: model_class for model_class in MODEL_TYPES.values()}
    model_name = checkpoint.get("model_name", "CaribbeanSurgeCNN")
    try:
        model_class = names[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported checkpoint model {model_name!r}") from exc
    model = model_class(
        input_steps=int(checkpoint["input_steps"]),
        variables=checkpoint["variables"],
        grid_size=int(checkpoint["grid_size"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
