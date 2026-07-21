"""On-demand Scheme-B samples with train-only scaling and temporal splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr


@dataclass
class Standardisation:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": np.asarray(self.mean).reshape(-1).tolist(), "scale": np.asarray(self.scale).reshape(-1).tolist()}

    @classmethod
    def from_state_dict(cls, state: dict[str, list[float]], shape: tuple[int, ...] = ()) -> "Standardisation":
        mean = np.asarray(state["mean"], dtype=np.float32)
        scale = np.asarray(state["scale"], dtype=np.float32)
        if shape:
            mean, scale = mean.reshape(shape), scale.reshape(shape)
        return cls(mean, scale)


def fit_scalers(
    atmosphere: np.ndarray | xr.DataArray,
    surge: np.ndarray,
    train_end: int,
) -> dict[str, Standardisation]:
    if train_end <= 0:
        raise ValueError("train_end must be positive")
    if len(atmosphere.shape) != 4:
        raise ValueError("Atmosphere must have shape (time, variables, latitude, longitude)")
    variables = int(atmosphere.shape[1])
    sums = np.zeros(variables, dtype=np.float64)
    sums_squared = np.zeros(variables, dtype=np.float64)
    counts = np.zeros(variables, dtype=np.int64)
    # Chunked moments avoid materialising years of 40x40 fields at once.
    for start in range(0, train_end, 168):
        chunk = np.asarray(atmosphere[start : min(train_end, start + 168)], dtype=np.float64)
        finite = np.isfinite(chunk)
        safe = np.where(finite, chunk, 0.0)
        sums += safe.sum(axis=(0, 2, 3))
        sums_squared += (safe * safe).sum(axis=(0, 2, 3))
        counts += finite.sum(axis=(0, 2, 3))
    if np.any(counts == 0):
        raise ValueError("At least one ERA5 variable has no finite training values")
    variable_mean = sums / counts
    variable_variance = np.maximum(0.0, sums_squared / counts - variable_mean**2)
    mean = variable_mean.astype(np.float32).reshape(1, variables, 1, 1)
    scale = np.sqrt(variable_variance).astype(np.float32).reshape(1, variables, 1, 1)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    train_surge = np.asarray(surge[:train_end], dtype=np.float64)
    surge_mean = np.asarray([np.nanmean(train_surge)], dtype=np.float32)
    surge_scale = np.asarray([np.nanstd(train_surge)], dtype=np.float32)
    surge_scale[~np.isfinite(surge_scale) | (surge_scale < 1e-8)] = 1.0
    return {
        "atmosphere": Standardisation(mean, scale),
        "surge": Standardisation(surge_mean, surge_scale),
    }


def valid_targets(times: object, atmosphere: np.ndarray | xr.DataArray, surge: np.ndarray, input_steps: int) -> tuple[list[int], dict[str, int]]:
    index = pd.DatetimeIndex(pd.to_datetime(times))
    surge_values = np.asarray(surge, dtype=float)
    atmospheric_valid = np.zeros(len(index), dtype=bool)
    for start in range(0, len(index), 168):
        chunk = np.asarray(atmosphere[start : start + 168])
        atmospheric_valid[start : start + len(chunk)] = np.isfinite(chunk).all(axis=(1, 2, 3))
    valid: list[int] = []
    skipped = {"non_contiguous": 0, "missing_atmosphere": 0, "missing_surge": 0}
    for target in range(input_steps, len(index)):
        window_times = index[target - input_steps : target + 1]
        if len(window_times) != input_steps + 1 or not np.all(np.diff(window_times.values) == np.timedelta64(1, "h")):
            skipped["non_contiguous"] += 1
            continue
        if not atmospheric_valid[target - input_steps : target].all():
            skipped["missing_atmosphere"] += 1
            continue
        if not np.isfinite(surge_values[target - input_steps : target + 1]).all():
            skipped["missing_surge"] += 1
            continue
        valid.append(target)
    return valid, skipped


class SchemeBDataset(Dataset):
    """Materialise only one rolling window per __getitem__ call."""

    def __init__(
        self,
        atmosphere: np.ndarray | xr.DataArray,
        surge: np.ndarray,
        times: object,
        targets: list[int],
        input_steps: int = 24,
        scalers: dict[str, Standardisation] | None = None,
    ) -> None:
        self.atmosphere = atmosphere
        self.surge = np.asarray(surge, dtype=np.float32)
        self.times = pd.DatetimeIndex(pd.to_datetime(times))
        self.targets = list(targets)
        self.input_steps = int(input_steps)
        self.scalers = scalers

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = self.targets[index]
        atmosphere = np.asarray(self.atmosphere[target - self.input_steps : target], dtype=np.float32)
        history = self.surge[target - self.input_steps : target].copy()
        label = np.float32(self.surge[target])
        if self.scalers:
            atmosphere = self.scalers["atmosphere"].transform(atmosphere)
            history = self.scalers["surge"].transform(history)
            label = np.float32(self.scalers["surge"].transform(np.asarray([label]))[0])
        channels = atmosphere.reshape(-1, atmosphere.shape[-2], atmosphere.shape[-1])
        return torch.from_numpy(channels.copy()), torch.from_numpy(history.copy()), torch.tensor(label)


def build_datasets(
    atmosphere: np.ndarray | xr.DataArray,
    surge: np.ndarray,
    times: object,
    input_steps: int = 24,
    train_ratio: float = 0.8,
) -> tuple[SchemeBDataset, SchemeBDataset, dict[str, Any]]:
    if not 0.5 <= train_ratio < 1:
        raise ValueError("train_ratio must be in [0.5, 1)")
    targets, skipped = valid_targets(times, atmosphere, surge, input_steps)
    if len(targets) < 2:
        raise ValueError("Fewer than two valid Scheme-B samples remain")
    split = max(1, min(len(targets) - 1, int(len(targets) * train_ratio)))
    train_targets, validation_targets = targets[:split], targets[split:]
    # The last training target is the latest observation used to fit either scaler.
    scalers = fit_scalers(atmosphere, np.asarray(surge), train_targets[-1] + 1)
    train = SchemeBDataset(atmosphere, surge, times, train_targets, input_steps, scalers)
    validation = SchemeBDataset(atmosphere, surge, times, validation_targets, input_steps, scalers)
    report = {
        "input_steps": input_steps,
        "valid_samples": len(targets),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "skipped": skipped,
        "train_time_range": [str(train.times[train_targets[0]]), str(train.times[train_targets[-1]])],
        "validation_time_range": [str(validation.times[validation_targets[0]]), str(validation.times[validation_targets[-1]])],
        "scalers": {name: scaler.state_dict() for name, scaler in scalers.items()},
    }
    return train, validation, report
