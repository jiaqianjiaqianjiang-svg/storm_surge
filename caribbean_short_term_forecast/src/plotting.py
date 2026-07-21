"""PNG-only forecast plotting helpers (400 dpi by default)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: str | Path, dpi: int) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".png":
        raise ValueError("Plot output must use the .png suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return destination


def observed_vs_predicted(dates: object, observed_m: object, predicted_m: object, path: str | Path, dpi: int = 400) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(pd.to_datetime(dates), np.asarray(observed_m) * 100, label="Observed", linewidth=1.5)
    ax.plot(pd.to_datetime(dates), np.asarray(predicted_m) * 100, label="Predicted", linewidth=1.5)
    ax.set(xlabel="Time (UTC)", ylabel="Storm surge (cm)")
    ax.grid(alpha=0.25); ax.legend()
    return _save(fig, path, dpi)

def validation_scatter(observed_m: object, predicted_m: object, path: str | Path, dpi: int = 400) -> Path:
    observed, predicted = np.asarray(observed_m) * 100, np.asarray(predicted_m) * 100
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(observed, predicted, s=12, alpha=0.55)
    finite = np.concatenate([observed[np.isfinite(observed)], predicted[np.isfinite(predicted)]])
    if len(finite):
        low, high = finite.min(), finite.max(); ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set(xlabel="Observed (cm)", ylabel="Predicted (cm)"); ax.grid(alpha=0.25)
    return _save(fig, path, dpi)


def rolling_error(frame: pd.DataFrame, path: str | Path, dpi: int = 400) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(frame.lead_time, frame.error * 100, marker="o")
    ax.axhline(0, color="black", linewidth=0.8); ax.set(xlabel="Lead time (h)", ylabel="Error (cm)"); ax.grid(alpha=0.25)
    return _save(fig, path, dpi)


def cumulative_error(frame: pd.DataFrame, path: str | Path, dpi: int = 400) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(frame.lead_time, frame.absolute_error.fillna(0).cumsum() * 100, marker="o")
    ax.set(xlabel="Lead time (h)", ylabel="Cumulative absolute error (cm)"); ax.grid(alpha=0.25)
    return _save(fig, path, dpi)
