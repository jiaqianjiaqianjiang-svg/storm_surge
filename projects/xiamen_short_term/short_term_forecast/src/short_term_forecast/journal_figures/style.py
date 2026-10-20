"""Shared journal-style plotting settings.

The helpers here intentionally depend only on matplotlib/numpy so the figure
scripts can run on the remote experiment machine without extra plotting
packages such as seaborn.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


DPI = 400

MODEL_COLORS = {
    "observed": "#202428",
    "observation": "#202428",
    "persistence": "#8B9298",
    "ridge": "#555E66",
    "surge_mlp": "#899DA5",
    "era5_cnn": "#A7A9A5",
    "cnn": "#698896",
    "cnn_lstm": "#3F7180",
    "cnn_gru": "#20566B",
    "cnn_gru_rollout6": "#A74F49",
    "tcn": "#858F88",
    "transformer": "#8C8077",
    "forecast": "#20566B",
    "predicted": "#20566B",
}

POSITIVE_ERROR_COLOR = "#20566B"
NEGATIVE_ERROR_COLOR = "#A74F49"

MODEL_LINESTYLES = {
    "persistence": "--", "ridge": "-.", "surge_mlp": ":",
    "era5_cnn": "--", "cnn": "-", "cnn_lstm": "--",
    "cnn_gru": "-", "cnn_gru_rollout6": "-", "tcn": ":",
    "transformer": "-.",
}
MODEL_MARKERS = {
    "persistence": "^", "ridge": "s", "surge_mlp": "D",
    "era5_cnn": "v", "cnn": "o", "cnn_lstm": "P",
    "cnn_gru": "o", "cnn_gru_rollout6": "s", "tcn": "X",
    "transformer": "d",
}

MODEL_LABELS = {
    "observed": "Observed",
    "persistence": "Persistence",
    "ridge": "Ridge",
    "surge_mlp": "Surge MLP",
    "era5_cnn": "ERA5 CNN",
    "cnn": "CNN",
    "cnn_lstm": "CNN-LSTM",
    "cnn_gru": "CNN-GRU",
    "cnn_gru_rollout6": "CNN-GRU rollout-6",
    "tcn": "TCN",
    "transformer": "Transformer",
    "forecast": "Forecast",
    "predicted": "Predicted",
}

PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)")


def _font_available(name: str) -> bool:
    for font in font_manager.fontManager.ttflist:
        if font.name.lower() == name.lower():
            return True
    return False


def resolve_font(language: str = "en") -> str:
    """Return a readable installed font for English or Chinese labels."""

    candidates = ["Arial", "DejaVu Sans"]
    if language == "zh":
        candidates = [
            "Microsoft YaHei",
            "PingFang SC",
            "Noto Sans CJK SC",
            "SimHei",
            *candidates,
        ]
    return next((name for name in candidates if _font_available(name)), "DejaVu Sans")


def setup_journal_style(font_size: float = 8.0, language: str = "en") -> None:
    """Apply consistent matplotlib rcParams for journal figures."""

    font = resolve_font(language)
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": font_size - 1,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "grid.color": "#E3E6E8",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    if language == "zh":
        plt.rcParams["axes.unicode_minus"] = False


def cm_label(text: str = "Storm surge") -> str:
    return f"{text} (cm)"


def metric_label(metric: str) -> str:
    labels = {
        "pearson_r": "Pearson r",
        "rmse": "RMSE (cm)",
        "mae": "MAE (cm)",
        "rrmse": "RRMSE (%)",
        "bias": "Bias (cm)",
        "error": "Error (cm)",
        "absolute_error": "Absolute error (cm)",
    }
    return labels.get(metric, metric)


def model_color(name: str | None) -> str:
    if not name:
        return MODEL_COLORS["forecast"]
    return MODEL_COLORS.get(str(name).lower(), MODEL_COLORS["forecast"])


def model_line_style(name: str) -> dict[str, str]:
    key = str(name).lower()
    return {
        "color": model_color(key),
        "linestyle": MODEL_LINESTYLES.get(key, "-"),
        "marker": MODEL_MARKERS.get(key, "o"),
    }


def model_label(name: str | None) -> str:
    if not name:
        return "Forecast"
    key = str(name).lower()
    return MODEL_LABELS.get(key, str(name))


def apply_axis_style(ax: plt.Axes, grid: bool = True) -> None:
    """Apply shared axis styling."""

    if grid:
        ax.grid(axis="y", color="#E3E6E8", linewidth=0.55, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(0.8)


def format_lead_axis(ax: plt.Axes, leads: Iterable[float], language: str = "en") -> None:
    """Show selected lead times legibly without changing their values."""

    values = sorted({float(value) for value in leads if np.isfinite(value) and value > 0})
    ax.set_xscale("log")
    ax.set_xticks(values)
    ax.set_xticklabels([f"{value:g}" for value in values])
    ax.minorticks_off()
    ax.set_xlabel("提前量（h，对数刻度）" if language == "zh" else "Lead time (h, log scale)")


def label_panels(axes: Iterable[plt.Axes], x: float = -0.12, y: float = 1.04) -> None:
    """Add panel labels such as (a), (b), (c), (d)."""

    for label, ax in zip(PANEL_LABELS, axes):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
        )


def format_time_axis(ax: plt.Axes) -> None:
    ax.figure.autofmt_xdate(rotation=25, ha="right")


def add_bar_labels(ax: plt.Axes, values: Iterable[float], fmt: str = "{:.2f}") -> None:
    values_array = np.asarray(list(values), dtype=float)
    finite = values_array[np.isfinite(values_array)]
    span = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size else 1.0
    offset = span * 0.03 if span > 0 else 0.03
    for patch, value in zip(ax.patches, values_array):
        if not np.isfinite(value):
            continue
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=plt.rcParams["font.size"] - 1,
        )


def save_figure(fig: plt.Figure, output_dir: Path, figure_name: str) -> dict[str, Path]:
    """Save one figure as a high-resolution PNG and return the path."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{figure_name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return {"png": path}


def close_figure(fig: plt.Figure) -> None:
    plt.close(fig)


def ensure_writable_output_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None
