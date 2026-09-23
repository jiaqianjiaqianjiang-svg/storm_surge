"""Create the main journal figures for the formal Xiamen forecast experiment.

The script reads existing metrics and prediction tables. It never loads the
large prepared ERA5 dataset, model checkpoints, or retrains a model.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import io_utils
from .plot_peak_analysis import plot_peak_analysis
from .plot_residual_diagnostics import plot_residual_diagnostics
from .style import (
    apply_axis_style,
    close_figure,
    label_panels,
    model_color,
    model_label,
    save_figure,
    setup_journal_style,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ORDER = (
    "persistence",
    "ridge",
    "surge_mlp",
    "era5_cnn",
    "cnn",
    "cnn_lstm",
    "cnn_gru",
    "tcn",
    "transformer",
)
ROLLING_MODELS = (*MODEL_ORDER, "cnn_gru_rollout6")
COMPETITIVE_MODELS = (
    "ridge",
    "cnn",
    "cnn_lstm",
    "cnn_gru",
    "transformer",
    "cnn_gru_rollout6",
)
EXTREME_MODELS = ("ridge", "cnn_gru", "cnn_gru_rollout6")


@dataclass
class XiamenSources:
    validation_metrics: Path | None = None
    rolling_rmse: Path | None = None
    rolling_metrics: Path | None = None
    rollout_loss: Path | None = None
    baseline_predictions: Path | None = None
    model_predictions: dict[str, Path] = field(default_factory=dict)
    rolling_predictions: Path | None = None
    strong_events: list[Path] = field(default_factory=list)


def _repository_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _first_file(*candidates: Path | None) -> Path | None:
    return next((path for path in candidates if path is not None and path.is_file()), None)


def discover_sources(
    project_root: Path,
    station: str = "xiamen",
    validation_year: int = 1996,
    seed: int = 42,
    result_package: Path | None = None,
    split: str = "validation",
) -> XiamenSources:
    """Locate full laboratory outputs, falling back to the compact Git package."""

    project_root = Path(project_root)
    repository = _repository_root(project_root)
    if result_package is None and repository is not None:
        result_package = (
            repository
            / "reports"
            / "experiment_results"
            / f"xiamen_short_term_{validation_year}_seed{seed}"
        )
    package = Path(result_package) if result_package is not None else None
    experiment_root = project_root / "outputs" / "experiments" / station
    comparison = experiment_root / "model_comparison"
    rolling = experiment_root / f"rolling_{validation_year}_seed{seed}"
    model_root = project_root / "models" / station
    formal = model_root / f"formal_seed{seed}"
    rollout = formal / "cnn_gru_rollout6"

    source = XiamenSources(
        validation_metrics=_first_file(
            comparison / f"{split}_model_metrics.csv",
            package / f"{split}_model_metrics.csv" if package else None,
        ),
        rolling_rmse=_first_file(
            rolling / "rolling_rmse_table.csv",
            package / "rolling_rmse_table.csv" if package else None,
        ),
        rolling_metrics=_first_file(
            rolling / "rolling_metrics_long.csv",
            package / "rolling_metrics_long.csv" if package else None,
        ),
        rollout_loss=_first_file(
            rollout / "loss_history.csv",
            package / "rollout_loss_history.csv" if package else None,
        ),
        baseline_predictions=_first_file(
            model_root / "baselines" / f"{split}_baseline_predictions.csv"
        ),
        rolling_predictions=_first_file(
            rolling / "rolling_predictions_selected_leads.csv"
        ),
    )
    for model in MODEL_ORDER[2:]:
        path = _first_file(formal / model / f"{split}_predictions.csv")
        if path is not None:
            source.model_predictions[model] = path

    event_root = rolling if list(rolling.glob("strong_event_*.png")) else package
    if event_root is not None and event_root.is_dir():
        source.strong_events = sorted(event_root.glob("strong_event_*.png"))
    return source


def _read_csv(path: Path | None, description: str) -> pd.DataFrame | None:
    if path is None:
        io_utils.warn(f"Missing {description}; dependent figures will be skipped")
        return None
    frame = io_utils.safe_read_csv(path)
    if frame is None or frame.empty:
        io_utils.warn(f"Empty {description}: {path}")
        return None
    return frame


def _labels(language: str) -> dict[str, str]:
    if language == "zh":
        return {
            "pearson": "相关系数 r",
            "rmse": "均方根误差 (cm)",
            "mae": "平均绝对误差 (cm)",
            "bias": "偏差 (cm)",
            "lead": "预报提前量 (h)",
            "overall": "全部模型",
            "competitive": "主要候选模型",
            "reduction": "相对普通 CNN-GRU 的 RMSE 降幅 (%)",
            "top5": "强增水 Top 5% RMSE (cm)",
            "rapid": "快速上涨过程 RMSE (cm)",
            "scaled_mse": "标准化 MSE",
            "val_loss": "递归验证损失（标准化 MSE）",
            "val_rmse": "递归验证 RMSE (cm)",
            "teacher": "Teacher forcing 比例",
            "observed": "观测值 (cm)",
            "predicted": "预测值 (cm)",
            "count": "样本数",
        }
    return {
        "pearson": "Pearson r",
        "rmse": "RMSE (cm)",
        "mae": "MAE (cm)",
        "bias": "Bias (cm)",
        "lead": "Lead time (h)",
        "overall": "All models",
        "competitive": "Main candidate models",
        "reduction": "RMSE reduction vs CNN-GRU (%)",
        "top5": "Top 5% surge RMSE (cm)",
        "rapid": "Rapid-rise RMSE (cm)",
        "scaled_mse": "Scaled MSE",
        "val_loss": "Recursive validation loss (scaled MSE)",
        "val_rmse": "Recursive validation RMSE (cm)",
        "teacher": "Teacher-forcing ratio",
        "observed": "Observed (cm)",
        "predicted": "Predicted (cm)",
        "count": "Count",
    }


def _ordered(frame: pd.DataFrame, order: tuple[str, ...]) -> pd.DataFrame:
    positions = {name: index for index, name in enumerate(order)}
    out = frame[frame["model"].isin(order)].copy()
    out["_order"] = out["model"].map(positions)
    return out.sort_values("_order").drop(columns="_order")


def _annotate_bars(ax: plt.Axes, values: np.ndarray, decimals: int = 2) -> None:
    finite = values[np.isfinite(values)]
    span = float(np.ptp(finite)) if finite.size else 1.0
    offset = max(span * 0.025, 0.015)
    for bar, value in zip(ax.patches, values):
        if not np.isfinite(value):
            continue
        y = value + offset if value >= 0 else value - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=6.5,
            rotation=90 if len(values) > 7 else 0,
        )


def plot_one_step_summary(
    metrics: pd.DataFrame,
    output_dir: Path,
    language: str = "en",
) -> dict[str, Path]:
    required = {"model", "pearson_r", "rmse_cm", "mae_cm", "bias_cm"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"validation metrics missing columns: {sorted(missing)}")
    setup_journal_style(font_size=8.0, language=language)
    text = _labels(language)
    table = _ordered(metrics, MODEL_ORDER)
    labels = [model_label(name) for name in table["model"]]
    colors = [model_color(name) for name in table["model"]]
    panels = (
        ("pearson_r", text["pearson"], 3),
        ("rmse_cm", text["rmse"], 2),
        ("mae_cm", text["mae"], 2),
        ("bias_cm", text["bias"], 2),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    for ax, (column, ylabel, decimals) in zip(axes.ravel(), panels):
        values = table[column].to_numpy(float)
        ax.bar(labels, values, color=colors, edgecolor="#333333", linewidth=0.45)
        ax.axhline(0, color="#333333", linewidth=0.7)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        _annotate_bars(ax, values, decimals)
        apply_axis_style(ax)
    label_panels(axes.ravel(), x=-0.10, y=1.02)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "fig01_one_step_model_comparison")
    close_figure(fig)
    return paths


def _plot_rmse_lines(ax: plt.Axes, table: pd.DataFrame, models: tuple[str, ...]) -> None:
    for model in models:
        if model not in table.columns:
            continue
        emphasized = model == "cnn_gru_rollout6"
        ax.plot(
            table["lead_hours"],
            table[model],
            marker="o",
            markersize=3.2,
            linewidth=2.2 if emphasized else 1.15,
            color=model_color(model),
            label=model_label(model),
            alpha=1.0 if emphasized else 0.82,
            zorder=4 if emphasized else 2,
        )


def plot_rolling_overview(
    rolling_rmse: pd.DataFrame,
    output_dir: Path,
    language: str = "en",
) -> dict[str, Path]:
    if "lead_hours" not in rolling_rmse:
        raise ValueError("rolling RMSE table missing lead_hours")
    setup_journal_style(font_size=8.0, language=language)
    text = _labels(language)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    _plot_rmse_lines(axes[0], rolling_rmse, ROLLING_MODELS)
    _plot_rmse_lines(axes[1], rolling_rmse, COMPETITIVE_MODELS)
    for ax, title in zip(axes, (text["overall"], text["competitive"])):
        ax.set_xlabel(text["lead"])
        ax.set_ylabel(text["rmse"])
        ax.set_xticks(rolling_rmse["lead_hours"])
        ax.set_title(title, loc="left")
        apply_axis_style(ax)
    axes[0].legend(loc="upper left", ncol=2, fontsize=6.2)
    axes[1].legend(loc="upper left", ncol=2, fontsize=6.4)
    label_panels(axes, x=-0.12, y=1.03)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "fig02_rmse_by_lead")
    close_figure(fig)
    return paths


def plot_rollout_gain(
    rolling_rmse: pd.DataFrame,
    output_dir: Path,
    language: str = "en",
) -> dict[str, Path]:
    required = {"lead_hours", "cnn_gru", "cnn_gru_rollout6"}
    if not required.issubset(rolling_rmse.columns):
        raise ValueError(f"rolling table needs columns {sorted(required)}")
    setup_journal_style(font_size=8.0, language=language)
    text = _labels(language)
    leads = rolling_rmse["lead_hours"].to_numpy(float)
    base = rolling_rmse["cnn_gru"].to_numpy(float)
    rollout = rolling_rmse["cnn_gru_rollout6"].to_numpy(float)
    reduction = 100.0 * (1.0 - rollout / base)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(leads, base, marker="o", color=model_color("cnn_gru"), label=model_label("cnn_gru"))
    axes[0].plot(
        leads,
        rollout,
        marker="o",
        linewidth=2.2,
        color=model_color("cnn_gru_rollout6"),
        label=model_label("cnn_gru_rollout6"),
    )
    axes[0].set(xlabel=text["lead"], ylabel=text["rmse"])
    axes[0].set_xticks(leads)
    axes[0].legend(loc="upper left")
    apply_axis_style(axes[0])

    lead_labels = [f"{int(value)}" if float(value).is_integer() else f"{value:g}" for value in leads]
    bars = axes[1].bar(lead_labels, reduction, color=model_color("cnn_gru_rollout6"), edgecolor="#333333", linewidth=0.5)
    axes[1].axhline(0, color="#333333", linewidth=0.7)
    axes[1].set(xlabel=text["lead"], ylabel=text["reduction"])
    for bar, value in zip(bars, reduction):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.35, f"{value:.1f}%", ha="center", va="bottom", fontsize=6.5)
    apply_axis_style(axes[1])
    label_panels(axes, x=-0.12, y=1.03)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "fig03_rollout_improvement")
    close_figure(fig)
    return paths


def plot_extreme_metrics(
    rolling_metrics: pd.DataFrame,
    output_dir: Path,
    language: str = "en",
) -> dict[str, Path]:
    required = {"lead_hours", "model", "rmse_cm", "top5_rmse_cm", "rapid_rise_rmse_cm"}
    missing = required.difference(rolling_metrics.columns)
    if missing:
        raise ValueError(f"rolling metrics missing columns: {sorted(missing)}")
    setup_journal_style(font_size=8.0, language=language)
    text = _labels(language)
    panels = (
        ("rmse_cm", text["rmse"]),
        ("top5_rmse_cm", text["top5"]),
        ("rapid_rise_rmse_cm", text["rapid"]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharex=True)
    for ax, (metric, ylabel) in zip(axes, panels):
        for model in EXTREME_MODELS:
            subset = rolling_metrics[rolling_metrics["model"] == model].sort_values("lead_hours")
            if subset.empty:
                continue
            ax.plot(
                subset["lead_hours"],
                subset[metric],
                marker="o",
                markersize=3,
                linewidth=2.0 if model == "cnn_gru_rollout6" else 1.25,
                color=model_color(model),
                label=model_label(model),
            )
        ax.set_xlabel(text["lead"])
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(rolling_metrics["lead_hours"].unique()))
        apply_axis_style(ax)
    axes[0].legend(loc="upper left", fontsize=6.3)
    label_panels(axes, x=-0.13, y=1.03)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "fig04_extreme_and_rapid_rise_metrics")
    close_figure(fig)
    return paths


def plot_rollout_training(
    history: pd.DataFrame,
    output_dir: Path,
    language: str = "en",
) -> dict[str, Path]:
    required = {
        "epoch",
        "teacher_forcing_ratio",
        "train_scaled_mse",
        "validation_recursive_scaled_mse",
        "validation_recursive_rmse_cm",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"rollout history missing columns: {sorted(missing)}")
    setup_journal_style(font_size=8.0, language=language)
    text = _labels(language)
    epochs = history["epoch"].to_numpy(float)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    axes[0].plot(epochs, history["train_scaled_mse"], marker="o", label="Train")
    axes[0].plot(epochs, history["validation_recursive_scaled_mse"], marker="o", label="Validation")
    axes[0].set(xlabel="Epoch", ylabel=text["scaled_mse"])
    axes[0].legend(loc="best")
    apply_axis_style(axes[0])

    rmse = history["validation_recursive_rmse_cm"].to_numpy(float)
    axes[1].plot(epochs, rmse, marker="o", color=model_color("cnn_gru_rollout6"))
    finite = np.isfinite(rmse)
    if finite.any():
        best_index = int(np.nanargmin(rmse))
        axes[1].scatter(epochs[best_index], rmse[best_index], s=35, facecolor="white", edgecolor="black", zorder=5)
        axes[1].annotate(
            f"Best: {rmse[best_index]:.2f} cm",
            (epochs[best_index], rmse[best_index]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=6.5,
        )
    axes[1].set(xlabel="Epoch", ylabel=text["val_rmse"])
    apply_axis_style(axes[1])

    axes[2].bar(epochs, history["teacher_forcing_ratio"], color="#999999", edgecolor="#333333", linewidth=0.4)
    axes[2].set(xlabel="Epoch", ylabel=text["teacher"])
    axes[2].set_ylim(0, 1.08)
    apply_axis_style(axes[2])
    label_panels(axes, x=-0.14, y=1.03)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "fig05_rollout_training")
    close_figure(fig)
    return paths


def _prediction_pair(path: Path, predicted_column: str | None = None) -> pd.DataFrame | None:
    raw = io_utils.safe_read_csv(path)
    if raw is None:
        return None
    observed_column = next((name for name in ("observed_m", "observed", "observation", "y_true") if name in raw), None)
    if predicted_column is None:
        predicted_column = next((name for name in ("predicted_m", "predicted", "prediction", "y_pred") if name in raw), None)
    if observed_column is None or predicted_column not in raw:
        io_utils.warn(f"Missing prediction columns in {path}")
        return None
    observed = pd.to_numeric(raw[observed_column], errors="coerce").to_numpy(float)
    predicted = pd.to_numeric(raw[predicted_column], errors="coerce").to_numpy(float)
    scale = 100.0 if observed_column.endswith("_m") else io_utils.unit_scale_to_cm(observed)[2]
    frame = pd.DataFrame({"observed": observed * scale, "predicted": predicted * scale})
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def _scatter_panel(ax: plt.Axes, frame: pd.DataFrame, title: str, labels: dict[str, str]) -> None:
    obs = frame["observed"].to_numpy(float)
    pred = frame["predicted"].to_numpy(float)
    metric = io_utils.compute_metrics_from_predictions(frame)
    hb = ax.hexbin(obs, pred, gridsize=42, mincnt=1, cmap="Blues", linewidths=0)
    low = float(np.nanmin([obs.min(), pred.min()]))
    high = float(np.nanmax([obs.max(), pred.max()]))
    pad = max((high - low) * 0.03, 0.5)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="black", linestyle="--", linewidth=0.85)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left")
    ax.set_xlabel(labels["observed"])
    ax.set_ylabel(labels["predicted"])
    ax.text(
        0.04,
        0.96,
        f"r = {metric['pearson_r']:.3f}\nRMSE = {metric['rmse']:.2f} cm\nMAE = {metric['mae']:.2f} cm\nn = {metric['n']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
    )
    apply_axis_style(ax, grid=False)
    ax.figure.colorbar(hb, ax=ax, label=labels["count"], fraction=0.046, pad=0.03)


def plot_validation_scatter(
    sources: XiamenSources,
    output_dir: Path,
    language: str = "en",
) -> dict[str, Path] | None:
    frames: list[tuple[str, pd.DataFrame]] = []
    if sources.baseline_predictions is not None:
        ridge = _prediction_pair(sources.baseline_predictions, "ridge_m")
        if ridge is not None:
            frames.append(("ridge", ridge))
    for model in ("cnn_lstm", "cnn_gru"):
        path = sources.model_predictions.get(model)
        if path is not None:
            frame = _prediction_pair(path)
            if frame is not None:
                frames.append((model, frame))
    if len(frames) < 2:
        io_utils.warn("Need at least two validation prediction tables for scatter comparison")
        return None
    setup_journal_style(font_size=8.0, language=language)
    labels = _labels(language)
    fig, axes = plt.subplots(1, len(frames), figsize=(2.75 * len(frames), 2.75), squeeze=False)
    for ax, (model, frame) in zip(axes.ravel(), frames):
        _scatter_panel(ax, frame, model_label(model), labels)
    label_panels(axes.ravel(), x=-0.16, y=1.03)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "fig06_validation_scatter")
    close_figure(fig)
    return paths


def plot_rollout_scatter_by_lead(
    prediction_csv: Path,
    output_dir: Path,
    language: str = "en",
    model: str = "cnn_gru_rollout6",
) -> dict[str, Path] | None:
    raw = io_utils.safe_read_csv(prediction_csv)
    if raw is None:
        return None
    observed_column = "observed_m" if "observed_m" in raw else "observed"
    predicted_column = f"{model}_m" if f"{model}_m" in raw else model
    required = {"lead_hours", observed_column, predicted_column}
    if not required.issubset(raw.columns):
        io_utils.warn(f"Rolling prediction table missing columns: {sorted(required.difference(raw.columns))}")
        return None
    setup_journal_style(font_size=8.0, language=language)
    labels = _labels(language)
    available = sorted(pd.to_numeric(raw["lead_hours"], errors="coerce").dropna().astype(int).unique())
    preferred = [lead for lead in (1, 6, 24, 72) if lead in available]
    leads = preferred if preferred else available[:4]
    if not leads:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.4), squeeze=False)
    flat = axes.ravel()
    for ax, lead in zip(flat, leads):
        subset = raw[pd.to_numeric(raw["lead_hours"], errors="coerce") == lead]
        scale = 100.0 if observed_column.endswith("_m") else io_utils.unit_scale_to_cm(subset[observed_column])[2]
        frame = pd.DataFrame(
            {
                "observed": pd.to_numeric(subset[observed_column], errors="coerce") * scale,
                "predicted": pd.to_numeric(subset[predicted_column], errors="coerce") * scale,
            }
        ).dropna()
        _scatter_panel(ax, frame, f"{lead} h", labels)
    for ax in flat[len(leads):]:
        ax.set_visible(False)
    label_panels(flat[: len(leads)], x=-0.13, y=1.03)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "fig07_rollout_scatter_by_lead")
    close_figure(fig)
    return paths


def _manifest_row(name: str, source: str, path: Path | None, status: str, warning: str = "") -> dict[str, str]:
    return {
        "figure_name": name,
        "source": source,
        "output_png": str(path) if path else "",
        "status": status,
        "warning": warning,
    }


def _run_figure(
    rows: list[dict[str, str]],
    name: str,
    source: str,
    function: Callable[..., dict[str, Path] | None],
    *args,
    **kwargs,
) -> None:
    try:
        paths = function(*args, **kwargs)
        if paths and paths.get("png"):
            rows.append(_manifest_row(name, source, paths["png"], "ok"))
        else:
            rows.append(_manifest_row(name, source, None, "skipped", "missing required input"))
    except Exception as exc:
        io_utils.warn(f"{name} failed: {exc}")
        rows.append(_manifest_row(name, source, None, "failed", str(exc)))


def _write_guide(
    output_dir: Path,
    manifest: pd.DataFrame,
    evaluation_year: int,
    split: str,
) -> None:
    completed = manifest.loc[manifest["status"] == "ok", "figure_name"].tolist()
    skipped = manifest.loc[manifest["status"] != "ok", ["figure_name", "warning"]]
    lines = [
        "# Xiamen journal figure guide",
        "",
        f"These figures use existing {evaluation_year} {split} and historical-hindcast results. No model was retrained.",
        "The rolling experiment uses known future ERA5 reanalysis forcing and is not an operational forecast.",
        "",
        "## Completed figures",
        "",
        *[f"- `{name}`" for name in completed],
    ]
    if not skipped.empty:
        lines.extend(["", "## Skipped figures", ""])
        lines.extend(f"- `{row.figure_name}`: {row.warning}" for row in skipped.itertuples(index=False))
    lines.extend(
        [
            "",
            "## Recommended presentation order",
            "",
            "1. `fig01_one_step_model_comparison`: one-hour model selection.",
            "2. `fig02_rmse_by_lead`: error growth across forecast lead times.",
            "3. `fig03_rollout_improvement`: direct evidence that rollout training reduces accumulated error.",
            "4. `fig04_extreme_and_rapid_rise_metrics`: performance for strong surges and rapid rises.",
            "5. `fig06_validation_scatter` and `fig07_rollout_scatter_by_lead`: agreement and spread diagnostics.",
            "6. `fig08_cnn_gru_residual_diagnostics` and `fig09_cnn_gru_peak_analysis`: detailed error diagnostics.",
            "7. `fig10_strong_event_*`: representative strong-event hindcasts.",
            "",
        ]
    )
    (output_dir / "FIGURE_GUIDE.md").write_text("\n".join(lines), encoding="utf-8")


def make_xiamen_journal_figures(
    project_root: Path = PROJECT_ROOT,
    output_dir: Path | None = None,
    result_package: Path | None = None,
    station: str = "xiamen",
    validation_year: int = 1996,
    seed: int = 42,
    language: str = "en",
    split: str | None = None,
) -> pd.DataFrame:
    project_root = Path(project_root)
    split = split or ("validation" if validation_year == 1996 else "test")
    output_dir = Path(output_dir) if output_dir else (
        project_root / "outputs" / "journal_figures" / f"{station}_{validation_year}_seed{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(
        project_root, station, validation_year, seed, result_package, split
    )
    rows: list[dict[str, str]] = []

    validation = _read_csv(sources.validation_metrics, f"{split} model metrics")
    if validation is not None:
        _run_figure(rows, "fig01_one_step_model_comparison", str(sources.validation_metrics), plot_one_step_summary, validation, output_dir, language)
    else:
        rows.append(_manifest_row("fig01_one_step_model_comparison", "", None, "skipped", f"missing {split} metrics"))

    rolling_rmse = _read_csv(sources.rolling_rmse, "rolling RMSE table")
    if rolling_rmse is not None:
        _run_figure(rows, "fig02_rmse_by_lead", str(sources.rolling_rmse), plot_rolling_overview, rolling_rmse, output_dir, language)
        _run_figure(rows, "fig03_rollout_improvement", str(sources.rolling_rmse), plot_rollout_gain, rolling_rmse, output_dir, language)
    else:
        for name in ("fig02_rmse_by_lead", "fig03_rollout_improvement"):
            rows.append(_manifest_row(name, "", None, "skipped", "missing rolling RMSE table"))

    rolling_metrics = _read_csv(sources.rolling_metrics, "rolling metric table")
    if rolling_metrics is not None:
        _run_figure(rows, "fig04_extreme_and_rapid_rise_metrics", str(sources.rolling_metrics), plot_extreme_metrics, rolling_metrics, output_dir, language)
    else:
        rows.append(_manifest_row("fig04_extreme_and_rapid_rise_metrics", "", None, "skipped", "missing rolling metrics"))

    history = _read_csv(sources.rollout_loss, "rollout loss history")
    if history is not None:
        _run_figure(rows, "fig05_rollout_training", str(sources.rollout_loss), plot_rollout_training, history, output_dir, language)
    else:
        rows.append(_manifest_row("fig05_rollout_training", "", None, "skipped", "missing rollout history"))

    _run_figure(rows, "fig06_validation_scatter", "validation prediction CSVs", plot_validation_scatter, sources, output_dir, language)
    if sources.rolling_predictions is not None:
        _run_figure(rows, "fig07_rollout_scatter_by_lead", str(sources.rolling_predictions), plot_rollout_scatter_by_lead, sources.rolling_predictions, output_dir, language)
    else:
        rows.append(_manifest_row("fig07_rollout_scatter_by_lead", "", None, "skipped", "missing rolling prediction table"))

    cnn_gru_predictions = sources.model_predictions.get("cnn_gru")
    if cnn_gru_predictions is not None:
        _run_figure(
            rows,
            "fig08_cnn_gru_residual_diagnostics",
            str(cnn_gru_predictions),
            plot_residual_diagnostics,
            cnn_gru_predictions,
            output_dir,
            "fig08_cnn_gru_residual_diagnostics",
            language,
        )
        _run_figure(
            rows,
            "fig09_cnn_gru_peak_analysis",
            str(cnn_gru_predictions),
            plot_peak_analysis,
            cnn_gru_predictions,
            output_dir,
            figure_name="fig09_cnn_gru_peak_analysis",
            language=language,
        )
    else:
        for name in ("fig08_cnn_gru_residual_diagnostics", "fig09_cnn_gru_peak_analysis"):
            rows.append(_manifest_row(name, "", None, "skipped", "missing CNN-GRU validation predictions"))

    for index, source in enumerate(sources.strong_events, start=1):
        destination = output_dir / f"fig10_strong_event_{index}.png"
        shutil.copy2(source, destination)
        rows.append(_manifest_row(destination.stem, str(source), destination, "ok"))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    _write_guide(output_dir, manifest, validation_year, split)
    print(manifest.to_string(index=False))
    print(f"Figures: {output_dir}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--result-package", type=Path)
    parser.add_argument("--station", default="xiamen")
    parser.add_argument("--validation-year", type=int, default=1996)
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_xiamen_journal_figures(
        project_root=args.project_root,
        output_dir=args.output_dir,
        result_package=args.result_package,
        station=args.station,
        validation_year=args.validation_year,
        seed=args.seed,
        language=args.language,
        split=args.split,
    )


if __name__ == "__main__":
    main()
