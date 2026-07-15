"""Robust readers for remote short-term forecast result directories."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KNOWN_RESULT_FILES = {
    "metrics": "metrics.json",
    "val_predictions": "val_predictions.csv",
    "rolling_forecast": "rolling_forecast.csv",
    "predictions": "predictions.csv",
    "loss_history": "loss_history.csv",
}

OBSERVED_ALIASES = ("observed", "observation", "obs", "y_true", "target", "actual")
PREDICTED_ALIASES = ("predicted", "prediction", "y_pred", "forecast", "pred", "model")
PERSISTENCE_ALIASES = ("persistence", "persistence_prediction", "persistence_forecast", "baseline")
TIME_ALIASES = ("datetime", "time", "date", "timestamp")
LEAD_ALIASES = ("lead_time", "lead_step", "forecast_step", "step", "lead")


@dataclass
class ExperimentResult:
    """Normalized description of one result directory."""

    name: str
    path: Path
    files: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, float | str | int | None] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def input_steps(self) -> int | None:
        value = self.metadata.get("input_steps")
        return int(value) if value is not None else None

    @property
    def forecast_steps(self) -> int | None:
        value = self.metadata.get("forecast_steps")
        return int(value) if value is not None else None

    @property
    def model_name(self) -> str | None:
        value = self.metadata.get("model_name")
        return str(value) if value is not None else None

    @property
    def is_rolling(self) -> bool:
        return "rolling_forecast" in self.files or self.forecast_steps is not None or "rolling" in self.name.lower()


def warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    print(f"[WARN] {message}")


def _find_alias(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for key, original in normalized.items():
        for alias in aliases:
            if key.endswith(alias) or alias in key:
                return original
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _looks_like_meters(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    q95 = float(np.nanpercentile(np.abs(finite), 95))
    return q95 <= 5.0


def unit_scale_to_cm(values: np.ndarray | pd.Series) -> tuple[np.ndarray, str, float]:
    """Return values in cm, detected unit label, and applied scale."""

    array = np.asarray(values, dtype=float)
    if _looks_like_meters(array):
        return array * 100.0, "m", 100.0
    return array, "cm", 1.0


def metric_to_cm(value: object) -> float:
    numeric = float(value)
    return numeric * 100.0 if np.isfinite(numeric) and abs(numeric) <= 5.0 else numeric


def rrmse_to_percent(value: object) -> float:
    numeric = float(value)
    return numeric * 100.0 if np.isfinite(numeric) and abs(numeric) <= 2.0 else numeric


def safe_read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warn(f"Cannot read JSON {path}: {exc}")
        return {}


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warn(f"Cannot read CSV {path}: {exc}")
        return None


def infer_input_steps(name: str, metrics: dict[str, object] | None = None) -> int | None:
    metrics = metrics or {}
    for key in ("input_steps", "input_window", "window", "t"):
        if key in metrics and pd.notna(metrics[key]):
            try:
                return int(metrics[key])
            except (TypeError, ValueError):
                pass
    match = re.search(r"(?:^|_)t(\d+)(?:_|$)", name)
    return int(match.group(1)) if match else None


def infer_forecast_steps(name: str, metrics: dict[str, object] | None = None, df: pd.DataFrame | None = None) -> int | None:
    metrics = metrics or {}
    for key in ("forecast_steps", "steps", "n_steps"):
        if key in metrics and pd.notna(metrics[key]):
            try:
                return int(metrics[key])
            except (TypeError, ValueError):
                pass
    match = re.search(r"(?:^|_)n(\d+)(?:_|$)", name)
    if match:
        return int(match.group(1))
    if df is not None and not df.empty:
        lead_col = _find_alias(df.columns, LEAD_ALIASES)
        if lead_col is not None:
            values = pd.to_numeric(df[lead_col], errors="coerce")
            if values.notna().any():
                return int(values.max())
        return int(len(df))
    return None


def infer_model_name(name: str, metrics: dict[str, object] | None = None) -> str | None:
    metrics = metrics or {}
    for key in ("model_name", "model", "model_type"):
        if key in metrics and metrics[key]:
            return str(metrics[key])
    lowered = name.lower()
    for model in ("persistence", "cnn_lstm", "cnn_gru", "transformer", "tcn", "cnn"):
        if lowered.endswith(f"_{model}") or f"_{model}_" in lowered:
            return model
    return None


def normalize_prediction_frame(df: pd.DataFrame, source: str = "") -> tuple[pd.DataFrame | None, dict[str, object]]:
    """Normalize prediction columns and convert storm surge values to cm."""

    obs_col = _find_alias(df.columns, OBSERVED_ALIASES)
    pred_col = _find_alias(df.columns, PREDICTED_ALIASES)
    time_col = _find_alias(df.columns, TIME_ALIASES)
    lead_col = _find_alias(df.columns, LEAD_ALIASES)
    persistence_col = _find_alias(df.columns, PERSISTENCE_ALIASES)

    if obs_col is None or pred_col is None:
        warn(f"{source} missing observed/predicted columns; skipped")
        return None, {"warning": "missing observed or predicted"}

    out = pd.DataFrame()
    out["observed_raw"] = _coerce_numeric(df[obs_col])
    out["predicted_raw"] = _coerce_numeric(df[pred_col])
    observed_cm, observed_unit, observed_scale = unit_scale_to_cm(out["observed_raw"].to_numpy(float))
    predicted_cm = _coerce_numeric(df[pred_col]).to_numpy(float) * observed_scale
    out["observed"] = observed_cm
    out["predicted"] = predicted_cm

    if persistence_col is not None:
        out["persistence"] = _coerce_numeric(df[persistence_col]).to_numpy(float) * observed_scale
    if time_col is not None:
        out["datetime"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        out["datetime"] = pd.NaT
    if lead_col is not None:
        out["lead_time"] = pd.to_numeric(df[lead_col], errors="coerce")
    else:
        out["lead_time"] = np.arange(1, len(out) + 1)

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["observed", "predicted"]).reset_index(drop=True)
    if out.empty:
        warn(f"{source} contains no finite observed/predicted pairs; skipped")
        return None, {"warning": "no finite pairs"}
    return out, {
        "source_unit": observed_unit,
        "unit_scale_to_cm": observed_scale,
        "observed_column": str(obs_col),
        "predicted_column": str(pred_col),
        "time_column": str(time_col) if time_col is not None else None,
        "lead_column": str(lead_col) if lead_col is not None else None,
    }


def normalize_metrics(metrics: dict[str, object]) -> dict[str, float | str | int | None]:
    """Return common metric names with RMSE/MAE in cm and RRMSE in percent."""

    aliases = {
        "pearson_r": ("pearson_r", "r", "correlation", "corr"),
        "rmse": ("rmse", "rmse_cm"),
        "mae": ("mae", "mae_cm"),
        "rrmse": ("rrmse", "rrmse_percent"),
    }
    out: dict[str, float | str | int | None] = {}
    lowered = {str(key).lower(): value for key, value in metrics.items()}
    for normalized, keys in aliases.items():
        value = None
        for key in keys:
            if key in lowered:
                value = lowered[key]
                break
        if value is None:
            continue
        try:
            if normalized in {"rmse", "mae"} and key not in {"rmse_cm", "mae_cm"}:
                out[normalized] = metric_to_cm(value)
            elif normalized == "rrmse" and key != "rrmse_percent":
                out[normalized] = rrmse_to_percent(value)
            else:
                out[normalized] = float(value)
        except (TypeError, ValueError):
            continue
    for key in (
        "model_name",
        "model",
        "input_steps",
        "forecast_steps",
        "n_train",
        "n_val",
        "val_size",
        "validation_samples",
        "n_validation",
    ):
        if key in lowered:
            out[key] = lowered[key]
    return out


def compute_metrics_from_predictions(df: pd.DataFrame) -> dict[str, float]:
    obs = df["observed"].to_numpy(float)
    pred = df["predicted"].to_numpy(float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) == 0:
        return {"pearson_r": np.nan, "rmse": np.nan, "mae": np.nan, "rrmse": np.nan, "n": 0}
    error = pred - obs
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    denom = float(np.sqrt(np.mean(obs**2)))
    rrmse = float(rmse / denom * 100.0) if denom > 0 else float("nan")
    if len(obs) >= 2 and np.nanstd(obs) > 0 and np.nanstd(pred) > 0:
        pearson_r = float(np.corrcoef(obs, pred)[0, 1])
    else:
        pearson_r = float("nan")
    return {"pearson_r": pearson_r, "rmse": rmse, "mae": mae, "rrmse": rrmse, "n": int(len(obs))}


def discover_result_dirs(results_root: Path) -> list[Path]:
    """Recursively find directories containing known result files."""

    results_root = Path(results_root)
    if not results_root.exists():
        warn(f"Results root does not exist: {results_root}")
        return []
    dirs: set[Path] = set()
    known_names = set(KNOWN_RESULT_FILES.values())
    for path in results_root.rglob("*"):
        if path.is_file() and path.name in known_names:
            dirs.add(path.parent)
    return sorted(dirs)


def load_experiment(path: Path) -> ExperimentResult:
    path = Path(path)
    result = ExperimentResult(name=path.name, path=path)
    for key, filename in KNOWN_RESULT_FILES.items():
        candidate = path / filename
        if candidate.exists():
            result.files[key] = candidate

    raw_metrics: dict[str, object] = {}
    if "metrics" in result.files:
        raw_metrics.update(safe_read_json(result.files["metrics"]))
    elif path.name.lower().startswith("persistence") and (path / "baseline_metrics.json").exists():
        result.files["metrics"] = path / "baseline_metrics.json"
        raw_metrics.update(safe_read_json(path / "baseline_metrics.json"))
    else:
        result.warnings.append("missing metrics.json")

    result.metrics = normalize_metrics(raw_metrics)
    rolling_df = safe_read_csv(result.files["rolling_forecast"]) if "rolling_forecast" in result.files else None
    result.metadata = {
        "input_steps": infer_input_steps(result.name, raw_metrics),
        "forecast_steps": infer_forecast_steps(result.name, raw_metrics, rolling_df),
        "model_name": infer_model_name(result.name, raw_metrics),
    }
    return result


def scan_results(results_root: Path) -> list[ExperimentResult]:
    """Scan a remote results root and load experiment metadata."""

    experiments = [load_experiment(path) for path in discover_result_dirs(results_root)]
    if not experiments:
        warn(f"No recognized result directories found under {results_root}")
    return experiments


def load_predictions_for_experiment(experiment: ExperimentResult, prefer: str = "val") -> pd.DataFrame | None:
    keys = ("val_predictions", "predictions", "rolling_forecast") if prefer == "val" else ("rolling_forecast", "predictions", "val_predictions")
    for key in keys:
        if key not in experiment.files:
            continue
        raw = safe_read_csv(experiment.files[key])
        if raw is None:
            continue
        normalized, meta = normalize_prediction_frame(raw, source=str(experiment.files[key]))
        if normalized is None:
            experiment.warnings.append(str(meta.get("warning", "prediction normalization failed")))
            continue
        normalized["experiment"] = experiment.name
        normalized["model_name"] = experiment.model_name or "forecast"
        normalized["input_steps"] = experiment.input_steps
        normalized["forecast_steps"] = experiment.forecast_steps
        return normalized
    warn(f"{experiment.name} has no readable prediction CSV")
    return None


def load_prediction_csv(path: Path) -> tuple[pd.DataFrame | None, dict[str, object]]:
    raw = safe_read_csv(Path(path))
    if raw is None:
        return None, {"warning": "cannot read csv"}
    return normalize_prediction_frame(raw, source=str(path))
