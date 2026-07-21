"""Conservative tide-gauge quality control and sensor-channel selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BAD_QC = {"1", "2", "3", "4", "9", "bad", "fail", "failed", "invalid"}
BAD_USE = {"0", "false", "f", "no", "n", "reject", "invalid"}


def _normalise_flags(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _sensor_score(group: pd.DataFrame) -> tuple[float, int]:
    valid = group.dropna(subset=["datetime", "water_level"]).sort_values("datetime")
    if valid.empty:
        return (0.0, 0)
    span_hours = max(1.0, (valid.datetime.iloc[-1] - valid.datetime.iloc[0]).total_seconds() / 3600 + 1)
    completeness = min(1.0, len(valid) / span_hours)
    gaps = valid.datetime.diff().dt.total_seconds().div(3600)
    continuity = float((gaps.dropna() <= 1.5).mean()) if len(valid) > 1 else 1.0
    return (0.7 * completeness + 0.3 * continuity, len(valid))


def quality_control(
    frame: pd.DataFrame,
    report_path: str | Path | None = None,
    unreasonable_limit_m: float = 20.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"datetime", "water_level", "qc_flag", "use_flag", "sensor"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"QC input is missing columns: {sorted(missing_columns)}")
    work = frame.copy()
    raw_count = len(work)
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce", utc=True)
    work["water_level"] = pd.to_numeric(work["water_level"], errors="coerce")
    invalid_datetime = int(work.datetime.isna().sum())
    work = work.dropna(subset=["datetime"])
    work = work.sort_values("datetime", kind="stable")
    duplicate_count = int(work.duplicated(["datetime", "sensor"]).sum())
    work = work.drop_duplicates(["datetime", "sensor"], keep="first")

    missing_value_count = int(work.water_level.isna().sum())
    work = work.dropna(subset=["water_level"])
    qc_bad = _normalise_flags(work.qc_flag).isin(BAD_QC)
    use_bad = _normalise_flags(work.use_flag).isin(BAD_USE)
    flag_count = int((qc_bad | use_bad).sum())
    work = work.loc[~(qc_bad | use_bad)].copy()
    unreasonable = work.water_level.abs() > unreasonable_limit_m
    unreasonable_count = int(unreasonable.sum())
    work = work.loc[~unreasonable].copy()
    if work.empty:
        raise ValueError("No valid tide-gauge records remain after quality control")

    channel_stats: dict[str, dict[str, float | int]] = {}
    for sensor, group in work.groupby("sensor", dropna=False):
        score, count = _sensor_score(group)
        channel_stats[str(sensor)] = {"score": round(score, 6), "valid_records": count}
    selected_sensor = max(channel_stats, key=lambda name: (channel_stats[name]["score"], channel_stats[name]["valid_records"]))
    clean = work.loc[work.sensor.astype(str) == selected_sensor].sort_values("datetime").reset_index(drop=True)
    expected = max(1, int((clean.datetime.iloc[-1] - clean.datetime.iloc[0]).total_seconds() // 3600) + 1)
    missing_rate = max(0.0, 1.0 - len(clean) / expected)
    report: dict[str, Any] = {
        "raw_record_count": raw_count,
        "invalid_datetime_count": invalid_datetime,
        "removed_duplicate_count": duplicate_count,
        "removed_missing_count": missing_value_count,
        "removed_quality_flag_count": flag_count,
        "removed_unreasonable_count": unreasonable_count,
        "final_record_count": len(clean),
        "time_range": [clean.datetime.iloc[0].isoformat(), clean.datetime.iloc[-1].isoformat()],
        "missing_rate": round(missing_rate, 8),
        "selected_sensor": selected_sensor,
        "sensor_statistics": channel_stats,
    }
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return clean, report
