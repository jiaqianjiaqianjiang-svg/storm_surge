"""Conservative tide-gauge quality control and sensor-channel selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# GESLA-3 contributor flags: 0=no QC, 1=correct, 2=interpolated,
# 3=doubtful, 4=spike/wrong, and 5=missing. GESLA's separate use flag is
# authoritative for whether a record is recommended for analysis.
BAD_QC = {"3", "4", "5", "9", "bad", "fail", "failed", "invalid"}
BAD_USE = {"0", "false", "f", "no", "n", "reject", "invalid"}


def _normalise_flags(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _flag_counts(series: pd.Series) -> dict[str, int]:
    normalised = _normalise_flags(series).fillna("<missing>")
    return {
        str(flag): int(count)
        for flag, count in normalised.value_counts(dropna=False).sort_index().items()
    }


def _sensor_metrics(group: pd.DataFrame) -> dict[str, float | int | str]:
    valid = group.dropna(subset=["datetime", "water_level"]).sort_values("datetime")
    if valid.empty:
        return {"score": 0.0, "valid_records": 0}
    span_hours = max(1.0, (valid.datetime.iloc[-1] - valid.datetime.iloc[0]).total_seconds() / 3600 + 1)
    completeness = min(1.0, len(valid) / span_hours)
    gaps = valid.datetime.diff().dt.total_seconds().div(3600)
    continuity = float((gaps.dropna() <= 1.5).mean()) if len(valid) > 1 else 1.0
    return {
        "score": round(0.7 * completeness + 0.3 * continuity, 6),
        "valid_records": len(valid),
        "start": valid.datetime.iloc[0].isoformat(),
        "end": valid.datetime.iloc[-1].isoformat(),
        "completeness": round(completeness, 8),
        "hourly_continuity": round(continuity, 8),
        "non_hourly_gaps": int((gaps.dropna() > 1.5).sum()),
        "minimum_m": round(float(valid.water_level.min()), 6),
        "maximum_m": round(float(valid.water_level.max()), 6),
        "median_m": round(float(valid.water_level.median()), 6),
    }


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
    quality_flag_counts = _flag_counts(work.qc_flag)
    use_flag_counts = _flag_counts(work.use_flag)
    flag_pairs = (
        _normalise_flags(work.qc_flag).fillna("<missing>")
        + "|"
        + _normalise_flags(work.use_flag).fillna("<missing>")
    )
    quality_use_flag_pairs = {
        str(pair): int(count)
        for pair, count in flag_pairs.value_counts().sort_index().items()
    }
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
    bad_qc_count = int(qc_bad.sum())
    bad_use_count = int(use_bad.sum())
    bad_both_count = int((qc_bad & use_bad).sum())
    flag_count = int((qc_bad | use_bad).sum())
    work = work.loc[~(qc_bad | use_bad)].copy()
    unreasonable = work.water_level.abs() > unreasonable_limit_m
    unreasonable_count = int(unreasonable.sum())
    work = work.loc[~unreasonable].copy()
    if work.empty:
        raise ValueError("No valid tide-gauge records remain after quality control")

    channel_stats: dict[str, dict[str, float | int | str]] = {}
    for sensor, group in work.groupby("sensor", dropna=False):
        channel_stats[str(sensor)] = _sensor_metrics(group)
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
        "removed_bad_qc_flag_count": bad_qc_count,
        "removed_bad_use_flag_count": bad_use_count,
        "removed_bad_qc_and_use_count": bad_both_count,
        "removed_unreasonable_count": unreasonable_count,
        "final_record_count": len(clean),
        "time_range": [clean.datetime.iloc[0].isoformat(), clean.datetime.iloc[-1].isoformat()],
        "missing_rate": round(missing_rate, 8),
        "selected_sensor": selected_sensor,
        "sensor_statistics": channel_stats,
        "quality_flag_counts": quality_flag_counts,
        "use_flag_counts": use_flag_counts,
        "quality_use_flag_pairs": quality_use_flag_pairs,
        "gesla_flag_policy": {
            "accepted_qc_flags": "0 (not assessed), 1 (correct), 2 (interpolated)",
            "rejected_qc_flags": sorted(BAD_QC),
            "rejected_use_flags": sorted(BAD_USE),
            "rule": "reject when either the contributor QC flag or GESLA use flag is rejected",
        },
    }
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return clean, report
