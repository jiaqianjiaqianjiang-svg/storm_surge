"""UTide harmonic analysis and hourly storm-surge label generation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONSTITUENTS = (
    "M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "M4", "MS4", "MN4",
    "2N2", "MU2", "NU2", "L2", "T2", "J1", "OO1", "M6", "M8",
)


def separate_tide(
    cleaned: pd.DataFrame,
    latitude: float,
    output_dir: str | Path | None = None,
    minimum_days: int = 30,
    calibration_end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompute tide, optionally fitting constituents only through a cutoff.

    ``calibration_end`` prevents validation/test observations from influencing
    the harmonic coefficients while still reconstructing tide for the full
    requested record.
    """
    if latitude is None or not -90 <= float(latitude) <= 90:
        raise ValueError("A verified station latitude in [-90, 90] is required for UTide")
    required = {"datetime", "water_level"}
    if not required <= set(cleaned.columns):
        raise ValueError(f"Expected columns {sorted(required)}")
    frame = cleaned.copy()
    frame["datetime"] = pd.to_datetime(frame.datetime, errors="coerce", utc=True)
    frame["water_level"] = pd.to_numeric(frame.water_level, errors="coerce")
    frame = frame.dropna(subset=["datetime", "water_level"]).sort_values("datetime")
    duplicates = int(frame.datetime.duplicated().sum())
    if duplicates:
        raise ValueError(f"UTide input contains {duplicates} duplicate timestamps; run quality control first")
    intervals = frame.datetime.diff().dt.total_seconds().div(3600).dropna()
    irregular = int((~np.isclose(intervals, 1.0, atol=1e-6)).sum())
    if irregular:
        warnings.warn(f"UTide input contains {irregular} non-hourly intervals", RuntimeWarning)
    if calibration_end is None:
        calibration_mask = np.ones(len(frame), dtype=bool)
        cutoff = None
    else:
        cutoff = pd.Timestamp(calibration_end)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        calibration_mask = (frame.datetime <= cutoff).to_numpy()
    calibration = frame.loc[calibration_mask]
    if calibration.empty:
        raise ValueError("No observations fall inside the UTide calibration period")
    duration_days = (
        calibration.datetime.iloc[-1] - calibration.datetime.iloc[0]
    ).total_seconds() / 86400
    if len(calibration) < minimum_days * 20 or duration_days < minimum_days - 1:
        raise ValueError(
            f"Only {len(calibration)} calibration observations over {duration_days:.1f} days; "
            f"at least {minimum_days} days are required. No pseudo-result was generated."
        )

    try:
        from utide import reconstruct, solve
    except ImportError as exc:
        raise ImportError("UTide is required: install dependencies from requirements.txt") from exc
    naive_utc = frame.datetime.dt.tz_convert("UTC").dt.tz_localize(None)
    epoch = naive_utc.iloc[0].to_pydatetime()
    observed = frame.water_level.to_numpy(dtype=float)
    fit_days = (
        naive_utc.loc[calibration_mask] - naive_utc.iloc[0]
    ).dt.total_seconds().to_numpy() / 86400.0
    fit_observed = observed[calibration_mask]
    coef = solve(
        fit_days, fit_observed, lat=float(latitude), epoch=epoch, constit=CONSTITUENTS,
        method="ols", trend=False, conf_int="none", verbose=False,
    )
    hourly_time = pd.date_range(frame.datetime.iloc[0], frame.datetime.iloc[-1], freq="1h")
    hourly_naive = hourly_time.tz_convert("UTC").tz_localize(None)
    hourly_days = (hourly_naive - naive_utc.iloc[0]).total_seconds().to_numpy() / 86400.0
    predicted = reconstruct(
        hourly_days, coef, epoch=epoch, constit=coef.name, verbose=False,
    ).h
    observed_series = frame.set_index("datetime").water_level.reindex(hourly_time)
    result = pd.DataFrame(
        {
            "datetime": hourly_time,
            "observed_water_level_m": observed_series.to_numpy(dtype=float),
            "predicted_tide_m": np.asarray(predicted, dtype=float),
        }
    )
    result["storm_surge_m"] = result.observed_water_level_m - result.predicted_tide_m
    result["qc_valid"] = result.observed_water_level_m.notna()
    diagnostic = {
        "latitude": float(latitude),
        "record_count": len(result),
        "observed_record_count": int(result.qc_valid.sum()),
        "missing_hourly_observation_count": int((~result.qc_valid).sum()),
        "time_range": [str(result.datetime.iloc[0]), str(result.datetime.iloc[-1])],
        "calibration_end": cutoff.isoformat() if cutoff is not None else None,
        "calibration_record_count": int(calibration_mask.sum()),
        "irregular_interval_count": irregular,
        "constituents": [str(name) for name in coef.name],
        "used_supplied_tide_estimate": False,
        "auxiliary_tide_estimate_compared": bool(
            "tide_estimate" in frame and frame.tide_estimate.notna().any()
        ),
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / "cleaned_water_level.csv", index=False)
        result[["datetime", "predicted_tide_m"]].to_csv(destination / "tide_reconstruction.csv", index=False)
        result.to_csv(destination / "hourly_storm_surge.csv", index=False)
        (destination / "utide_diagnostics.json").write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result, diagnostic
