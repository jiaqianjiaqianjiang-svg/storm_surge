"""Leakage-safe exact-hour alignment of ERA5 fields and storm-surge labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def to_utc_index(values: object, timezone: str | None, label: str) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if parsed.isna().any():
        raise ValueError(f"{label} contains {int(parsed.isna().sum())} invalid timestamps")
    if parsed.tz is None:
        if not timezone:
            raise ValueError(
                f"{label} timestamps are timezone-naive and timezone is not configured; "
                "set the verified station timezone in configs/stations.yaml"
            )
        parsed = parsed.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return parsed.tz_convert("UTC")


def _contiguous_segments(index: pd.DatetimeIndex) -> int:
    if index.empty:
        return 0
    gaps = index.to_series().diff().gt(pd.Timedelta(hours=1))
    return int(gaps.sum() + 1)


def align_hourly(
    era5: xr.DataArray,
    surge: pd.DataFrame | pd.Series,
    surge_timezone: str | None = "UTC",
) -> tuple[xr.DataArray, pd.Series, dict[str, Any]]:
    if "time" not in era5.dims:
        raise ValueError("ERA5 array must have a 'time' dimension")
    era_index = to_utc_index(era5.time.values, "UTC", "ERA5")
    if isinstance(surge, pd.DataFrame):
        if "datetime" not in surge or "storm_surge_m" not in surge:
            raise ValueError("Surge DataFrame requires datetime and storm_surge_m columns")
        surge_index = to_utc_index(surge.datetime, surge_timezone, "storm surge")
        surge_values = pd.to_numeric(surge.storm_surge_m, errors="coerce").to_numpy()
    else:
        surge_index = to_utc_index(surge.index, surge_timezone, "storm surge")
        surge_values = pd.to_numeric(surge, errors="coerce").to_numpy()
    era_series = pd.Series(np.arange(len(era_index)), index=era_index)
    era_series = era_series[~era_series.index.duplicated(keep="first")].sort_index()
    surge_series = pd.Series(surge_values, index=surge_index, name="storm_surge_m")
    surge_series = surge_series[~surge_series.index.duplicated(keep="first")].sort_index()
    common = era_series.index.intersection(surge_series.dropna().index).sort_values()
    era_positions = era_series.loc[common].to_numpy(dtype=int)
    aligned_era = era5.isel(time=xr.DataArray(era_positions, dims="time")).assign_coords(
        time=common.tz_localize(None).to_numpy()
    )
    aligned_surge = surge_series.loc[common]
    report = {
        "era5_time_range": [era_index.min().isoformat(), era_index.max().isoformat()] if len(era_index) else None,
        "surge_time_range": [surge_index.min().isoformat(), surge_index.max().isoformat()] if len(surge_index) else None,
        "common_time_range": [common.min().isoformat(), common.max().isoformat()] if len(common) else None,
        "matched_times": len(common),
        "missing_era5_times": len(surge_series.dropna().index.difference(era_series.index)),
        "missing_surge_times": len(era_series.index.difference(surge_series.dropna().index)),
        "continuous_segments": _contiguous_segments(common),
    }
    return aligned_era, aligned_surge, report
