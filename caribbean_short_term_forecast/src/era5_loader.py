"""ERA5 NetCDF/GRIB loading, coordinate normalisation, cropping and 40x40 resampling."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


LOGGER = logging.getLogger(__name__)
VARIABLE_ALIASES = {
    "U10": ("u10", "10u", "u_component_of_wind_10m"),
    "V10": ("v10", "10v", "v_component_of_wind_10m"),
    "MSL": ("msl", "slp", "mean_sea_level_pressure"),
}
COORD_ALIASES = {
    "time": ("time", "valid_time", "datetime"),
    "latitude": ("latitude", "lat", "y"),
    "longitude": ("longitude", "lon", "x"),
}


def _find_name(candidates: Sequence[str], available: Sequence[str], kind: str) -> str:
    lower = {name.lower(): name for name in available}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(f"Could not identify {kind}. Available names: {list(available)}")


def _open(path: Path) -> xr.Dataset:
    suffix = path.suffix.lower()
    if suffix in {".grib", ".grb", ".grib2", ".grb2"}:
        try:
            return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        except (ImportError, ValueError) as exc:
            raise RuntimeError(f"Could not open GRIB {path}; install cfgrib and eccodes") from exc
    return xr.open_dataset(path)


def load_era5_file(
    path: str | Path,
    latitude: float,
    longitude: float,
    grid_size: int = 40,
    region_size_degrees: float = 10.0,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> xr.DataArray:
    """Load one manageable ERA5 period and return (time, variable, lat, lon) float32 data."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ERA5 file not found: {path}")
    if latitude is None or longitude is None:
        raise ValueError("Verified station latitude and longitude are required to crop ERA5")
    dataset = _open(path)
    all_names = list(dataset.coords) + list(dataset.dims)
    time_name = _find_name(COORD_ALIASES["time"], all_names, "time coordinate")
    lat_name = _find_name(COORD_ALIASES["latitude"], all_names, "latitude coordinate")
    lon_name = _find_name(COORD_ALIASES["longitude"], all_names, "longitude coordinate")
    rename = {time_name: "time", lat_name: "latitude", lon_name: "longitude"}
    dataset = dataset.rename({old: new for old, new in rename.items() if old != new})
    dataset = dataset.assign_coords(
        longitude=((dataset.longitude + 180) % 360) - 180
    ).sortby("longitude").sortby("latitude")
    station_lon = ((float(longitude) + 180) % 360) - 180
    half = float(region_size_degrees) / 2
    subset = dataset.sel(
        latitude=slice(float(latitude) - half, float(latitude) + half),
        longitude=slice(station_lon - half, station_lon + half),
    )
    if start_time is not None or end_time is not None:
        subset = subset.sel(time=slice(start_time, end_time))
    raw_shape = (subset.sizes.get("latitude", 0), subset.sizes.get("longitude", 0))
    if min(raw_shape) == 0:
        dataset.close()
        raise ValueError(
            f"ERA5 crop around ({latitude}, {longitude}) is empty; verify coordinates and file coverage"
        )

    arrays = []
    for canonical, aliases in VARIABLE_ALIASES.items():
        name = _find_name(aliases, list(subset.data_vars), canonical)
        array = subset[name].squeeze(drop=True).transpose("time", "latitude", "longitude")
        arrays.append(array.expand_dims(variable=[canonical]))
    combined = xr.concat(arrays, dim="variable").transpose("time", "variable", "latitude", "longitude")
    interpolated = raw_shape != (grid_size, grid_size)
    if interpolated:
        target_lat = np.linspace(float(combined.latitude.min()), float(combined.latitude.max()), grid_size)
        target_lon = np.linspace(float(combined.longitude.min()), float(combined.longitude.max()), grid_size)
        combined = combined.interp(latitude=target_lat, longitude=target_lon)
    combined = combined.astype("float32").load()
    dataset.close()
    times = pd.DatetimeIndex(pd.to_datetime(combined.time.values))
    expected = pd.date_range(times.min(), times.max(), freq="1h") if len(times) else pd.DatetimeIndex([])
    missing_count = len(expected.difference(times))
    LOGGER.info(
        "ERA5 %s raw_region_shape=%s interpolated=%s final_shape=%s time_range=%s..%s missing_times=%d",
        path.name, raw_shape, interpolated, tuple(combined.shape),
        times.min() if len(times) else None, times.max() if len(times) else None, missing_count,
    )
    return combined


def iter_era5_periods(
    paths: Sequence[str | Path],
    **load_kwargs: object,
) -> Iterator[xr.DataArray]:
    """Yield one file/period at a time so multi-year data need not reside in memory."""
    for path in paths:
        yield load_era5_file(path, **load_kwargs)


def cache_period(array: xr.DataArray, output_path: str | Path) -> Path:
    """Store a processed year/month cache; callers control chunk size via the input period."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array.to_netcdf(destination)
    return destination
