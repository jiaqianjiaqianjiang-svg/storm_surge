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
    """Load one file containing all three variables (backward-compatible wrapper)."""
    return load_era5_files(
        [path],
        latitude=latitude,
        longitude=longitude,
        grid_size=grid_size,
        region_size_degrees=region_size_degrees,
        start_time=start_time,
        end_time=end_time,
    )


def load_era5_files(
    paths: Sequence[str | Path],
    latitude: float,
    longitude: float,
    grid_size: int = 40,
    region_size_degrees: float = 10.0,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> xr.DataArray:
    """Merge U10, V10 and MSL files for one period.

    The three variables may be stored together or in separate files. Coordinate
    equality is checked before they are combined, preventing silent time/grid
    misalignment.
    """
    period_paths = [Path(path) for path in paths]
    if not period_paths:
        raise ValueError("At least one ERA5 file is required")
    for path in period_paths:
        if not path.is_file():
            raise FileNotFoundError(f"ERA5 file not found: {path}")
    if latitude is None or longitude is None:
        raise ValueError("Verified station latitude and longitude are required to crop ERA5")
    station_lon = ((float(longitude) + 180) % 360) - 180
    half = float(region_size_degrees) / 2
    datasets: list[xr.Dataset] = []
    variables: dict[str, xr.DataArray] = {}
    raw_shapes: list[tuple[int, int]] = []
    try:
        for path in period_paths:
            dataset = _open(path)
            datasets.append(dataset)
            all_names = list(dataset.coords) + list(dataset.dims)
            time_name = _find_name(COORD_ALIASES["time"], all_names, "time coordinate")
            lat_name = _find_name(COORD_ALIASES["latitude"], all_names, "latitude coordinate")
            lon_name = _find_name(COORD_ALIASES["longitude"], all_names, "longitude coordinate")
            rename = {time_name: "time", lat_name: "latitude", lon_name: "longitude"}
            normalised = dataset.rename({old: new for old, new in rename.items() if old != new})
            normalised = normalised.assign_coords(
                longitude=((normalised.longitude + 180) % 360) - 180
            ).sortby("longitude").sortby("latitude")
            subset = normalised.sel(
                latitude=slice(float(latitude) - half, float(latitude) + half),
                longitude=slice(station_lon - half, station_lon + half),
            )
            if start_time is not None or end_time is not None:
                subset = subset.sel(time=slice(start_time, end_time))
            raw_shape = (subset.sizes.get("latitude", 0), subset.sizes.get("longitude", 0))
            if min(raw_shape) == 0:
                raise ValueError(
                    f"ERA5 crop around ({latitude}, {longitude}) is empty in {path}; "
                    "verify coordinates and file coverage"
                )
            raw_shapes.append(raw_shape)
            for canonical, aliases in VARIABLE_ALIASES.items():
                matches = [
                    name for name in subset.data_vars
                    if name.lower() in {alias.lower() for alias in aliases}
                ]
                if not matches:
                    continue
                if canonical in variables:
                    raise ValueError(
                        f"Duplicate ERA5 variable {canonical} found while merging {period_paths}"
                    )
                variables[canonical] = (
                    subset[matches[0]].squeeze(drop=True).transpose("time", "latitude", "longitude")
                )
        missing = [name for name in VARIABLE_ALIASES if name not in variables]
        if missing:
            raise ValueError(
                f"ERA5 period is missing variables {missing}; files: "
                f"{[path.name for path in period_paths]}"
            )
        ordered = [variables[name] for name in VARIABLE_ALIASES]
        try:
            aligned = xr.align(*ordered, join="exact", copy=False)
        except ValueError as exc:
            raise ValueError(
                f"ERA5 variable files do not share identical time/latitude/longitude coordinates: "
                f"{[path.name for path in period_paths]}"
            ) from exc
        arrays = [
            array.expand_dims(variable=[canonical])
            for canonical, array in zip(VARIABLE_ALIASES, aligned, strict=True)
        ]
        combined = xr.concat(arrays, dim="variable").transpose(
            "time", "variable", "latitude", "longitude"
        )
        raw_shape = raw_shapes[0]
        interpolated = raw_shape != (grid_size, grid_size)
        if interpolated:
            target_lat = np.linspace(
                float(combined.latitude.min()), float(combined.latitude.max()), grid_size
            )
            target_lon = np.linspace(
                float(combined.longitude.min()), float(combined.longitude.max()), grid_size
            )
            combined = combined.interp(latitude=target_lat, longitude=target_lon)
        combined = combined.astype("float32").load()
    finally:
        for dataset in datasets:
            dataset.close()
    times = pd.DatetimeIndex(pd.to_datetime(combined.time.values))
    expected = pd.date_range(times.min(), times.max(), freq="1h") if len(times) else pd.DatetimeIndex([])
    missing_count = len(expected.difference(times))
    LOGGER.info(
        "ERA5 %s raw_region_shape=%s interpolated=%s final_shape=%s time_range=%s..%s missing_times=%d",
        "+".join(path.name for path in period_paths), raw_shape, interpolated, tuple(combined.shape),
        times.min() if len(times) else None, times.max() if len(times) else None, missing_count,
    )
    return combined


def iter_era5_periods(
    paths: Sequence[str | Path] | Sequence[Sequence[str | Path]],
    **load_kwargs: object,
) -> Iterator[xr.DataArray]:
    """Yield one period at a time so multi-year data need not reside in memory."""
    for period in paths:
        if isinstance(period, (str, Path)):
            yield load_era5_file(period, **load_kwargs)
        else:
            yield load_era5_files(period, **load_kwargs)


def cache_period(array: xr.DataArray, output_path: str | Path) -> Path:
    """Store a processed year/month cache; callers control chunk size via the input period."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array.to_netcdf(destination)
    return destination
