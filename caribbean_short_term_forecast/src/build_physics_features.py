"""Build cached 2011-2017 physical features without loading the 2018 test period."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

try:
    from .direct_multistep_baselines import direct_origins, hourly_atmosphere_valid
    from .era5_loader import COORD_ALIASES, _find_name
    from .physics_features import (
        build_origin_physics_features,
        derive_hourly_physics,
        normalise_longitudes,
    )
    from .station_registry import get_station
    from .train_station import load_prepared
except ImportError:
    from direct_multistep_baselines import direct_origins, hourly_atmosphere_valid
    from era5_loader import COORD_ALIASES, _find_name
    from physics_features import build_origin_physics_features, derive_hourly_physics, normalise_longitudes
    from station_registry import get_station
    from train_station import load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--era5-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--end-year", type=int, default=2017)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _era5_candidates(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"ERA5 path not found: {path}")
    return sorted(
        file for file in path.iterdir()
        if file.is_file() and not file.name.startswith("._") and file.suffix.lower() in {".nc", ".grib", ".grb", ".grib2", ".grb2"}
    )


def audit_era5_source_variables(era5_path: str | Path) -> dict[str, dict[str, str]]:
    aliases = {
        "U10": {"u10", "10u", "u_component_of_wind_10m"},
        "V10": {"v10", "10v", "v_component_of_wind_10m"},
        "MSL": {"msl", "slp", "mean_sea_level_pressure"},
    }
    found: dict[str, dict[str, str]] = {}
    for path in _era5_candidates(Path(era5_path)):
        dataset = xr.open_dataset(path)
        try:
            for canonical, candidates in aliases.items():
                if canonical in found:
                    continue
                match = next((name for name in dataset.data_vars if name.lower() in candidates), None)
                if match is not None:
                    found[canonical] = {
                        "source_file": str(path), "source_variable": match,
                        "units": str(dataset[match].attrs.get("units", "missing")),
                        "long_name": str(dataset[match].attrs.get("long_name", "missing")),
                    }
        finally:
            dataset.close()
        if len(found) == 3:
            break
    if set(found) != set(aliases):
        raise ValueError(f"Could not audit all ERA5 variables; found {sorted(found)}")
    return found


def audit_tide_reconstruction(dataset_path: Path, end_year: int) -> dict[str, Any]:
    path = dataset_path.parent / "tide" / "tide_reconstruction.csv"
    if not path.is_file():
        return {"available": False, "expected_path": str(path)}
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    records = 0
    finite_records = 0
    # The file is chronological. Stop before consuming any row from a locked year.
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = pd.Timestamp(row["datetime"])
            if timestamp.year > end_year:
                break
            first = timestamp if first is None else first
            last = timestamp
            records += 1
            try:
                finite_records += int(np.isfinite(float(row["predicted_tide_m"])))
            except (TypeError, ValueError):
                pass
    return {
        "available": True, "path": str(path), "unit": "m",
        "records_through_end_year": records, "finite_records_through_end_year": finite_records,
        "audited_end_year": end_year,
        "time_range_utc": [first.isoformat() if first is not None else None, last.isoformat() if last is not None else None],
    }


def load_prepared_grid_coordinates(
    era5_path: str | Path,
    station_latitude: float,
    station_longitude: float,
    grid_size: int = 40,
    region_size_degrees: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover the exact crop/interpolation coordinates used by era5_loader."""
    candidates = _era5_candidates(Path(era5_path))
    if not candidates:
        raise FileNotFoundError(f"No usable ERA5 file in {era5_path}")
    source = candidates[0]
    dataset = xr.open_dataset(source)
    try:
        available = list(dataset.coords) + list(dataset.dims)
        lat_name = _find_name(COORD_ALIASES["latitude"], available, "latitude coordinate")
        lon_name = _find_name(COORD_ALIASES["longitude"], available, "longitude coordinate")
        latitude = np.asarray(dataset[lat_name].values, dtype=np.float64)
        longitude = normalise_longitudes(dataset[lon_name].values)
    finally:
        dataset.close()
    latitude = np.sort(latitude)
    longitude = np.sort(longitude)
    half = float(region_size_degrees) / 2
    station_lon = float(normalise_longitudes(station_longitude))
    latitude = latitude[(latitude >= station_latitude - half) & (latitude <= station_latitude + half)]
    longitude = longitude[(longitude >= station_lon - half) & (longitude <= station_lon + half)]
    raw_shape = [int(len(latitude)), int(len(longitude))]
    if min(raw_shape) == 0:
        raise ValueError("ERA5 crop is empty; station coordinates or raw coverage are wrong")
    interpolated = raw_shape != [grid_size, grid_size]
    if interpolated:
        latitude = np.linspace(float(latitude.min()), float(latitude.max()), grid_size)
        longitude = np.linspace(float(longitude.min()), float(longitude.max()), grid_size)
    metadata = {
        "source_file": str(source),
        "source_coordinate_names": {"latitude": lat_name, "longitude": lon_name},
        "raw_cropped_shape": raw_shape,
        "interpolated_to_grid_size": bool(interpolated),
        "latitude_ascending": bool(np.all(np.diff(latitude) > 0)),
        "longitude_ascending": bool(np.all(np.diff(longitude) > 0)),
        "latitude_range": [float(latitude.min()), float(latitude.max())],
        "longitude_range": [float(longitude.min()), float(longitude.max())],
    }
    return latitude, longitude, metadata


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=MODULE_ROOT.parent, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_cache(
    station_id: str,
    dataset_path: Path,
    era5_path: Path,
    output_dir: Path,
    end_year: int = 2017,
    smoke_test: bool = False,
) -> dict[str, Any]:
    if end_year >= 2018:
        raise ValueError("Physics model development must not load or evaluate 2018")
    station = get_station(station_id).require("latitude", "longitude")
    physics_config = station.physics or {}
    leads = tuple(int(value) for value in physics_config.get("forecast_leads", [1, 3, 6, 12, 24]))
    past_windows = tuple(int(value) for value in physics_config.get("past_accumulation_hours", [3, 6, 12, 24]))
    atmosphere, surge, times = load_prepared(dataset_path, 2011, end_year)
    if times.max().year >= 2018:
        raise AssertionError("2018 entered physics feature development")
    if smoke_test:
        stop = min(len(times), 24 * 120)
        atmosphere, surge, times = atmosphere[:stop], surge[:stop], times[:stop]
    if atmosphere.shape[1:] != (3, 40, 40):
        raise ValueError(f"Expected prepared atmosphere (*,3,40,40), got {atmosphere.shape}")
    latitude, longitude, coordinate_metadata = load_prepared_grid_coordinates(
        era5_path, float(station.latitude), float(station.longitude)
    )
    source_variable_audit = audit_era5_source_variables(era5_path)
    tide_audit = audit_tide_reconstruction(dataset_path, end_year)
    if (len(latitude), len(longitude)) != atmosphere.shape[-2:]:
        raise ValueError("Recovered coordinates do not match prepared ERA5 grid")
    train_mask = (times.year >= 2011) & (times.year <= 2016)
    hourly = derive_hourly_physics(
        atmosphere, latitude, longitude, float(station.latitude), float(station.longitude),
        train_mask, physics_config,
    )
    atmosphere_valid = hourly_atmosphere_valid(atmosphere)
    years = set(int(value) for value in np.unique(times.year))
    origins = direct_origins(times, atmosphere_valid, surge, years, 24, max(leads))
    if not len(origins):
        raise ValueError("No complete direct origins remain for physical features")

    arrays: dict[str, np.ndarray] = {
        "hourly_values": hourly.values,
        "time": times.to_numpy(dtype="datetime64[ns]"),
        "origin_indices": origins,
        "origin_time": times[origins].to_numpy(dtype="datetime64[ns]"),
        "latitude": latitude.astype(np.float64),
        "longitude": longitude.astype(np.float64),
    }
    names_by_lead: dict[str, list[str]] = {}
    for lead in leads:
        matrix, names = build_origin_physics_features(
            hourly.values, hourly.names, times, origins, lead, past_windows
        )
        arrays[f"lead_{lead:02d}h"] = matrix
        names_by_lead[str(lead)] = names

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "physics_features.npz", **arrays)
    (output_dir / "physics_feature_names.json").write_text(
        json.dumps({"hourly": hourly.names, "by_lead": names_by_lead}, indent=2), encoding="utf-8"
    )
    metadata: dict[str, Any] = {
        "station_id": station_id,
        "station_coordinate": [float(station.latitude), float(station.longitude)],
        "prepared_dataset": str(dataset_path),
        "era5_path": str(era5_path),
        "array_shape": list(atmosphere.shape),
        "variable_order": ["U10", "V10", "MSL"],
        "variable_units": ["m/s", "m/s", "Pa"],
        "time_range": [times.min().isoformat(), times.max().isoformat()],
        "train_years_for_climatology": [2011, 2016],
        "development_end_year": end_year,
        "2018_loaded": False,
        "origin_count": int(len(origins)),
        "origin_counts_by_year": {str(year): int(np.sum(times[origins].year == year)) for year in sorted(years)},
        "forecast_leads": list(leads),
        "past_accumulation_hours": list(past_windows),
        "formulas": {
            "inverse_barometer": "-(p_local-p_ref)/(rho_water*g) [m]",
            "wind_stress": "rho_air*Cd*speed*(u10,v10) [Pa]",
            "future_accumulation": "sum(t+1..t+h)*3600 [Pa s]",
        },
        "coordinates": coordinate_metadata,
        "raw_era5_variable_audit": source_variable_audit,
        "tide_reconstruction_audit": tide_audit,
        "hourly_physics": hourly.metadata,
        "hourly_feature_units": dict(zip(hourly.names, hourly.units, strict=True)),
        "git_commit_at_build": git_commit(),
        "smoke_test": smoke_test,
    }
    (output_dir / "physics_feature_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit = {
        "status": "passed",
        "2018_loaded": False,
        "atmosphere_shape": list(atmosphere.shape),
        "atmosphere_dtype": str(atmosphere.dtype),
        "surge_unit": "m",
        "msl_unit": "Pa",
        "coordinate_source": coordinate_metadata["source_file"],
        "coordinate_shape": [len(latitude), len(longitude)],
        "raw_era5_variable_audit": source_variable_audit,
        "tide_reconstruction_audit": tide_audit,
        "origin_count": int(len(origins)),
        "onshore_features_available": hourly.metadata["onshore_features_available"],
        "manual_action": (
            None if hourly.metadata["onshore_features_available"]
            else "Verify Prickly Bay offshore-to-bay bearing before enabling directional stress."
        ),
    }
    (output_dir / "DATA_AUDIT_REPORT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    station = get_station(args.station)
    dataset = args.dataset_path or MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    era5 = args.era5_path or Path(station.era5_path or "")
    output = args.output_dir or MODULE_ROOT / "outputs" / "experiments" / args.station / "physics_fusion_v1"
    metadata = build_cache(args.station, dataset, era5, output, args.end_year, args.smoke_test)
    print(json.dumps({key: metadata[key] for key in ("time_range", "origin_count", "2018_loaded")}, indent=2))
    print(f"Physics cache: {output.resolve()}")


if __name__ == "__main__":
    main()
