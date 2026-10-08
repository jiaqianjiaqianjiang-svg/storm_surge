"""Prepare leakage-safe, memory-mapped ERA5 and surge arrays by station."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .era5_loader import inspect_era5_files, load_era5_files
    from .station_config import (
        STATIONS,
        get_station_config,
        validate_output_location,
    )
    from .tide_gauge_loader import load_tide_gauge
    from .tide_processing import separate_tide
    from .tide_quality_control import quality_control
except ImportError:
    from era5_loader import inspect_era5_files, load_era5_files
    from station_config import STATIONS, get_station_config, validate_output_location
    from tide_gauge_loader import load_tide_gauge
    from tide_processing import separate_tide
    from tide_quality_control import quality_control


MODULE_ROOT = Path(__file__).resolve().parents[2]
VARIABLE_HINTS = {
    "U10": ("10u", "u10", "var165"),
    "V10": ("v10", "10v", "var166"),
    "MSL": ("slp", "msl", "var151"),
}


def resolve_era5_files(
    root: str | Path,
    expected_filenames: dict[str, str] | None = None,
) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"ERA5 directory is not available: {root}")
    if expected_filenames:
        expected = [root / expected_filenames[name] for name in VARIABLE_HINTS]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Configured ERA5 files are missing; no substitute file was selected: "
                f"{missing}"
            )
        return expected
    candidates = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".nc", ".nc4", ".cdf"}
    )
    selected: list[Path] = []
    for variable, hints in VARIABLE_HINTS.items():
        matches = [
            path for path in candidates
            if any(hint in path.name.lower() for hint in hints)
        ]
        if not matches:
            raise FileNotFoundError(
                f"Missing {variable} NetCDF under {root}; expected a filename containing {hints}"
            )
        selected.append(matches[0])
    if len(set(selected)) != 3:
        raise ValueError(f"ERA5 variable detection selected duplicate files: {selected}")
    return selected


def _assert_output_is_separate(
    output: Path, era5_dir: Path, gesla_file: Path,
) -> None:
    destination = output.resolve(strict=False)
    for source in (era5_dir.resolve(strict=False), gesla_file.parent.resolve(strict=False)):
        try:
            destination.relative_to(source)
        except ValueError:
            continue
        raise ValueError(
            f"Output directory must not be inside a raw-data directory: {destination}"
        )


def prepare_station(
    station_id: str = "xiamen",
    era5_dir: str | Path | None = None,
    gesla_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    calibration_end_year: int | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    grid_size: int | None = None,
    region_size_degrees: float | None = None,
) -> Path:
    station = get_station_config(station_id)
    era5_dir = Path(era5_dir or station.era5_dir)
    gesla_file = Path(gesla_file or station.gesla_file)
    output = Path(output_dir or station.processed_root)
    start_year = station.data_start_year if start_year is None else start_year
    end_year = station.data_end_year if end_year is None else end_year
    calibration_end_year = (
        station.train_end_year
        if calibration_end_year is None else calibration_end_year
    )
    latitude = station.latitude if latitude is None else latitude
    longitude = station.longitude if longitude is None else longitude
    grid_size = station.grid_size if grid_size is None else grid_size
    region_size_degrees = (
        station.region_size_degrees
        if region_size_degrees is None else region_size_degrees
    )
    if not start_year <= calibration_end_year < end_year:
        raise ValueError("Expected start_year <= calibration_end_year < end_year")
    _assert_output_is_separate(output, era5_dir, gesla_file)
    existing_metadata = output / "aligned_dataset" / "dataset_metadata.json"
    if existing_metadata.is_file():
        existing = json.loads(existing_metadata.read_text(encoding="utf-8"))
        if existing.get("station_id") not in (None, station.station_id):
            raise ValueError(
                f"Refusing to overwrite {existing.get('station_id')} data with "
                f"station {station.station_id}: {output}"
            )
    output.mkdir(parents=True, exist_ok=True)
    era5_files = resolve_era5_files(era5_dir, station.era5_filenames)
    era5_source_audit = inspect_era5_files(era5_files)
    if era5_source_audit["status"] == "failed":
        raise ValueError(
            "ERA5 source audit failed: "
            + "; ".join(era5_source_audit["errors"])
        )

    raw = load_tide_gauge(
        gesla_file,
        station_id=station.station_id,
        source="GESLA 3",
        water_level_unit="m",
    )
    raw_valid_times = pd.to_datetime(raw.datetime, errors="coerce", utc=True).dropna()
    cleaned, qc_report = quality_control(raw, output / "tide_qc_report.json")
    start = pd.Timestamp(start_year, 1, 1, tz="UTC")
    end = pd.Timestamp(end_year, 12, 31, 23, tz="UTC")
    cleaned = cleaned.loc[
        (cleaned.datetime >= start) & (cleaned.datetime <= end)
    ].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError(
            f"No valid {station.name} observations remain in {start_year}-{end_year}"
        )
    surge_frame, utide_report = separate_tide(
        cleaned,
        latitude,
        output / "tide",
        calibration_end=pd.Timestamp(calibration_end_year, 12, 31, 23, tz="UTC"),
    )
    surge_times = pd.DatetimeIndex(pd.to_datetime(surge_frame.datetime, utc=True))
    surge_times = surge_times.tz_convert("UTC").tz_localize(None)

    dataset_dir = output / "aligned_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    count = len(surge_times)
    atmosphere = np.lib.format.open_memmap(
        dataset_dir / "atmosphere.npy",
        mode="w+",
        dtype="float32",
        shape=(count, 3, grid_size, grid_size),
    )
    atmosphere[:] = np.nan
    surge = np.lib.format.open_memmap(
        dataset_dir / "surge.npy", mode="w+", dtype="float32", shape=(count,)
    )
    surge[:] = surge_frame.storm_surge_m.to_numpy(dtype=np.float32)
    times = np.lib.format.open_memmap(
        dataset_dir / "time.npy", mode="w+", dtype="datetime64[ns]", shape=(count,)
    )
    times[:] = surge_times.to_numpy(dtype="datetime64[ns]")

    written = np.zeros(count, dtype=bool)
    yearly_reports = []
    for year in range(start_year, end_year + 1):
        year_start = pd.Timestamp(year, 1, 1)
        year_end = pd.Timestamp(year, 12, 31, 23)
        era5 = load_era5_files(
            era5_files,
            latitude,
            longitude,
            grid_size=grid_size,
            region_size_degrees=region_size_degrees,
            start_time=year_start,
            end_time=year_end,
        )
        era_times = pd.DatetimeIndex(pd.to_datetime(era5.time.values))
        positions = surge_times.get_indexer(era_times)
        matched = positions >= 0
        if matched.any():
            atmosphere[positions[matched]] = np.asarray(
                era5.values[matched], dtype=np.float32
            )
            written[positions[matched]] = True
        atmosphere.flush()
        yearly_reports.append(
            {
                "year": year,
                "era5_times": len(era_times),
                "matched_times": int(matched.sum()),
            }
        )
        print(
            f"[PREPARE] {year}: ERA5={len(era_times):,}, matched={int(matched.sum()):,}",
            flush=True,
        )

    surge.flush()
    times.flush()
    report = {
        **station.as_metadata(),
        "station_id": station.station_id,
        "station_name": station.name,
        "latitude": latitude,
        "longitude": longitude,
        "start_year": start_year,
        "end_year": end_year,
        "utide_calibration_end_year": calibration_end_year,
        "records": count,
        "matched_era5_times": int(written.sum()),
        "missing_era5_times": int((~written).sum()),
        "era5_files": [str(path) for path in era5_files],
        "era5_source_audit": era5_source_audit,
        "gesla_file": str(Path(gesla_file)),
        "gesla_raw_time_range": (
            [raw_valid_times.min().isoformat(), raw_valid_times.max().isoformat()]
            if len(raw_valid_times) else None
        ),
        "selected_period_observation_count": int(len(cleaned)),
        "qc": qc_report,
        "utide": utide_report,
        "years": yearly_reports,
    }
    (output / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dataset_metadata = {
        **station.as_metadata(),
        "station_id": station.station_id,
        "station_name": station.name,
        "prepared_years": [start_year, end_year],
        "utide_calibration_end_year": calibration_end_year,
        "array_schema": {
            "atmosphere": [count, 3, grid_size, grid_size],
            "surge": [count],
            "time": [count],
            "atmosphere_variable_order": ["U10", "V10", "MSL"],
            "surge_unit": "m",
            "time_standard": "UTC hourly",
        },
    }
    (dataset_dir / "dataset_metadata.json").write_text(
        json.dumps(dataset_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared dataset: {dataset_dir.resolve()}")
    return dataset_dir


def prepare_xiamen(
    era5_dir: str | Path,
    gesla_file: str | Path,
    output_dir: str | Path,
    start_year: int = 1970,
    end_year: int = 1997,
    calibration_end_year: int = 1995,
    latitude: float = 24.45,
    longitude: float = 118.067,
    grid_size: int = 40,
    region_size_degrees: float = 10.0,
) -> Path:
    """Backward-compatible Xiamen API."""
    return prepare_station(
        "xiamen", era5_dir, gesla_file, output_dir, start_year, end_year,
        calibration_end_year, latitude, longitude, grid_size,
        region_size_degrees,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", choices=tuple(STATIONS), default="xiamen")
    parser.add_argument("--era5-dir", type=Path)
    parser.add_argument("--gesla-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--calibration-end-year", type=int)
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--region-size-degrees", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    cli_station = get_station_config(args.station)
    cli_output = args.output_dir or cli_station.processed_root
    validate_output_location(
        cli_output, cli_station.processed_root, "Prepared-data output"
    )
    prepare_station(
        args.station,
        args.era5_dir,
        args.gesla_file,
        args.output_dir,
        args.start_year,
        args.end_year,
        args.calibration_end_year,
        args.latitude,
        args.longitude,
        args.grid_size,
        args.region_size_degrees,
    )
