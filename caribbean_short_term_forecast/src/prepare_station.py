"""Prepare QC/UTide/ERA5-aligned memory-mapped arrays for one configured station."""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .era5_loader import load_era5_file
    from .station_registry import get_station
    from .tide_gauge_loader import load_tide_gauge
    from .tide_processing import separate_tide
    from .tide_quality_control import quality_control
except ImportError:
    from era5_loader import load_era5_file
    from station_registry import get_station
    from tide_gauge_loader import load_tide_gauge
    from tide_processing import separate_tide
    from tide_quality_control import quality_control


MODULE_ROOT = Path(__file__).resolve().parents[1]
ERA5_SUFFIXES = {".nc", ".nc4", ".cdf", ".grib", ".grb", ".grib2", ".grb2"}


def resolve_era5_files(value: str) -> list[Path]:
    candidate = Path(value)
    if candidate.is_file():
        paths = [candidate]
    elif candidate.is_dir():
        paths = [path for path in candidate.rglob("*") if path.is_file() and path.suffix.lower() in ERA5_SUFFIXES]
    else:
        paths = [Path(path) for path in glob.glob(value, recursive=True)]
        paths = [path for path in paths if path.is_file() and path.suffix.lower() in ERA5_SUFFIXES]
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No ERA5 NetCDF/GRIB files found at configured path: {value}")
    return paths


def prepare_station(station_id: str, output_dir: str | Path | None = None) -> Path:
    station = get_station(station_id).require(
        "latitude", "longitude", "timezone", "water_level_unit", "file_path", "era5_path"
    )
    output = Path(output_dir) if output_dir else MODULE_ROOT / "outputs" / "processed" / station_id
    tide_output = output / "tide"
    output.mkdir(parents=True, exist_ok=True)
    raw = load_tide_gauge(
        station.file_path, station.station_id, source=station.source or "auto",
        water_level_unit=station.water_level_unit or "m",
    )
    cleaned, qc_report = quality_control(raw, output / "tide_qc_report.json")
    surge_frame, utide_report = separate_tide(cleaned, float(station.latitude), tide_output)
    surge_times = pd.DatetimeIndex(pd.to_datetime(surge_frame.datetime, utc=True)).tz_convert("UTC").tz_localize(None)
    if not surge_times.is_monotonic_increasing or surge_times.has_duplicates:
        raise ValueError("Processed storm-surge timestamps must be unique and increasing")
    dataset_dir = output / "aligned_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    count = len(surge_times)
    atmosphere = np.lib.format.open_memmap(
        dataset_dir / "atmosphere.npy", mode="w+", dtype="float32", shape=(count, 3, 40, 40)
    )
    atmosphere[:] = np.nan
    surge_memmap = np.lib.format.open_memmap(dataset_dir / "surge.npy", mode="w+", dtype="float32", shape=(count,))
    surge_memmap[:] = surge_frame.storm_surge_m.to_numpy(dtype=np.float32)
    time_memmap = np.lib.format.open_memmap(dataset_dir / "time.npy", mode="w+", dtype="datetime64[ns]", shape=(count,))
    time_memmap[:] = surge_times.to_numpy(dtype="datetime64[ns]")

    files = resolve_era5_files(station.era5_path or "")
    written = np.zeros(count, dtype=bool)
    file_reports = []
    for path in files:
        era5 = load_era5_file(path, float(station.latitude), float(station.longitude))
        era_times = pd.DatetimeIndex(pd.to_datetime(era5.time.values))
        positions = surge_times.get_indexer(era_times)
        matched = positions >= 0
        if matched.any():
            atmosphere[positions[matched]] = np.asarray(era5.values[matched], dtype=np.float32)
            written[positions[matched]] = True
        file_reports.append({"file": str(path), "era5_times": len(era_times), "matched_times": int(matched.sum())})
        atmosphere.flush()
    surge_memmap.flush(); time_memmap.flush()
    report = {
        "station_id": station_id,
        "records": count,
        "time_range": [str(surge_times.min()), str(surge_times.max())],
        "matched_era5_times": int(written.sum()),
        "missing_era5_times": int((~written).sum()),
        "continuous_fully_aligned_segments": int(
            ((pd.Series(written).shift(fill_value=False) != pd.Series(written)) & pd.Series(written)).sum()
        ),
        "qc": qc_report,
        "utide": utide_report,
        "era5_files": file_reports,
    }
    (output / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("station_id", "records", "matched_era5_times", "missing_era5_times")}, indent=2))
    print(f"Prepared memory-mapped dataset: {dataset_dir.resolve()}")
    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = parse_args()
    prepare_station(arguments.station, arguments.output_dir)
