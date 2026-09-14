"""Prepare leakage-safe, memory-mapped Xiamen ERA5 and surge arrays."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .era5_loader import load_era5_files
    from .tide_gauge_loader import load_tide_gauge
    from .tide_processing import separate_tide
    from .tide_quality_control import quality_control
except ImportError:
    from era5_loader import load_era5_files
    from tide_gauge_loader import load_tide_gauge
    from tide_processing import separate_tide
    from tide_quality_control import quality_control


MODULE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ERA5_DIR = Path(r"F:\ERA5-NEW\Xiamen")
DEFAULT_GESLA_FILE = Path(r"F:\GESLA\GESLA3\xiamen-376a-chn-uhslc")
VARIABLE_HINTS = {
    "U10": ("10u", "u10", "var165"),
    "V10": ("v10", "10v", "var166"),
    "MSL": ("slp", "msl", "var151"),
}


def resolve_era5_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"ERA5 directory is not available: {root}")
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
    if not start_year <= calibration_end_year < end_year:
        raise ValueError("Expected start_year <= calibration_end_year < end_year")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    era5_files = resolve_era5_files(era5_dir)

    raw = load_tide_gauge(
        gesla_file, station_id="xiamen", source="GESLA 3", water_level_unit="m"
    )
    cleaned, qc_report = quality_control(raw, output / "tide_qc_report.json")
    start = pd.Timestamp(start_year, 1, 1, tz="UTC")
    end = pd.Timestamp(end_year, 12, 31, 23, tz="UTC")
    cleaned = cleaned.loc[
        (cleaned.datetime >= start) & (cleaned.datetime <= end)
    ].reset_index(drop=True)
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
        "station_id": "xiamen",
        "latitude": latitude,
        "longitude": longitude,
        "start_year": start_year,
        "end_year": end_year,
        "utide_calibration_end_year": calibration_end_year,
        "records": count,
        "matched_era5_times": int(written.sum()),
        "missing_era5_times": int((~written).sum()),
        "era5_files": [str(path) for path in era5_files],
        "gesla_file": str(Path(gesla_file)),
        "qc": qc_report,
        "utide": utide_report,
        "years": yearly_reports,
    }
    (output / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared dataset: {dataset_dir.resolve()}")
    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-dir", type=Path, default=DEFAULT_ERA5_DIR)
    parser.add_argument("--gesla-file", type=Path, default=DEFAULT_GESLA_FILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODULE_ROOT / "outputs" / "processed" / "xiamen",
    )
    parser.add_argument("--start-year", type=int, default=1970)
    parser.add_argument("--end-year", type=int, default=1997)
    parser.add_argument("--calibration-end-year", type=int, default=1995)
    parser.add_argument("--latitude", type=float, default=24.45)
    parser.add_argument("--longitude", type=float, default=118.067)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--region-size-degrees", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    prepare_xiamen(
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
