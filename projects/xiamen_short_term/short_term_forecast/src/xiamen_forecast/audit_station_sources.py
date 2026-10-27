"""Discover and audit Shanwei source files before creating a local station config."""

from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path

import pandas as pd

from .era5_loader import inspect_era5_files
from .station_config import MODULE_ROOT, SHANWEI_CONFIG, validate_output_location
from .tide_gauge_loader import load_tide_gauge
from .tide_quality_control import quality_control


def discover(gesla_dir: Path, era5_dir: Path) -> dict[str, list[str]]:
    if not gesla_dir.is_dir() or not era5_dir.is_dir():
        raise FileNotFoundError(f"Source directories must exist: {gesla_dir}, {era5_dir}")
    return {
        "gesla_candidates": [str(p) for p in sorted(gesla_dir.iterdir()) if p.is_file() and "shanwei" in p.name.lower()],
        "era5_candidates": [
            str(p) for p in sorted(era5_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in {".nc", ".nc4", ".cdf"}
        ],
    }


def annual_coverage(
    valid_times: pd.DatetimeIndex,
    era_start: pd.Timestamp,
    era_end: pd.Timestamp,
    first_year: int,
    last_year: int,
) -> list[dict[str, int | float]]:
    unique = valid_times.drop_duplicates()
    reports = []
    for year in range(first_year, last_year + 1):
        start = pd.Timestamp(year, 1, 1, tz="UTC")
        end = pd.Timestamp(year, 12, 31, 23, tz="UTC")
        observed = int(((unique >= start) & (unique <= end)).sum())
        overlap_start, overlap_end = max(start, era_start), min(end, era_end)
        matched = int(((unique >= overlap_start) & (unique <= overlap_end)).sum()) if overlap_start <= overlap_end else 0
        expected = (366 if calendar.isleap(year) else 365) * 24
        reports.append({
            "year": year,
            "expected_hours": expected,
            "valid_gesla_hours": observed,
            "matched_hours": matched,
            "matched_fraction": round(matched / expected, 6),
        })
    return reports


def choose_training_start(coverage: list[dict[str, int | float]], minimum_fraction: float) -> int:
    by_year = {int(row["year"]): float(row["matched_fraction"]) for row in coverage}
    if by_year.get(1996, 0) < minimum_fraction or by_year.get(1997, 0) < minimum_fraction:
        raise ValueError("1996 validation or 1997 test lacks the required common hourly coverage")
    qualifying = [year for year in sorted(by_year) if year <= 1995 and by_year[year] >= minimum_fraction]
    if len(qualifying) < 5:
        raise ValueError("Fewer than five training years have sufficient common hourly coverage")
    return qualifying[0]


def configure(
    gesla_file: Path,
    era5_files: dict[str, Path],
    latitude: float,
    longitude: float,
    output_dir: Path,
    minimum_fraction: float = 0.8,
    config_path: Path = SHANWEI_CONFIG,
) -> dict[str, object]:
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Verified latitude/longitude must be within geographic bounds")
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum_fraction must be in (0, 1]")
    if set(era5_files) != {"U10", "V10", "MSL"}:
        raise ValueError("Exactly U10, V10 and MSL ERA5 files are required")
    paths = [Path(era5_files[name]) for name in ("U10", "V10", "MSL")]
    if len(set(paths)) != 3 or len({p.parent.resolve() for p in paths}) != 1:
        raise ValueError("ERA5 variables must be three distinct files in one directory")
    for destination in (output_dir, config_path):
        for raw_root in (gesla_file.parent, paths[0].parent):
            try:
                destination.resolve(strict=False).relative_to(raw_root.resolve(strict=False))
            except ValueError:
                continue
            raise ValueError(f"Audit/config output must not be inside a raw-data directory: {destination}")
    missing = [str(p) for p in (gesla_file, *paths) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing source files: " + "; ".join(missing))
    if output_dir.exists() or config_path.exists():
        raise FileExistsError("Audit/config already exists; preserve it and choose a new output location")
    era_report = inspect_era5_files(paths)
    if era_report["status"] != "passed":
        raise ValueError("ERA5 source audit must pass: " + "; ".join(era_report["errors"] + era_report["warnings"]))
    for name in ("U10", "V10", "MSL"):
        actual = Path(era_report["variables"][name]["file"])
        if actual.resolve() != Path(era5_files[name]).resolve():
            raise ValueError(f"{name} file contains a different ERA5 variable: {actual}")
    u10 = era_report["variables"]["U10"]
    if not (u10["latitude_range"][0] <= latitude <= u10["latitude_range"][1]
            and u10["longitude_range"][0] <= longitude <= u10["longitude_range"][1]):
        raise ValueError("Verified station coordinates fall outside the ERA5 grid")
    era_start = pd.Timestamp(u10["time_range"][0], tz="UTC")
    era_end = pd.Timestamp(u10["time_range"][1], tz="UTC")
    raw = load_tide_gauge(gesla_file, station_id="shanwei", source="GESLA 3", water_level_unit="m")
    cleaned, qc = quality_control(raw)
    timestamps = pd.DatetimeIndex(pd.to_datetime(cleaned.datetime, utc=True))
    hourly = timestamps[timestamps == timestamps.floor("h")]
    first_year = max(int(hourly.min().year), int(era_start.year))
    coverage = annual_coverage(hourly, era_start, era_end, first_year, 1997)
    training_start = choose_training_start(coverage, minimum_fraction)
    report: dict[str, object] = {
        "status": "passed",
        "station_id": "shanwei",
        "gesla_file": str(gesla_file),
        "gesla_raw_range": [str(raw.datetime.min()), str(raw.datetime.max())],
        "gesla_qc": qc,
        "era5": era_report,
        "annual_common_coverage": coverage,
        "minimum_annual_fraction": minimum_fraction,
        "selected_training_years": [training_start, 1995],
        "validation_year": 1996,
        "test_year": 1997,
        "coordinates_supplied_by_operator": {"latitude": latitude, "longitude": longitude},
        "note": "Source metadata/QC only; prepared arrays and model performance are not yet verified.",
    }
    config = {
        "station_id": "shanwei", "source_audit_status": "passed",
        "latitude": latitude, "longitude": longitude,
        "gesla_file": str(gesla_file), "era5_dir": str(paths[0].parent),
        "era5_filenames": {name: Path(era5_files[name]).name for name in ("U10", "V10", "MSL")},
        "data_start_year": training_start, "data_end_year": 1997,
        "train_start_year": training_start,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "source_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("discover")
    listing.add_argument("--gesla-dir", type=Path, required=True)
    listing.add_argument("--era5-dir", type=Path, required=True)
    setup = sub.add_parser("configure")
    setup.add_argument("--gesla-file", type=Path, required=True)
    setup.add_argument("--u10-file", type=Path, required=True)
    setup.add_argument("--v10-file", type=Path, required=True)
    setup.add_argument("--msl-file", type=Path, required=True)
    setup.add_argument("--latitude", type=float, required=True)
    setup.add_argument("--longitude", type=float, required=True)
    setup.add_argument("--minimum-annual-coverage", type=float, default=0.8)
    setup.add_argument("--output-dir", type=Path, default=MODULE_ROOT / "outputs" / "source_audits" / "shanwei")
    args = parser.parse_args()
    if args.action == "discover":
        print(json.dumps(discover(args.gesla_dir, args.era5_dir), ensure_ascii=False, indent=2))
    else:
        validate_output_location(args.output_dir, MODULE_ROOT / "outputs", "Source audit output")
        report = configure(
            args.gesla_file,
            {"U10": args.u10_file, "V10": args.v10_file, "MSL": args.msl_file},
            args.latitude, args.longitude, args.output_dir, args.minimum_annual_coverage,
        )
        print(json.dumps({"status": report["status"], "selected_training_years": report["selected_training_years"], "config": str(SHANWEI_CONFIG)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
