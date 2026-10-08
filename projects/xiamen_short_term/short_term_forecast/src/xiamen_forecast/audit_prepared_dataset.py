"""Audit a prepared memory-mapped dataset before model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .dataset_builder import valid_targets
    from .station_config import STATIONS, get_station_config
except ImportError:
    from dataset_builder import valid_targets
    from station_config import STATIONS, get_station_config


MODULE_ROOT = Path(__file__).resolve().parents[2]
VARIABLES = ("U10", "V10", "MSL")
EXPECTED_UNITS = {"U10": "m/s", "V10": "m/s", "MSL": "Pa"}
LEADS = (1, 3, 6, 12, 24, 48, 72)


def _variable_statistics(atmosphere: np.ndarray, chunk_hours: int = 168) -> list[dict[str, Any]]:
    variables = atmosphere.shape[1]
    counts = np.zeros(variables, dtype=np.int64)
    missing = np.zeros(variables, dtype=np.int64)
    sums = np.zeros(variables, dtype=np.float64)
    minima = np.full(variables, np.inf, dtype=np.float64)
    maxima = np.full(variables, -np.inf, dtype=np.float64)
    for start in range(0, len(atmosphere), chunk_hours):
        chunk = np.asarray(atmosphere[start : start + chunk_hours], dtype=np.float64)
        for variable in range(variables):
            values = chunk[:, variable]
            finite = np.isfinite(values)
            counts[variable] += int(finite.sum())
            missing[variable] += int((~finite).sum())
            if finite.any():
                finite_values = values[finite]
                sums[variable] += finite_values.sum()
                minima[variable] = min(minima[variable], float(finite_values.min()))
                maxima[variable] = max(maxima[variable], float(finite_values.max()))
    reports = []
    for index, name in enumerate(VARIABLES):
        reports.append(
            {
                "variable": name,
                "expected_unit": EXPECTED_UNITS[name],
                "finite_count": int(counts[index]),
                "missing_count": int(missing[index]),
                "finite_fraction": float(counts[index] / (counts[index] + missing[index])),
                "minimum": float(minima[index]),
                "maximum": float(maxima[index]),
                "mean": float(sums[index] / counts[index]),
            }
        )
    return reports


def _rolling_origin_count(
    times: pd.DatetimeIndex,
    atmosphere: np.ndarray,
    surge: np.ndarray,
    year: int,
    input_steps: int,
    max_lead: int = 72,
) -> int:
    atmospheric_valid = np.zeros(len(times), dtype=bool)
    for start in range(0, len(times), 168):
        chunk = np.asarray(atmosphere[start : start + 168])
        atmospheric_valid[start : start + len(chunk)] = np.isfinite(chunk).all(
            axis=(1, 2, 3)
        )
    surge_values = np.asarray(surge, dtype=float)
    count = 0
    for origin in np.flatnonzero(times.year == year):
        end = int(origin) + max_lead - 1
        if origin < input_steps or end >= len(times) or times[end].year != year:
            continue
        if not np.all(
            np.diff(times[origin - input_steps : end + 1].values)
            == np.timedelta64(1, "h")
        ):
            continue
        if not atmospheric_valid[origin - input_steps : end].all():
            continue
        if not np.isfinite(surge_values[origin - input_steps : origin]).all():
            continue
        verification = int(origin) + np.asarray(LEADS) - 1
        if not np.isfinite(surge_values[verification]).all():
            continue
        if not np.isfinite(surge_values[verification - 1]).all():
            continue
        count += 1
    return count


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def audit_dataset(
    dataset_dir: str | Path,
    input_steps: int = 24,
    output_path: str | Path | None = None,
    station_id: str | None = None,
    train_start_year: int | None = None,
    train_end_year: int | None = None,
    validation_year: int | None = None,
    test_year: int | None = None,
    raise_on_error: bool = True,
    expected_grid_size: int | None = None,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    station = get_station_config(station_id or root.parent.name)
    train_start_year = station.train_start_year if train_start_year is None else train_start_year
    train_end_year = station.train_end_year if train_end_year is None else train_end_year
    validation_year = station.validation_year if validation_year is None else validation_year
    test_year = station.test_year if test_year is None else test_year
    expected_grid_size = (
        station.grid_size if expected_grid_size is None else expected_grid_size
    )
    if not train_start_year <= train_end_year < validation_year < test_year:
        raise ValueError("Invalid chronological train/validation/test year split")
    paths = {name: root / f"{name}.npy" for name in ("atmosphere", "surge", "time")}
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Prepared dataset is incomplete: {missing_files}")
    atmosphere = np.load(paths["atmosphere"], mmap_mode="r", allow_pickle=False)
    surge = np.load(paths["surge"], mmap_mode="r", allow_pickle=False)
    times = pd.DatetimeIndex(
        pd.to_datetime(np.load(paths["time"], mmap_mode="r", allow_pickle=False))
    )
    if atmosphere.shape != (
        len(times), 3, expected_grid_size, expected_grid_size
    ):
        raise ValueError(f"Unexpected atmosphere shape: {atmosphere.shape}")
    if surge.shape != (len(times),):
        raise ValueError(f"Unexpected surge shape: {surge.shape}")

    statistics = _variable_statistics(atmosphere)
    targets, skipped = valid_targets(times, atmosphere, surge, input_steps)
    target_years = times[np.asarray(targets, dtype=int)].year if targets else np.asarray([])
    yearly: dict[str, Any] = {}
    for year in sorted(set(times.year)):
        positions = np.flatnonzero(times.year == year)
        start, stop = int(positions[0]), int(positions[-1]) + 1
        year_targets, year_skipped = valid_targets(
            times[start:stop], atmosphere[start:stop], surge[start:stop], input_steps
        )
        yearly[str(year)] = {
            "hours": stop - start,
            "time_range": [str(times[start]), str(times[stop - 1])],
            "finite_observations": int(np.isfinite(surge[start:stop]).sum()),
            "missing_observations": int((~np.isfinite(surge[start:stop])).sum()),
            "valid_samples": len(year_targets),
            "skipped": year_skipped,
        }

    msl = next(item for item in statistics if item["variable"] == "MSL")
    msl_unit_check = 80_000 <= msl["mean"] <= 120_000
    finite_surge = np.asarray(surge, dtype=float)
    finite_surge = finite_surge[np.isfinite(finite_surge)]
    if not finite_surge.size:
        raise ValueError("Prepared surge array has no finite values")
    preparation_report = _read_json(root.parent / "preparation_report.json")
    dataset_metadata = _read_json(root / "dataset_metadata.json")
    qc_report = _read_json(root.parent / "tide_qc_report.json")
    split_samples = {
        "training": int(
            np.sum((target_years >= train_start_year) & (target_years <= train_end_year))
        ),
        "validation": int(np.sum(target_years == validation_year)),
        "test": int(np.sum(target_years == test_year)),
    }
    rolling_origins = {
        "validation": _rolling_origin_count(
            times, atmosphere, surge, validation_year, input_steps
        ),
        "test": _rolling_origin_count(
            times, atmosphere, surge, test_year, input_steps
        ),
    }
    errors: list[str] = []
    warnings: list[str] = []
    strictly_hourly = bool(
        np.all(np.diff(times.values) == np.timedelta64(1, "h"))
    )
    if not strictly_hourly:
        errors.append("time.npy is not strictly hourly and continuous")
    if not msl_unit_check:
        errors.append("MSL mean is outside the expected 80,000-120,000 Pa range")
    for name, count in split_samples.items():
        if count == 0:
            errors.append(f"no valid 24-hour samples remain in the {name} split")
    for name, count in rolling_origins.items():
        if count == 0:
            errors.append(f"no valid common 72-hour origins remain in the {name} split")
    if dataset_metadata is None:
        errors.append(
            "dataset_metadata.json is missing; station identity and preparation settings cannot be verified"
        )
    elif dataset_metadata.get("station_id") != station.station_id:
        errors.append(
            "dataset station metadata does not match requested station: "
            f"{dataset_metadata.get('station_id')} != {station.station_id}"
        )
    if dataset_metadata is not None:
        prepared_years = dataset_metadata.get("prepared_years")
        if prepared_years and (
            int(prepared_years[0]) > train_start_year
            or int(prepared_years[1]) < test_year
        ):
            errors.append(
                "prepared dataset does not cover the configured train/validation/test years"
            )
        calibration_end = dataset_metadata.get("utide_calibration_end_year")
        if calibration_end is None:
            warnings.append(
                "UTide calibration cutoff is absent from dataset metadata"
            )
        elif int(calibration_end) != train_end_year:
            errors.append(
                "UTide calibration cutoff does not equal the final training year: "
                f"{calibration_end} != {train_end_year}"
            )
        schema = dataset_metadata.get("array_schema", {})
        variable_order = schema.get("atmosphere_variable_order")
        if variable_order is None:
            warnings.append("ERA5 variable order is absent from dataset metadata")
        elif list(variable_order) != list(VARIABLES):
            errors.append(
                f"ERA5 variable order is {variable_order}; expected {list(VARIABLES)}"
            )
    if preparation_report is None:
        warnings.append("preparation_report.json is missing; raw source coverage cannot be audited")
    if qc_report is None:
        warnings.append("tide_qc_report.json is missing; GESLA flag rejection cannot be audited")
    elif qc_report.get("raw_record_count"):
        removed = int(qc_report.get("removed_quality_flag_count", 0))
        fraction = removed / int(qc_report["raw_record_count"])
        if fraction > 0.2:
            warnings.append(f"GESLA quality/use flags rejected {fraction:.1%} of raw records")
        rejected_5_0 = int(
            qc_report.get("quality_use_flag_pairs", {}).get("5|0", 0)
        )
        if rejected_5_0 and removed < rejected_5_0:
            errors.append(
                "GESLA 5|0 records were reported but not fully rejected by quality control"
            )
    source_audit = (
        preparation_report.get("era5_source_audit")
        if preparation_report else None
    )
    if source_audit and source_audit.get("status") == "failed":
        errors.append("ERA5 source coordinate/unit audit failed during preparation")
    elif source_audit and source_audit.get("warnings"):
        warnings.extend(
            f"ERA5 source audit: {message}"
            for message in source_audit["warnings"]
        )
    if preparation_report and int(
        preparation_report.get("missing_era5_times", 0)
    ):
        errors.append(
            "prepared data contain unmatched ERA5 hours: "
            f"{preparation_report['missing_era5_times']}"
        )
    for item in statistics[:2]:
        if max(abs(item["minimum"]), abs(item["maximum"])) > 150:
            errors.append(f"{item['variable']} exceeds the conservative +/-150 m/s range")
    report: dict[str, Any] = {
        "status": "failed" if errors else ("warning" if warnings else "passed"),
        "station": station.as_metadata(),
        "dataset_dir": str(root.resolve()),
        "array_shapes": {
            "atmosphere": list(atmosphere.shape),
            "surge": list(surge.shape),
            "time": [len(times)],
        },
        "file_sizes_bytes": {name: path.stat().st_size for name, path in paths.items()},
        "atmosphere_size_gib": paths["atmosphere"].stat().st_size / 1024**3,
        "time_range": [str(times[0]), str(times[-1])],
        "strictly_hourly": strictly_hourly,
        "surge_finite_count": int(np.isfinite(surge).sum()),
        "surge_missing_count": int((~np.isfinite(surge)).sum()),
        "surge_statistics_cm": {
            "mean": float(np.mean(finite_surge) * 100),
            "standard_deviation": float(np.std(finite_surge) * 100),
            "minimum": float(np.min(finite_surge) * 100),
            "maximum": float(np.max(finite_surge) * 100),
            "absolute_p90": float(np.quantile(np.abs(finite_surge), 0.90) * 100),
            "absolute_p95": float(np.quantile(np.abs(finite_surge), 0.95) * 100),
            "absolute_p99": float(np.quantile(np.abs(finite_surge), 0.99) * 100),
        },
        "variables": statistics,
        "msl_unit_check": {
            "expected_unit": "Pa",
            "mean_in_expected_pressure_range": msl_unit_check,
            "conclusion": "MSL remains in Pa" if msl_unit_check else "MSL unit/value requires review",
        },
        "input_steps": input_steps,
        "expected_grid_size": expected_grid_size,
        "valid_samples": len(targets),
        "skipped": skipped,
        "yearly_samples": yearly,
        "split_years": {
            "training": [train_start_year, train_end_year],
            "validation": validation_year,
            "test": test_year,
        },
        "split_valid_samples": split_samples,
        "common_72h_origins": rolling_origins,
        "raw_source_coverage": {
            "gesla": preparation_report.get("gesla_raw_time_range") if preparation_report else None,
            "gesla_selected_period_observation_count": (
                preparation_report.get("selected_period_observation_count")
                if preparation_report else None
            ),
            "era5_yearly_match": preparation_report.get("years") if preparation_report else None,
            "era5_files": preparation_report.get("era5_files") if preparation_report else None,
            "era5_source_audit": (
                preparation_report.get("era5_source_audit")
                if preparation_report else None
            ),
        },
        "gesla_quality_control": qc_report,
        "errors": errors,
        "warnings": warnings,
    }
    destination = Path(output_path) if output_path else root.parent / "prepared_dataset_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors and raise_on_error:
        raise ValueError(
            f"Prepared dataset audit failed; see {destination}: " + "; ".join(errors)
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", choices=tuple(STATIONS), default="xiamen")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--input-steps", type=int, default=24)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-start-year", type=int)
    parser.add_argument("--train-end-year", type=int)
    parser.add_argument("--validation-year", type=int)
    parser.add_argument("--test-year", type=int)
    parser.add_argument(
        "--expected-grid-size",
        type=int,
        help="Testing override; formal station data default to the configured 40x40 grid.",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Write a failed audit report without raising an exception.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    station = get_station_config(arguments.station)
    dataset_dir = arguments.dataset_dir or station.dataset_dir
    result = audit_dataset(
        dataset_dir,
        arguments.input_steps,
        arguments.output,
        arguments.station,
        arguments.train_start_year,
        arguments.train_end_year,
        arguments.validation_year,
        arguments.test_year,
        not arguments.report_only,
        arguments.expected_grid_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
