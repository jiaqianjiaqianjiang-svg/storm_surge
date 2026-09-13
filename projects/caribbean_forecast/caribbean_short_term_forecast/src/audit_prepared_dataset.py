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
except ImportError:
    from dataset_builder import valid_targets


MODULE_ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ("U10", "V10", "MSL")
EXPECTED_UNITS = {"U10": "m/s", "V10": "m/s", "MSL": "Pa"}


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


def audit_dataset(
    dataset_dir: str | Path,
    input_steps: int = 24,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    paths = {name: root / f"{name}.npy" for name in ("atmosphere", "surge", "time")}
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Prepared dataset is incomplete: {missing_files}")
    atmosphere = np.load(paths["atmosphere"], mmap_mode="r", allow_pickle=False)
    surge = np.load(paths["surge"], mmap_mode="r", allow_pickle=False)
    times = pd.DatetimeIndex(
        pd.to_datetime(np.load(paths["time"], mmap_mode="r", allow_pickle=False))
    )
    if atmosphere.shape != (len(times), 3, 40, 40):
        raise ValueError(f"Unexpected atmosphere shape: {atmosphere.shape}")
    if surge.shape != (len(times),):
        raise ValueError(f"Unexpected surge shape: {surge.shape}")

    statistics = _variable_statistics(atmosphere)
    targets, skipped = valid_targets(times, atmosphere, surge, input_steps)
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
            "valid_samples": len(year_targets),
            "skipped": year_skipped,
        }

    msl = next(item for item in statistics if item["variable"] == "MSL")
    msl_unit_check = 80_000 <= msl["mean"] <= 120_000
    report: dict[str, Any] = {
        "dataset_dir": str(root.resolve()),
        "array_shapes": {
            "atmosphere": list(atmosphere.shape),
            "surge": list(surge.shape),
            "time": [len(times)],
        },
        "file_sizes_bytes": {name: path.stat().st_size for name, path in paths.items()},
        "atmosphere_size_gib": paths["atmosphere"].stat().st_size / 1024**3,
        "time_range": [str(times[0]), str(times[-1])],
        "strictly_hourly": bool(
            np.all(np.diff(times.values) == np.timedelta64(1, "h"))
        ),
        "surge_finite_count": int(np.isfinite(surge).sum()),
        "surge_missing_count": int((~np.isfinite(surge)).sum()),
        "variables": statistics,
        "msl_unit_check": {
            "expected_unit": "Pa",
            "mean_in_expected_pressure_range": msl_unit_check,
            "conclusion": "MSL remains in Pa" if msl_unit_check else "MSL unit/value requires review",
        },
        "input_steps": input_steps,
        "valid_samples": len(targets),
        "skipped": skipped,
        "yearly_samples": yearly,
    }
    destination = Path(output_path) if output_path else root.parent / "prepared_dataset_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=MODULE_ROOT / "outputs" / "processed" / "prickly_bay" / "aligned_dataset",
    )
    parser.add_argument("--input-steps", type=int, default=24)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = audit_dataset(arguments.dataset_dir, arguments.input_steps, arguments.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
