"""Create a lightweight inventory of Caribbean tide-gauge and ERA5 files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = MODULE_ROOT / "outputs" / "data_inventory.csv"
YEARS = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
STATIONS = {
    "prickly": "Prickly Bay",
    "calliaqua": "Calliaqua",
    "ganter": "Ganter's Bay",
    "san_juan": "San Juan",
    "san-juan": "San Juan",
    "charlotte": "Charlotte Amalie",
    "cristobal": "Cristobal",
}


def classify(path: Path) -> tuple[str, str, str]:
    text = str(path).lower()
    suffix = path.suffix.lower()
    station = next((name for token, name in STATIONS.items() if token in text), "")
    year_match = YEARS.search(path.name)
    year = year_match.group(1) if year_match else ""
    if suffix in {".nc", ".nc4", ".cdf"}:
        source = "ERA5 NetCDF"
    elif suffix in {".grib", ".grb", ".grib2", ".grb2"}:
        source = "ERA5 GRIB"
    elif "metadata" in text and "gesla" in text:
        source = "GESLA metadata"
    elif "gesla" in text:
        source = "GESLA station file"
    elif any(token in text for token in ("psmsl", "noc", "prickly", "calliaqua", "ganter")):
        source = "PSMSL/NOC station file"
    else:
        source = "unclassified"
    return source, station, year


def build_inventory(data_root: str | Path, output_path: str | Path = DEFAULT_OUTPUT) -> int:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data root {root} is not available. Please connect the external drive."
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file_path", "file_name", "suffix", "size_mb", "possible_source",
        "possible_station", "possible_year",
    ]
    count = 0
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError as exc:
                print(f"[WARN] Cannot inspect {path}: {exc}")
                continue
            source, station, year = classify(path)
            writer.writerow(
                {
                    "file_path": str(path.resolve()),
                    "file_name": path.name,
                    "suffix": path.suffix.lower(),
                    "size_mb": round(stat.st_size / 1024**2, 4),
                    "possible_source": source,
                    "possible_station": station,
                    "possible_year": year,
                }
            )
            count += 1
    print(f"[INVENTORY] {count} files -> {output.resolve()}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=r"F:\data")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        build_inventory(args.data_root, args.output)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
