"""Select coastal tide/storm-surge stations near a typhoon forecast path.

This utility combines:
- NMC typhoon raw JSON from nmc_typhoon_crawler.py
- NMEFC station list from nmefc_storm_surge_crawler.py

It is useful for deciding which tide-gauge stations should be searched for
observed water-level data near a coming typhoon.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
from runtime_paths import BAVI_TIDE_CANDIDATES_PATH, NMC_OUTPUT_DIR, NMEFC_OUTPUT_DIR

DEFAULT_OUTPUT_DIR = NMC_OUTPUT_DIR


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def load_babj_forecast(raw_json: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    typhoon = json.loads(raw_json.read_text(encoding="utf-8"))["typhoon"]
    points = typhoon[8]
    latest = points[-1]
    forecasts = latest[11].get("BABJ", [])
    path = [
        {
            "lead_hour": item[0],
            "lon": float(item[2]),
            "lat": float(item[3]),
            "pressure_hpa": item[4],
            "max_wind_ms": item[5],
            "grade": item[7],
        }
        for item in forecasts
    ]
    latest_row = {
        "time_bj": latest[1],
        "lon": latest[4],
        "lat": latest[5],
        "pressure_hpa": latest[6],
        "max_wind_ms": latest[7],
        "grade": latest[3],
    }
    return latest_row, path


def load_sites(path: Path) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            try:
                row["lat_float"] = float(row["lat"])
                row["lon_float"] = float(row["lon"])
            except (ValueError, TypeError):
                continue
            sites.append(row)
    return sites


def select_nearest(
    sites: list[dict[str, Any]],
    forecast_path: list[dict[str, Any]],
    limit: int,
    max_distance_km: float | None,
    provinces: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for site in sites:
        if provinces and not any(province in site.get("province", "") for province in provinces):
            continue
        nearest_distance = float("inf")
        nearest_lead = ""
        nearest_lon = ""
        nearest_lat = ""
        for point in forecast_path:
            distance = haversine_km(site["lat_float"], site["lon_float"], point["lat"], point["lon"])
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_lead = point["lead_hour"]
                nearest_lon = point["lon"]
                nearest_lat = point["lat"]
        if max_distance_km is not None and nearest_distance > max_distance_km:
            continue
        rows.append(
            {
                "province": site.get("province", ""),
                "siteName": site.get("siteName", ""),
                "siteCode": site.get("siteCode", ""),
                "station_lon": site.get("lon", ""),
                "station_lat": site.get("lat", ""),
                "siteArea": site.get("siteArea", ""),
                "blue": site.get("blue", ""),
                "yellow": site.get("yellow", ""),
                "orange": site.get("orange", ""),
                "red": site.get("red", ""),
                "nearest_lead_hour": nearest_lead,
                "nearest_forecast_lon": nearest_lon,
                "nearest_forecast_lat": nearest_lat,
                "min_distance_km": round(nearest_distance, 1),
            }
        )
    return sorted(rows, key=lambda row: row["min_distance_km"])[:limit]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "province",
        "siteName",
        "siteCode",
        "station_lon",
        "station_lat",
        "siteArea",
        "blue",
        "yellow",
        "orange",
        "red",
        "nearest_lead_hour",
        "nearest_forecast_lon",
        "nearest_forecast_lat",
        "min_distance_km",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select NMEFC coastal stations near an NMC typhoon path.")
    parser.add_argument(
        "--nmc-raw",
        default=str(DEFAULT_OUTPUT_DIR / "raw_view_3257931_巴威.json"),
        help="NMC raw typhoon JSON",
    )
    parser.add_argument(
        "--nmefc-sites",
        default=str(NMEFC_OUTPUT_DIR / "nmefc_typhoon_sites_all.csv"),
        help="NMEFC station CSV",
    )
    parser.add_argument("--limit", type=int, default=30, help="Maximum stations to output")
    parser.add_argument("--max-distance-km", type=float, default=None, help="Optional distance threshold")
    parser.add_argument("--province", action="append", default=[], help="Province filter. Can be repeated.")
    parser.add_argument(
        "--out",
        default=str(BAVI_TIDE_CANDIDATES_PATH),
        help="Output CSV",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    latest, forecast_path = load_babj_forecast(Path(args.nmc_raw))
    sites = load_sites(Path(args.nmefc_sites))
    rows = select_nearest(sites, forecast_path, args.limit, args.max_distance_km, args.province)
    write_csv(Path(args.out), rows)
    print(f"latest typhoon point: {latest}")
    print(f"forecast points: {len(forecast_path)}")
    print(f"station candidates: {len(rows)}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
