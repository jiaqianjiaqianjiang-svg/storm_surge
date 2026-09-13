#!/usr/bin/env python
"""Periodically fetch BAVI track data and GFS forcing fields.

This keeps the real-time workflow focused on two inputs:
1. China NMC BAVI track/forecast data.
2. GFS U10, V10 and PRMSL forcing from NOAA NOMADS.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import gfs_nomads_downloader as gfs
import nmc_typhoon_crawler as nmc


SCRIPT_DIR = Path(__file__).resolve().parent
from runtime_paths import TYPHOON_DATA_DIR

DEFAULT_OUTPUT_DIR = TYPHOON_DATA_DIR / "bavi_monitor"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_nmc(args: argparse.Namespace) -> None:
    targets = ", ".join(args.names) if args.names else args.name
    print(f"[{now_text()}] crawling NMC typhoon track: {targets}")
    nmc.crawl(
        SimpleNamespace(
            year=args.year,
            name=args.name,
            names=args.names,
            active=args.active_only,
            out=str(args.output_dir),
            save_list=False,
            save_raw=False,
            save_forecast=False,
        )
    )


def run_gfs(args: argparse.Namespace) -> tuple[str, str]:
    if args.gfs_date and args.gfs_cycle:
        date, cycle = args.gfs_date, args.gfs_cycle
    else:
        date, cycle = gfs.latest_cycle()

    bbox = {
        "leftlon": float(args.leftlon),
        "rightlon": float(args.rightlon),
        "bottomlat": float(args.bottomlat),
        "toplat": float(args.toplat),
    }
    hours = gfs.parse_hours(args.hours)
    out_dir = Path(args.output_dir) / "gfs" / f"gfs_{date}_{cycle}z"

    print(f"[{now_text()}] downloading GFS {date} {cycle}Z, hours={hours[0]}-{hours[-1]}")
    for hour in hours:
        url = gfs.build_url(date, cycle, hour, bbox)
        file_name = (
            f"gfs_{date}_{cycle}z_f{hour:03d}_u10_v10_prmsl_"
            f"{bbox['leftlon']}_{bbox['rightlon']}_{bbox['bottomlat']}_{bbox['toplat']}.grib2"
        )
        gfs.download_one(url, out_dir / file_name, args.overwrite_gfs)
    return date, cycle


def run_once(args: argparse.Namespace) -> bool:
    ok = True
    try:
        run_nmc(args)
    except Exception as exc:
        ok = False
        print(f"[{now_text()}] NMC failed: {exc}", file=sys.stderr)

    try:
        run_gfs(args)
    except Exception as exc:
        ok = False
        print(f"[{now_text()}] GFS failed: {exc}", file=sys.stderr)

    return ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuously fetch BAVI NMC track and GFS forcing data.")
    parser.add_argument("--name", default="巴威", help="Single typhoon keyword, e.g. 巴威, BAVI, 2609, or 3257931.")
    parser.add_argument("--names", nargs="+", default=None, help="Multiple typhoon keywords to crawl at once.")
    parser.add_argument("--year", default="2026", help="NMC typhoon list year.")
    parser.add_argument("--active-only", action="store_true", help="Only match active typhoons.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--interval-minutes", type=float, default=30, help="Polling interval in minutes.")
    parser.add_argument("--once", action="store_true", help="Run once and exit.")

    parser.add_argument("--gfs-date", default=None, help="Fixed GFS UTC date, e.g. 20260706.")
    parser.add_argument("--gfs-cycle", choices=["00", "06", "12", "18"], default=None, help="Fixed GFS UTC cycle.")
    parser.add_argument("--hours", nargs="+", default=["0-24"], help="GFS forecast hours, e.g. 0-24 or 0 1 2 6 12.")
    parser.add_argument("--leftlon", type=float, default=float(gfs.DEFAULT_BBOX["leftlon"]))
    parser.add_argument("--rightlon", type=float, default=float(gfs.DEFAULT_BBOX["rightlon"]))
    parser.add_argument("--bottomlat", type=float, default=float(gfs.DEFAULT_BBOX["bottomlat"]))
    parser.add_argument("--toplat", type=float, default=float(gfs.DEFAULT_BBOX["toplat"]))
    parser.add_argument("--overwrite-gfs", action="store_true", help="Overwrite existing GFS files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if bool(args.gfs_date) ^ bool(args.gfs_cycle):
        raise SystemExit("--gfs-date and --gfs-cycle should be provided together, or omitted together.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        run_once(args)
        if args.once:
            return 0
        sleep_seconds = max(args.interval_minutes, 0.1) * 60
        print(f"[{now_text()}] sleeping {args.interval_minutes:g} minutes. Press Ctrl+C to stop.")
        try:
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            print(f"\n[{now_text()}] stopped by user.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
