"""Download real-time GFS U10/V10/SLP forcing fields from NOAA NOMADS.

The downloaded files are GRIB2 subsets clipped by region and variables:
- UGRD at 10 m above ground
- VGRD at 10 m above ground
- PRMSL at mean sea level

Example:
python real_time_data_crawler/gfs_nomads_downloader.py --date 20260706 --cycle 06 --hours 0 1 2 3 6 12 24
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener


BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
SCRIPT_DIR = Path(__file__).resolve().parent
from runtime_paths import GFS_OUTPUT_DIR

DEFAULT_OUT_DIR = GFS_OUTPUT_DIR
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)

DEFAULT_BBOX = {
    "leftlon": 110,
    "rightlon": 130,
    "bottomlat": 15,
    "toplat": 35,
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def latest_cycle(now: datetime | None = None) -> tuple[str, str]:
    """Return a conservative latest GFS cycle date/cycle in UTC.

    GFS cycles are 00/06/12/18 UTC. NOMADS availability lags the nominal cycle,
    so this uses a 5-hour delay to avoid requesting a cycle that is still being
    generated.
    """

    now = now or datetime.now(timezone.utc)
    available_time = now - timedelta(hours=5)
    cycle_hour = (available_time.hour // 6) * 6
    cycle_dt = available_time.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    return cycle_dt.strftime("%Y%m%d"), f"{cycle_hour:02d}"


def build_url(date: str, cycle: str, forecast_hour: int, bbox: dict[str, float]) -> str:
    file_name = f"gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour:03d}"
    params = {
        "dir": f"/gfs.{date}/{cycle}/atmos",
        "file": file_name,
        "var_UGRD": "on",
        "var_VGRD": "on",
        "var_PRMSL": "on",
        "lev_10_m_above_ground": "on",
        "lev_mean_sea_level": "on",
        "subregion": "",
        "leftlon": bbox["leftlon"],
        "rightlon": bbox["rightlon"],
        "bottomlat": bbox["bottomlat"],
        "toplat": bbox["toplat"],
    }
    return f"{BASE_URL}?{urlencode(params)}"


def download_one(url: str, out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        print(f"skip existing: {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    opener = build_opener(ProxyHandler({}))

    try:
        with opener.open(request, timeout=120) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} when downloading {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error when downloading {url}: {exc}") from exc

    if len(data) < 2000 and (b"<!DOCTYPE" in data[:500] or b"Invalid" in data[:500] or b"Error" in data[:500]):
        preview = data[:500].decode("utf-8", errors="replace")
        raise RuntimeError(f"NOMADS did not return GRIB2 data for {url}\n{preview}")

    out_path.write_bytes(data)
    print(f"saved: {out_path} ({len(data) / 1024:.1f} KB, {content_type})")


def parse_hours(values: Iterable[str]) -> list[int]:
    hours: list[int] = []
    for value in values:
        if "-" in value:
            start_text, end_text = value.split("-", 1)
            start, end = int(start_text), int(end_text)
            hours.extend(range(start, end + 1))
        else:
            hours.append(int(value))
    return sorted(set(hours))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download GFS U10/V10/PRMSL GRIB2 subsets from NOMADS.")
    parser.add_argument("--date", default=None, help="UTC cycle date, e.g. 20260706. Default: latest available")
    parser.add_argument("--cycle", default=None, choices=["00", "06", "12", "18"], help="UTC cycle hour")
    parser.add_argument("--hours", nargs="+", default=["0-24"], help="Forecast hours, e.g. 0 1 2 3 or 0-24")
    parser.add_argument("--leftlon", type=float, default=DEFAULT_BBOX["leftlon"])
    parser.add_argument("--rightlon", type=float, default=DEFAULT_BBOX["rightlon"])
    parser.add_argument("--bottomlat", type=float, default=DEFAULT_BBOX["bottomlat"])
    parser.add_argument("--toplat", type=float, default=DEFAULT_BBOX["toplat"])
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing GRIB2 files")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    date, cycle = (args.date, args.cycle) if args.date and args.cycle else latest_cycle()
    if bool(args.date) ^ bool(args.cycle):
        raise SystemExit("--date and --cycle should be provided together, or omitted together.")

    bbox = {
        "leftlon": args.leftlon,
        "rightlon": args.rightlon,
        "bottomlat": args.bottomlat,
        "toplat": args.toplat,
    }
    out_dir = Path(args.out) / f"gfs_{date}_{cycle}z"

    for hour in parse_hours(args.hours):
        url = build_url(date, cycle, hour, bbox)
        file_name = f"gfs_{date}_{cycle}z_f{hour:03d}_u10_v10_prmsl_{bbox['leftlon']}_{bbox['rightlon']}_{bbox['bottomlat']}_{bbox['toplat']}.grib2"
        download_one(url, out_dir / file_name, args.overwrite)


if __name__ == "__main__":
    main()
