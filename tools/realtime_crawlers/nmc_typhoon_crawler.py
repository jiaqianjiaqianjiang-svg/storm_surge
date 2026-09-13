"""Fetch real-time typhoon track data from the China NMC typhoon website.

The NMC page exposes JSONP endpoints under:
http://typhoon.nmc.cn/weatherservice/typhoon/jsons/

By default this script saves only the observed track CSV, which is the current
real-time forecasting input. Extra raw/list/forecast files can be enabled with
command-line flags when needed for debugging.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener


BASE_URL = "http://typhoon.nmc.cn/weatherservice"
SCRIPT_DIR = Path(__file__).resolve().parent
from runtime_paths import NMC_OUTPUT_DIR

DEFAULT_OUT_DIR = NMC_OUTPUT_DIR
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


TYPHOON_LIST_COLUMNS = [
    "typhoon_id",
    "en_name",
    "cn_name",
    "code",
    "display_no",
    "international_id",
    "meaning",
    "status",
]

TRACK_COLUMNS = [
    "typhoon_id",
    "point_id",
    "time_bj",
    "time_iso",
    "grade",
    "lon",
    "lat",
    "pressure_hpa",
    "max_wind_ms",
    "move_dir",
    "move_speed_kmh",
    "wind_radii_json",
    "issue_time",
    "issue_time_text",
]

FORECAST_COLUMNS = [
    "typhoon_id",
    "point_id",
    "base_time_bj",
    "org",
    "lead_hour",
    "forecast_lon",
    "forecast_lat",
    "forecast_pressure_hpa",
    "forecast_max_wind_ms",
    "forecast_grade",
]


def fetch_text(path: str) -> str:
    url = f"{BASE_URL}{path}"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Referer": "http://typhoon.nmc.cn/web.html"})

    # The campus/Windows proxy environment may break this specific HTTP site.
    # Disable inherited proxy settings for reproducible crawling.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=20) as response:
            return response.read().decode("utf-8", errors="replace")
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def parse_jsonp(text: str) -> Any:
    text = text.strip()
    match = re.match(r"^[\w_.$]+\((.*)\)\s*;?\s*$", text, flags=re.S)
    payload = match.group(1).strip() if match else text

    # Some endpoints return callback(( {...} )), so remove extra parentheses.
    while payload.startswith("(") and payload.endswith(")"):
        payload = payload[1:-1].strip()

    return json.loads(payload)


def fetch_json(path: str) -> Any:
    return parse_jsonp(fetch_text(path))


def safe_name(value: str) -> str:
    value = value.strip() or "nameless"
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", value)


def parse_time_bj(value: Any) -> tuple[str, str]:
    if not value:
        return "", ""
    text = str(value)
    try:
        dt = datetime.strptime(text, "%Y%m%d%H%M")
        return text, dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return text, ""


def list_typhoons(year: str) -> list[list[Any]]:
    data = fetch_json(f"/typhoon/jsons/list_{year}")
    return data.get("typhoonList", [])


def find_typhoons(typhoons: list[list[Any]], name: str | None, active_only: bool) -> list[list[Any]]:
    rows = typhoons
    if active_only:
        rows = [row for row in rows if len(row) > 7 and row[7] == "start"]
    if name:
        key = name.lower()
        rows = [
            row
            for row in rows
            if key in str(row[0]).lower()
            or key in str(row[1]).lower()
            or key in str(row[2]).lower()
            or key in str(row[3]).lower()
            or key in str(row[4]).lower()
        ]
    return rows


def select_typhoons(typhoons: list[list[Any]], names: list[str] | None, active_only: bool) -> list[list[Any]]:
    if not names:
        return find_typhoons(typhoons, None, active_only)

    selected: list[list[Any]] = []
    seen_ids: set[str] = set()
    for name in names:
        for row in find_typhoons(typhoons, name, active_only):
            typhoon_id = str(row[0]) if row else ""
            if typhoon_id and typhoon_id not in seen_ids:
                selected.append(row)
                seen_ids.add(typhoon_id)
    return selected


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def normalize_list(rows: list[list[Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "typhoon_id": row[0] if len(row) > 0 else "",
                "en_name": row[1] if len(row) > 1 else "",
                "cn_name": row[2] if len(row) > 2 else "",
                "code": row[3] if len(row) > 3 else "",
                "display_no": row[4] if len(row) > 4 else "",
                "international_id": row[5] if len(row) > 5 else "",
                "meaning": row[6] if len(row) > 6 else "",
                "status": row[7] if len(row) > 7 else "",
            }
        )
    return normalized


def normalize_track(typhoon: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    typhoon_id = typhoon[0]
    points = typhoon[8] if len(typhoon) > 8 and isinstance(typhoon[8], list) else []
    track_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []

    for point in points:
        time_bj, time_iso = parse_time_bj(point[1] if len(point) > 1 else "")
        issue = point[12] if len(point) > 12 and isinstance(point[12], list) else []
        issue_time = issue[0] if len(issue) > 0 else ""
        issue_text = issue[1] if len(issue) > 1 else ""
        point_id = point[0] if len(point) > 0 else ""

        track_rows.append(
            {
                "typhoon_id": typhoon_id,
                "point_id": point_id,
                "time_bj": time_bj,
                "time_iso": time_iso,
                "grade": point[3] if len(point) > 3 else "",
                "lon": point[4] if len(point) > 4 else "",
                "lat": point[5] if len(point) > 5 else "",
                "pressure_hpa": point[6] if len(point) > 6 else "",
                "max_wind_ms": point[7] if len(point) > 7 else "",
                "move_dir": point[8] if len(point) > 8 else "",
                "move_speed_kmh": point[9] if len(point) > 9 else "",
                "wind_radii_json": json.dumps(point[10] if len(point) > 10 else [], ensure_ascii=False),
                "issue_time": issue_time,
                "issue_time_text": issue_text,
            }
        )

        forecasts = point[11] if len(point) > 11 and isinstance(point[11], dict) else {}
        for org, org_rows in forecasts.items():
            for forecast in org_rows:
                forecast_rows.append(
                    {
                        "typhoon_id": typhoon_id,
                        "point_id": point_id,
                        "base_time_bj": time_bj,
                        "org": org,
                        "lead_hour": forecast[0] if len(forecast) > 0 else "",
                        "forecast_lon": forecast[2] if len(forecast) > 2 else "",
                        "forecast_lat": forecast[3] if len(forecast) > 3 else "",
                        "forecast_pressure_hpa": forecast[4] if len(forecast) > 4 else "",
                        "forecast_max_wind_ms": forecast[5] if len(forecast) > 5 else "",
                        "forecast_grade": forecast[7] if len(forecast) > 7 else "",
                    }
                )

    return track_rows, forecast_rows


def crawl(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    year = args.year
    if year == "latest":
        year = str(fetch_json("/typhoon/jsons/maxYear")["year"])

    typhoon_list = list_typhoons(year)
    if args.save_list:
        write_csv(out_dir / f"typhoons_{year}.csv", TYPHOON_LIST_COLUMNS, normalize_list(typhoon_list))

    names = args.names or ([args.name] if args.name else None)
    selected = select_typhoons(typhoon_list, names, args.active)
    if not selected:
        raise SystemExit(f"No typhoon matched: year={year}, names={names!r}, active={args.active}")

    for row in selected:
        typhoon_id = row[0]
        display_name = safe_name(str(row[2] or row[1] or typhoon_id))
        data = fetch_json(f"/typhoon/jsons/view_{typhoon_id}")
        typhoon = data["typhoon"]

        if args.save_raw:
            raw_path = out_dir / f"raw_view_{typhoon_id}_{display_name}.json"
            raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        track_rows, forecast_rows = normalize_track(typhoon)
        write_csv(out_dir / f"track_{typhoon_id}_{display_name}.csv", TRACK_COLUMNS, track_rows)
        if args.save_forecast:
            write_csv(out_dir / f"forecast_{typhoon_id}_{display_name}.csv", FORECAST_COLUMNS, forecast_rows)

        latest = track_rows[-1] if track_rows else {}
        print(
            f"saved {display_name} ({typhoon_id}): "
            f"{len(track_rows)} observed points, {len(forecast_rows)} forecast points, "
            f"latest={latest.get('time_bj', '')} lon={latest.get('lon', '')} lat={latest.get('lat', '')}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crawl NMC real-time typhoon path data.")
    parser.add_argument("--year", default="latest", help="Year to crawl, e.g. 2026. Default: latest")
    parser.add_argument("--name", default=None, help="Typhoon id/name/no keyword, e.g. BAVI or 巴威")
    parser.add_argument("--names", nargs="+", default=None, help="Crawl multiple typhoon keywords at once")
    parser.add_argument("--active", action="store_true", help="Only crawl active typhoons")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--save-list", action="store_true", help="Also save the yearly typhoon list CSV")
    parser.add_argument("--save-raw", action="store_true", help="Also save raw NMC JSON")
    parser.add_argument("--save-forecast", action="store_true", help="Also save agency forecast-path CSV")
    return parser


if __name__ == "__main__":
    try:
        crawl(build_parser().parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
