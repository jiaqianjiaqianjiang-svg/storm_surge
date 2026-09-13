"""Crawl NMEFC typhoon storm-surge station forecast data.

Source page:
https://www.nmefc.cn/stormSurgeViews/typhoon

The Vue frontend uses API endpoints under https://www.nmefc.cn/api, including:
- /data/init/typhoon
- /cms/dimsite/typhoon
- /cms/dimsite/tree/typhoon
- /data/site/info
- /data/typhoon/statistics
- /data/typhoon/path

The station data are forecast products, not raw real-time tide-gauge
observations. They contain astronomical tide, surge, and total water level
series for forecast scenarios.
"""

from __future__ import annotations

import argparse
import csv
import json
import ssl
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener

from runtime_paths import NMEFC_OUTPUT_DIR


BASE_URL = "https://www.nmefc.cn/api"
REFERER = "https://www.nmefc.cn/stormSurgeViews/typhoon"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)

SITE_COLUMNS = [
    "id",
    "province",
    "siteName",
    "siteCode",
    "type",
    "siteArea",
    "lat",
    "lon",
    "blue",
    "yellow",
    "orange",
    "red",
    "sort",
    "version",
]

SITE_INFO_COLUMNS = [
    "province",
    "siteName",
    "siteCode",
    "siteArea",
    "lat",
    "lon",
    "pathId",
    "tyCode",
    "levelColor",
    "maxSurge",
    "maxSurgeTime",
    "minSurge",
    "minSurgeTime",
    "maxWl",
    "maxWlTime",
    "minWl",
    "minWlTime",
    "cm50",
    "cm100",
    "cm150",
    "cm200",
    "cm250",
    "cm300",
    "blue",
    "yellow",
    "orange",
    "red",
]

FORECAST_PATH_COLUMNS = [
    "pathId",
    "pointId",
    "tyCode",
    "initialTime",
    "forecastTime",
    "forecastInterval",
    "lat",
    "lon",
    "bp",
    "rd",
    "sid",
]

PRODUCT_COLUMNS = [
    "id",
    "typeCode",
    "type",
    "element",
    "bak1",
    "bizDate",
    "updateDate",
    "version",
    "url",
]

TIMESERIES_COLUMNS = [
    "site",
    "tyCode",
    "pathId",
    "runTime",
    "initialTime",
    "forecastTime",
    "lead_index",
    "tide_cm",
    "surge_cm",
    "water_level_cm",
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def opener():
    # The site can fail on some Windows TLS/cert paths. The browser page itself
    # is public, so this uses an unverified context for reproducible scraping.
    return build_opener(
        ProxyHandler({}),
        HTTPSHandler(context=ssl._create_unverified_context()),
    )


def get_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = f"?{urlencode(params)}" if params else ""
    url = f"{BASE_URL}{path}{query}"
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
            "Referer": REFERER,
        },
    )
    try:
        with opener().open(request, timeout=60) as response:
            text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"NMEFC API HTTP {exc.code}: {url}\n{body}") from exc
    except URLError as exc:
        raise RuntimeError(f"NMEFC API network error: {url}\n{exc}") from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"NMEFC API did not return JSON: {url}\n{text[:1000]}") from exc


def require_obj(payload: dict[str, Any], label: str) -> Any:
    if not payload.get("success", False):
        raise RuntimeError(f"{label} failed: {payload}")
    return payload.get("obj")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_tree(tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in tree:
        for child in group.get("children", []):
            rows.append(child)
    return rows


def filter_sites(
    sites: list[dict[str, Any]],
    area: str | None,
    province: str | None,
    site_names: list[str],
) -> list[dict[str, Any]]:
    rows = sites
    if area:
        rows = [row for row in rows if str(row.get("siteArea", "")) == str(area)]
    if province:
        rows = [row for row in rows if province in str(row.get("province", ""))]
    if site_names:
        keys = set(site_names)
        rows = [row for row in rows if row.get("siteName") in keys or row.get("siteCode") in keys]
    return rows


def path_id(number: int) -> str:
    return f"p{number:03d}"


def forecast_time(initial_time: str, lead_index: int) -> str:
    try:
        start = datetime.strptime(initial_time, "%Y-%m-%d %H:%M:%S")
        return (start + timedelta(hours=lead_index)).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ""


def normalize_forecast_path(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in data:
        scenario_path_id = scenario.get("pathId", "")
        for point in scenario.get("pointData", []):
            rows.append({"pathId": scenario_path_id, **point})
    return rows


def normalize_site_info(site_info: dict[str, Any], ty_code: str, scenario_path_id: str) -> dict[str, Any]:
    return {"tyCode": ty_code, "pathId": scenario_path_id, **site_info}


def normalize_timeseries(statistics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in statistics.get("data", []):
        tides = scenario.get("tides", [])
        surges = scenario.get("surges", [])
        water_levels = scenario.get("wl", [])
        initial_time = scenario.get("initialTime", "")
        max_len = max(len(tides), len(surges), len(water_levels))
        for index in range(max_len):
            rows.append(
                {
                    "site": scenario.get("site", statistics.get("site", "")),
                    "tyCode": scenario.get("tyCode", statistics.get("tyCode", "")),
                    "pathId": scenario.get("pathId", ""),
                    "runTime": scenario.get("runTime", statistics.get("runTime", "")),
                    "initialTime": initial_time,
                    "forecastTime": forecast_time(initial_time, index),
                    "lead_index": index,
                    "tide_cm": tides[index] if index < len(tides) else "",
                    "surge_cm": surges[index] if index < len(surges) else "",
                    "water_level_cm": water_levels[index] if index < len(water_levels) else "",
                }
            )
    return rows


def crawl(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_payload = get_json("/data/init/typhoon")
    init_obj = require_obj(init_payload, "init")
    ty_code = args.ty_code or init_obj.get("tyCode")
    if not ty_code:
        raise SystemExit("No active typhoon storm-surge process found on NMEFC.")

    default_area = str(init_obj.get("area", "")) if init_obj.get("area") is not None else None
    selected_area = args.area if args.area is not None else (None if args.ty_code else default_area)
    scenario_path_id = path_id(args.path_number)

    write_json(out_dir / "nmefc_typhoon_init.json", init_payload)

    tree_payload = get_json("/cms/dimsite/tree/typhoon")
    site_tree = require_obj(tree_payload, "site tree")
    write_json(out_dir / "nmefc_typhoon_site_tree.json", tree_payload)
    all_sites = flatten_tree(site_tree)
    write_csv(out_dir / "nmefc_typhoon_sites_all.csv", SITE_COLUMNS, all_sites)

    selected_sites = filter_sites(all_sites, selected_area, args.province, args.site)
    if not selected_sites:
        raise SystemExit("No station matched the selected filters.")
    write_csv(out_dir / "nmefc_typhoon_sites_selected.csv", SITE_COLUMNS, selected_sites)

    forecast_payload = get_json("/data/typhoon/path", {"tyCode": ty_code, "pathId": scenario_path_id})
    forecast_obj = require_obj(forecast_payload, "forecast path")
    write_json(out_dir / f"nmefc_typhoon_forecast_path_{ty_code}_{scenario_path_id}.json", forecast_payload)
    write_csv(
        out_dir / f"nmefc_typhoon_forecast_path_{ty_code}_{scenario_path_id}.csv",
        FORECAST_PATH_COLUMNS,
        normalize_forecast_path(forecast_obj),
    )

    product_rows: list[dict[str, Any]] = []
    for element in args.product_element:
        product_payload = get_json("/cms/file/typhoon", {"element": element, "typeCode": ty_code})
        product_obj = require_obj(product_payload, f"product {element}")
        write_json(out_dir / f"nmefc_typhoon_products_{ty_code}_{element}.json", product_payload)
        product_rows.extend(product_obj or [])
    if product_rows:
        write_csv(out_dir / f"nmefc_typhoon_products_{ty_code}.csv", PRODUCT_COLUMNS, product_rows)

    site_info_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    max_sites = len(selected_sites) if args.all_sites else min(len(selected_sites), args.max_sites)

    for site in selected_sites[:max_sites]:
        site_name = site["siteName"]
        info_payload = get_json(
            "/data/site/info",
            {"site": site_name, "type": "typhoon", "pathId": scenario_path_id, "tyCode": ty_code},
        )
        info_obj = require_obj(info_payload, f"site info {site_name}")
        site_info_rows.append(normalize_site_info(info_obj, ty_code, scenario_path_id))

        if args.statistics or args.site:
            stat_payload = get_json("/data/typhoon/statistics", {"site": site_name, "tyCode": ty_code})
            stat_obj = require_obj(stat_payload, f"statistics {site_name}")
            write_json(out_dir / f"nmefc_typhoon_statistics_{ty_code}_{site_name}.json", stat_payload)
            timeseries_rows.extend(normalize_timeseries(stat_obj))

    write_csv(out_dir / f"nmefc_typhoon_site_info_{ty_code}_{scenario_path_id}.csv", SITE_INFO_COLUMNS, site_info_rows)
    if timeseries_rows:
        write_csv(out_dir / f"nmefc_typhoon_timeseries_{ty_code}.csv", TIMESERIES_COLUMNS, timeseries_rows)

    print(f"typhoon: {ty_code} {init_obj.get('tyName', '')}")
    print(f"selected stations: {len(selected_sites)}; crawled station summaries: {len(site_info_rows)}")
    if timeseries_rows:
        print(f"timeseries rows: {len(timeseries_rows)}")
    print(f"output: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crawl NMEFC typhoon storm-surge station forecast data.")
    parser.add_argument("--out", default=str(NMEFC_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--ty-code", default=None, help="Typhoon code, e.g. TY2610. Default: current NMEFC process")
    parser.add_argument("--area", default=None, help="NMEFC siteArea filter. Default: current process area")
    parser.add_argument("--province", default=None, help="Province name filter, e.g. 浙江 or 福建")
    parser.add_argument("--site", action="append", default=[], help="Station name/code. Can be repeated.")
    parser.add_argument("--path-number", type=int, default=1, help="Scenario path number, p001 by default")
    parser.add_argument("--max-sites", type=int, default=20, help="Station summaries to crawl unless --all-sites is set")
    parser.add_argument("--all-sites", action="store_true", help="Crawl summaries for all selected stations")
    parser.add_argument("--statistics", action="store_true", help="Also crawl tide/surge/water-level time series")
    parser.add_argument(
        "--product-element",
        action="append",
        default=["maxsurges"],
        help="Product file element to save from /cms/file/typhoon. Can be repeated.",
    )
    return parser


if __name__ == "__main__":
    crawl(build_parser().parse_args())
