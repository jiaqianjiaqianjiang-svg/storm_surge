"""Download tide/sea-state observations from Taiwan CWA Open Data.

Most CWA Open Data API endpoints require an Authorization token from:
https://opendata.cwa.gov.tw/

This crawler is intentionally dataset-id driven because CWA dataset IDs can
change by product. Use the CWA portal search terms such as:
- 潮位站
- 潮高
- 浮標站與潮位站觀測資料
- 海象監測

Example:
python real_time_data_crawler/cwa_tide_crawler.py --dataset-id DATASET_ID --authorization YOUR_TOKEN
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from runtime_paths import CWA_OUTPUT_DIR

import requests


CWA_DATASTORE_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/{dataset_id}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)

OBS_COLUMNS = [
    "StationID",
    "DateTime",
    "TideHeight",
    "TideLevel",
    "WaveHeight",
    "WaveDirection",
    "WaveDirectionDescription",
    "WavePeriod",
    "SeaTemperature",
    "Temperature",
    "StationPressure",
    "WindSpeed",
    "WindScale",
    "WindDirection",
    "WindDirectionDescription",
    "MaximumWindSpeed",
    "MaximumWindScale",
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def fetch_json(dataset_id: str, authorization: str, extra_params: dict[str, str]) -> dict[str, Any]:
    params = {"Authorization": authorization, **extra_params}
    url = f"{CWA_DATASTORE_URL.format(dataset_id=dataset_id)}?{urlencode(params)}"
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.get(
            CWA_DATASTORE_URL.format(dataset_id=dataset_id),
            params=params,
            timeout=90,
            verify=False,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        # Keep urllib fallback for environments without requests-compatible TLS.
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except (HTTPError, URLError) as fallback_exc:
            raise RuntimeError(f"CWA API network error: {exc}; fallback: {fallback_exc}") from fallback_exc
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"CWA API HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"CWA API network error: {exc}") from exc


def flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested CWA JSON into a single row-friendly dict."""
    row: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            row.update(flatten_json(item, next_prefix))
    elif isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            row[prefix] = json.dumps(value, ensure_ascii=False)
        else:
            for index, item in enumerate(value):
                next_prefix = f"{prefix}.{index}" if prefix else str(index)
                row.update(flatten_json(item, next_prefix))
    else:
        row[prefix] = value
    return row


def find_records(payload: dict[str, Any]) -> list[Any]:
    """Find the likely observation record list in common CWA response shapes."""
    records = payload.get("records")
    if isinstance(records, dict):
        for key in ("SeaSurfaceObs", "Station", "location", "locations", "records"):
            value = records.get(key)
            if isinstance(value, list):
                return value
        # Fallback: use the first list under records.
        for value in records.values():
            if isinstance(value, list):
                return value
    if isinstance(records, list):
        return records
    return []


def normalize_sea_surface_observations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("Records") or payload.get("records") or {}
    locations = records.get("SeaSurfaceObs", {}).get("Location", [])
    rows: list[dict[str, Any]] = []
    for location in locations:
        station = location.get("Station", {})
        station_id = station.get("StationID", "")
        times = location.get("StationObsTimes", {}).get("StationObsTime", [])
        for item in times:
            weather = item.get("WeatherElements", {})
            wind = weather.get("PrimaryAnemometer", {})
            rows.append(
                {
                    "StationID": station_id,
                    "DateTime": item.get("DateTime", ""),
                    "TideHeight": weather.get("TideHeight", ""),
                    "TideLevel": weather.get("TideLevel", ""),
                    "WaveHeight": weather.get("WaveHeight", ""),
                    "WaveDirection": weather.get("WaveDirection", ""),
                    "WaveDirectionDescription": weather.get("WaveDirectionDescription", ""),
                    "WavePeriod": weather.get("WavePeriod", ""),
                    "SeaTemperature": weather.get("SeaTemperature", ""),
                    "Temperature": weather.get("Temperature", ""),
                    "StationPressure": weather.get("StationPressure", ""),
                    "WindSpeed": wind.get("WindSpeed", ""),
                    "WindScale": wind.get("WindScale", ""),
                    "WindDirection": wind.get("WindDirection", ""),
                    "WindDirectionDescription": wind.get("WindDirectionDescription", ""),
                    "MaximumWindSpeed": wind.get("MaximumWindSpeed", ""),
                    "MaximumWindScale": wind.get("MaximumWindScale", ""),
                }
            )
    return rows


def write_outputs(payload: dict[str, Any], out_dir: Path, dataset_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"cwa_{dataset_id}_raw.json"
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    obs_rows = normalize_sea_surface_observations(payload)
    if obs_rows:
        obs_path = out_dir / f"cwa_{dataset_id}_observations.csv"
        with obs_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=OBS_COLUMNS)
            writer.writeheader()
            writer.writerows(obs_rows)
        print(f"saved raw: {raw_path}")
        print(f"saved observations: {obs_path}")
        print(f"observations: {len(obs_rows)}")
        return

    records = find_records(payload)
    if not records:
        print(f"saved raw only: {raw_path}")
        return

    rows = [flatten_json(record) for record in records]
    columns = sorted({key for row in rows for key in row.keys()})
    csv_path = out_dir / f"cwa_{dataset_id}_flattened.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved raw: {raw_path}")
    print(f"saved csv: {csv_path}")
    print(f"records: {len(rows)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Taiwan CWA tide/sea-state open data.")
    parser.add_argument("--dataset-id", required=True, help="CWA datastore id, found in the CWA Open Data portal")
    parser.add_argument(
        "--authorization",
        default=os.environ.get("CWA_AUTHORIZATION"),
        help="CWA API token. Can also be set as CWA_AUTHORIZATION.",
    )
    parser.add_argument("--out", default=str(CWA_OUTPUT_DIR), help="Output directory")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Extra query parameter in key=value form, e.g. --param StationID=46695",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.authorization:
        raise SystemExit("Missing CWA authorization token. Set --authorization or CWA_AUTHORIZATION.")

    extra_params: dict[str, str] = {}
    for item in args.param:
        if "=" not in item:
            raise SystemExit(f"Invalid --param {item!r}; expected key=value")
        key, value = item.split("=", 1)
        extra_params[key] = value

    payload = fetch_json(args.dataset_id, args.authorization, extra_params)
    write_outputs(payload, Path(args.out), args.dataset_id)


if __name__ == "__main__":
    main()
OBS_COLUMNS = [
    "StationID",
    "DateTime",
    "TideHeight",
    "TideLevel",
    "WaveHeight",
    "WaveDirection",
    "WaveDirectionDescription",
    "WavePeriod",
    "SeaTemperature",
    "Temperature",
    "StationPressure",
    "WindSpeed",
    "WindScale",
    "WindDirection",
    "WindDirectionDescription",
    "MaximumWindSpeed",
    "MaximumWindScale",
]
