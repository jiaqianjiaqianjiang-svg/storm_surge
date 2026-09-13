#!/usr/bin/env python
"""Download Wenzhou open hydrology/tide datasets.

Main target found for tide observations:
  - iid 6348: Wenzhou hydrological tide level information
  - api iid 6349
  - fields in downloaded CSV: merged_station_code, station_code, time, tide_level
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
from runtime_paths import WENZHOU_OUTPUT_DIR

DEFAULT_OUTPUT_DIR = WENZHOU_OUTPUT_DIR

BASE_URL = "https://data.wenzhou.gov.cn/jdop_front"
PUBLIC_PAGE = f"{BASE_URL}/channal/data_public.do?deptId=330302ZF100210&domainId="
LIST_URL = f"{BASE_URL}/channal/datapubliclist.do"
DOWNLOAD_URL = f"{BASE_URL}/resource/data/download.do"

WATER_BUREAU_DEPT_ID = "330302ZF100210"
TIDE_IID = "6348"


def make_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.verify = False
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": PUBLIC_PAGE,
        }
    )
    session.get(PUBLIC_PAGE, timeout=30)
    return session


def list_water_bureau_datasets(session: requests.Session, output_dir: Path) -> Path:
    params = {
        "pageNumber": "1",
        "pageSize": "20",
        "type": "",
        "domainId": "0",
        "deptId": WATER_BUREAU_DEPT_ID,
        "regionId": "",
        "keyword": "",
        "content": "",
        "searchString": "",
        "orderDefault": "",
        "isDownload": "1",
        "dimensionId": "0",
        "openState": "",
    }
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }
    response = session.post(LIST_URL, data=params, headers=headers, timeout=60)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    html = payload.get("data", "")

    rows: list[dict[str, str]] = []
    blocks = re.split(r'<div class="search_result">', html)
    for block in blocks[1:]:
        title_match = re.search(r'<a href="([^"]*detail/data\.do\?iid=(\d+)[^"]*)"[^>]*>(.*?)</a>', block, re.S)
        fields_match = re.search(r"<p class='search_result_left_stit'>(.*?)</p>", block, re.S)
        api_match = re.search(r"detail/api\.do\?iid=(\d+)", block)
        update_match = re.search(r"更新时间：\s*</span>\s*([0-9-]+)", block)
        if not title_match:
            continue
        title = re.sub(r"<.*?>", "", title_match.group(3)).strip()
        fields = re.sub(r"<.*?>", "", fields_match.group(1)).strip() if fields_match else ""
        rows.append(
            {
                "title": title,
                "data_iid": title_match.group(2),
                "api_iid": api_match.group(1) if api_match else "",
                "fields": fields,
                "update_date": update_match.group(1) if update_match else "",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "wenzhou_water_bureau_datasets.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "data_iid", "api_iid", "fields", "update_date"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {out_path} ({len(rows)} datasets)")
    for row in rows:
        print(f"{row['data_iid']}\tapi={row['api_iid']}\t{row['title']}\t{row['fields']}")
    return out_path


def download_dataset(session: requests.Session, iid: str, file_type: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    response = session.get(DOWNLOAD_URL, params={"fileType": file_type, "iid": iid}, timeout=180)
    response.raise_for_status()
    out_path = output_dir / f"wenzhou_iid_{iid}_{file_type}_download"
    response_path = out_path.with_suffix(".zip") if response.content[:2] == b"PK" else out_path
    response_path.write_bytes(response.content)
    print(f"saved: {response_path} ({len(response.content)} bytes)")
    if response.content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            print(f"zip members: {len(zf.namelist())}")
            print(", ".join(zf.namelist()[:10]) + (" ..." if len(zf.namelist()) > 10 else ""))
    return response_path


def summarize_tide_zip(zip_path: Path, output_dir: Path) -> Path:
    station = defaultdict(lambda: {"rows": 0, "start": None, "end": None, "min_tide": None, "max_tide": None})
    total_rows = 0
    samples: list[list[str]] = []

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".csv"):
                continue
            text = zf.read(member).decode("gbk", errors="replace")
            reader = csv.reader(io.StringIO(text))
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                code = row[1].strip()
                timestamp = row[2].strip()
                tide_text = row[3].strip()
                if not code or not timestamp:
                    continue
                total_rows += 1
                if len(samples) < 8:
                    samples.append([row[0].strip(), code, timestamp, tide_text])
                entry = station[code]
                entry["rows"] += 1
                entry["start"] = timestamp if entry["start"] is None or timestamp < entry["start"] else entry["start"]
                entry["end"] = timestamp if entry["end"] is None or timestamp > entry["end"] else entry["end"]
                try:
                    tide = float(tide_text)
                except ValueError:
                    continue
                entry["min_tide"] = tide if entry["min_tide"] is None or tide < entry["min_tide"] else entry["min_tide"]
                entry["max_tide"] = tide if entry["max_tide"] is None or tide > entry["max_tide"] else entry["max_tide"]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "wenzhou_tide_6348_station_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["station_code", "rows", "start_time", "end_time", "min_tide", "max_tide"])
        for code, entry in sorted(station.items(), key=lambda item: item[0]):
            writer.writerow(
                [
                    code,
                    entry["rows"],
                    entry["start"],
                    entry["end"],
                    entry["min_tide"],
                    entry["max_tide"],
                ]
            )

    print(f"total rows: {total_rows}")
    print(f"station count: {len(station)}")
    if station:
        print(f"time range: {min(v['start'] for v in station.values())} to {max(v['end'] for v in station.values())}")
    print(f"saved: {summary_path}")
    for sample in samples:
        print("sample:", sample)
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Wenzhou hydrology/tide open data.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--list", action="store_true", help="List Wenzhou Water Bureau datasets.")
    parser.add_argument("--download-iid", default=TIDE_IID, help="Dataset iid to download. Default: 6348 tide levels.")
    parser.add_argument("--file-type", default="csv", choices=["csv", "json", "xls", "xml", "rdf"])
    parser.add_argument("--no-download", action="store_true", help="Do not download a dataset.")
    parser.add_argument("--summarize", action="store_true", help="Summarize downloaded tide CSV zip.")
    parser.add_argument("--zip-path", help="Existing tide CSV zip to summarize.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    session = make_session()

    if args.list:
        list_water_bureau_datasets(session, output_dir)

    zip_path = Path(args.zip_path) if args.zip_path else None
    if not args.no_download:
        zip_path = download_dataset(session, args.download_iid, args.file_type, output_dir)

    if args.summarize:
        if not zip_path:
            print("--summarize needs --zip-path or a downloaded zip.", file=sys.stderr)
            return 2
        summarize_tide_zip(zip_path, output_dir)

    if not args.list and args.no_download and not args.summarize:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
