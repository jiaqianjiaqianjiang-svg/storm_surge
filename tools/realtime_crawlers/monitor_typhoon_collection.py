#!/usr/bin/env python
"""Read-only terminal dashboard for one typhoon collection directory."""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def file_age_text(path: Path) -> str:
    if not path.exists():
        return "尚未生成"
    seconds = max(0, int(time.time() - path.stat().st_mtime))
    if seconds < 60:
        return f"{seconds} 秒前"
    if seconds < 3600:
        return f"{seconds // 60} 分钟前"
    return f"{seconds / 3600:.1f} 小时前"


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def latest_matching(root: Path, pattern: str) -> Path | None:
    files = list(root.glob(pattern)) if root.exists() else []
    return max(files, key=lambda item: item.stat().st_mtime) if files else None


def sqlite_table_stats(db_path: Path, table: str, time_column: str) -> tuple[int, str]:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=3) as connection:
        connection.execute("PRAGMA query_only = ON")
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            return 0, "表尚未创建"
        count = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        latest_row = connection.execute(
            f"SELECT {time_column} FROM {table} ORDER BY id DESC LIMIT 1"
        ).fetchone()
        latest = str(latest_row[0]) if latest_row and latest_row[0] else "暂无"
        return count, latest


def tail_lines(path: Path, limit: int = 12) -> list[str]:
    if not path.exists():
        return ["日志尚未生成"]
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip() for line in handle.readlines()[-limit:]]


def render(event_dir: Path, log_lines: int) -> None:
    water_dir = event_dir / "zhejiang_water_data"
    db_path = water_dir / "zhejiang_water_data.db"
    log_path = water_dir / "logs" / "zhejiang_crawler.log"
    track_path = latest_matching(event_dir, "track_*.csv")
    forecast_path = latest_matching(event_dir, "forecast_*.csv")
    gfs_files = list((event_dir / "gfs").rglob("*.grib2")) if (event_dir / "gfs").exists() else []
    gfs_bytes = sum(path.stat().st_size for path in gfs_files)
    latest_gfs = max(gfs_files, key=lambda item: item.stat().st_mtime) if gfs_files else None

    print("=" * 78)
    print(f"{event_dir.name} 数据采集监控（只读）")
    print(f"刷新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"事件目录：{event_dir}")
    print("=" * 78)

    if db_path.exists():
        try:
            water_count, latest_water = sqlite_table_stats(
                db_path, "water_level_records_v2", "crawl_time"
            )
            rain_count, latest_rain = sqlite_table_stats(
                db_path, "rainfall_records_v1", "crawl_time"
            )
            print(f"水位数据库：{human_size(db_path.stat().st_size)}，最近改动 {file_age_text(db_path)}")
            print(f"水位记录数：{water_count:,}，最后抓取时间：{latest_water}")
            print(f"雨量记录数：{rain_count:,}，最后抓取时间：{latest_rain}")
        except (sqlite3.Error, OSError) as exc:
            print(f"水位数据库：暂时无法读取（可能正在写入）：{exc}")
    else:
        print("水位数据库：尚未生成")

    print("-" * 78)
    if track_path:
        print(
            f"台风实况路径：{csv_row_count(track_path):,} 点，"
            f"最近更新 {file_age_text(track_path)}，文件 {track_path.name}"
        )
    else:
        print("台风实况路径：尚未生成")
    if forecast_path:
        print(
            f"台风预报路径：{csv_row_count(forecast_path):,} 点，"
            f"最近更新 {file_age_text(forecast_path)}，文件 {forecast_path.name}"
        )
    else:
        print("台风预报路径：尚未生成")
    print(
        f"GFS 文件：{len(gfs_files):,} 个，共 {human_size(gfs_bytes)}，"
        f"最近更新 {file_age_text(latest_gfs) if latest_gfs else '尚未生成'}"
    )

    print("-" * 78)
    print(f"水位日志最后 {log_lines} 行：")
    for line in tail_lines(log_path, log_lines):
        print(line)
    print("-" * 78)
    print("每隔一段时间自动刷新；按 Ctrl+C 退出监控，不会停止两个爬虫。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor one typhoon collection directory without modifying it.")
    parser.add_argument("--event-dir", type=Path, required=True, help="Typhoon event output directory")
    parser.add_argument("--refresh-seconds", type=float, default=30, help="Dashboard refresh interval")
    parser.add_argument("--log-lines", type=int, default=12, help="Number of water crawler log lines to display")
    parser.add_argument("--once", action="store_true", help="Render once and exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    event_dir = args.event_dir.resolve()
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        render(event_dir, max(1, args.log_lines))
        if args.once:
            return 0
        try:
            time.sleep(max(5.0, args.refresh_seconds))
        except KeyboardInterrupt:
            print("\n监控已停止，两个爬虫不会受到影响。")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
