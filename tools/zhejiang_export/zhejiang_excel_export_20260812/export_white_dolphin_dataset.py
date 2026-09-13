import csv
import json
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None


DB = Path(r"E:\AAAqian\storm_surge_runtime_data\zhejiang_water_data\zhejiang_water_data.db")
MAPPING_CSV = Path(r"E:\AAAqian\storm_surge_runtime_data\zhejiang_water_data\exports\station_type_mapping.csv")
OUT_DIR = Path(r"E:\AAAqian\storm_surge_runtime_data\zhejiang_water_data\exports\white_dolphin_20260808")
START = "2026-08-08"
VALID_TYPES = {"河道", "水库", "堰闸", "潮汐"}
WATER_COLUMNS = ["地级市", "市县", "站名", "站点类型", "上报时间", "水位_m", "抓取时间"]
RAIN_COLUMNS = ["地级市", "市县", "站名", "统计开始时间", "统计结束时间", "雨量_mm", "抓取时间"]


def norm(value):
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def load_station_type_mapping():
    mapping = {}
    if not MAPPING_CSV.exists():
        return mapping
    with MAPPING_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stype = norm(row.get("station_type"))
            if stype not in VALID_TYPES:
                continue
            key = (norm(row.get("city")), norm(row.get("city_county")), norm(row.get("station_name")))
            mapping[key] = stype
    return mapping


def export_water(conn, mapping):
    water_csv = OUT_DIR / "white_dolphin_water_level.csv"
    water_sample = OUT_DIR / "white_dolphin_water_level_sample_5000.csv"
    unknown_csv = OUT_DIR / "white_dolphin_water_level_unknown_type.csv"
    type_paths = {
        "河道": OUT_DIR / "by_station_type" / "white_dolphin_water_level_river.csv",
        "水库": OUT_DIR / "by_station_type" / "white_dolphin_water_level_reservoir.csv",
        "堰闸": OUT_DIR / "by_station_type" / "white_dolphin_water_level_gate.csv",
        "潮汐": OUT_DIR / "by_station_type" / "white_dolphin_water_level_tide.csv",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "by_station_type").mkdir(parents=True, exist_ok=True)

    files = {"all": water_csv.open("w", encoding="utf-8-sig", newline="")}
    files["sample"] = water_sample.open("w", encoding="utf-8-sig", newline="")
    files["unknown"] = unknown_csv.open("w", encoding="utf-8-sig", newline="")
    for stype, path in type_paths.items():
        files[stype] = path.open("w", encoding="utf-8-sig", newline="")

    writers = {name: csv.writer(handle) for name, handle in files.items()}
    for writer in writers.values():
        writer.writerow(WATER_COLUMNS)

    stats = {
        "count": 0,
        "sample_count": 0,
        "unknown_count": 0,
        "min_reported_at": None,
        "max_reported_at": None,
        "min_crawl_time": None,
        "max_crawl_time": None,
        "cities": Counter(),
        "types": Counter(),
        "stations": set(),
        "dates": Counter(),
    }

    sql = """
        SELECT city, city_county, station_name, station_type, reported_at, water_level, crawl_time
        FROM water_level_records_v2
        WHERE crawl_time >= ?
        ORDER BY reported_at, city, city_county, station_name
    """
    try:
        for city, county, station, stored_type, reported_at, water_level, crawl_time in conn.execute(sql, (START,)):
            key = (norm(city), norm(county), norm(station))
            station_type = mapping.get(key)
            if station_type not in VALID_TYPES:
                station_type = "未知"
            row = [norm(city), norm(county), norm(station), station_type, norm(reported_at), water_level, norm(crawl_time)]

            writers["all"].writerow(row)
            if station_type in VALID_TYPES:
                writers[station_type].writerow(row)
            else:
                writers["unknown"].writerow(row)
                stats["unknown_count"] += 1
            if stats["sample_count"] < 5000:
                writers["sample"].writerow(row)
                stats["sample_count"] += 1

            stats["count"] += 1
            stats["cities"][norm(city)] += 1
            stats["types"][station_type] += 1
            stats["stations"].add(key)
            if reported_at:
                stats["dates"][norm(reported_at)[:10]] += 1
                stats["min_reported_at"] = min(filter(None, [stats["min_reported_at"], norm(reported_at)]))
                stats["max_reported_at"] = max(filter(None, [stats["max_reported_at"], norm(reported_at)]))
            if crawl_time:
                stats["min_crawl_time"] = min(filter(None, [stats["min_crawl_time"], norm(crawl_time)]))
                stats["max_crawl_time"] = max(filter(None, [stats["max_crawl_time"], norm(crawl_time)]))
    finally:
        for handle in files.values():
            handle.close()

    stats["station_count"] = len(stats["stations"])
    stats["files"] = {"all": str(water_csv), "sample": str(water_sample), "unknown": str(unknown_csv)}
    stats["type_files"] = {k: str(v) for k, v in type_paths.items()}
    return stats


def export_rain(conn):
    rain_csv = OUT_DIR / "white_dolphin_rainfall.csv"
    rain_sample = OUT_DIR / "white_dolphin_rainfall_sample_5000.csv"
    with rain_csv.open("w", encoding="utf-8-sig", newline="") as all_file, rain_sample.open("w", encoding="utf-8-sig", newline="") as sample_file:
        all_writer = csv.writer(all_file)
        sample_writer = csv.writer(sample_file)
        all_writer.writerow(RAIN_COLUMNS)
        sample_writer.writerow(RAIN_COLUMNS)
        stats = {
            "count": 0,
            "sample_count": 0,
            "min_period_start": None,
            "max_period_end": None,
            "min_crawl_time": None,
            "max_crawl_time": None,
            "cities": Counter(),
            "stations": set(),
            "dates": Counter(),
        }
        sql = """
            SELECT city, city_county, station_name, period_start, period_end, rainfall, crawl_time
            FROM rainfall_records_v1
            WHERE crawl_time >= ?
            ORDER BY period_start, city, city_county, station_name
        """
        for city, county, station, period_start, period_end, rainfall, crawl_time in conn.execute(sql, (START,)):
            row = [norm(city), norm(county), norm(station), norm(period_start), norm(period_end), rainfall, norm(crawl_time)]
            all_writer.writerow(row)
            if stats["sample_count"] < 5000:
                sample_writer.writerow(row)
                stats["sample_count"] += 1
            stats["count"] += 1
            stats["cities"][norm(city)] += 1
            stats["stations"].add((norm(city), norm(county), norm(station)))
            if period_start:
                stats["dates"][norm(period_start)[:10]] += 1
                stats["min_period_start"] = min(filter(None, [stats["min_period_start"], norm(period_start)]))
            if period_end:
                stats["max_period_end"] = max(filter(None, [stats["max_period_end"], norm(period_end)]))
            if crawl_time:
                stats["min_crawl_time"] = min(filter(None, [stats["min_crawl_time"], norm(crawl_time)]))
                stats["max_crawl_time"] = max(filter(None, [stats["max_crawl_time"], norm(crawl_time)]))
    stats["station_count"] = len(stats["stations"])
    stats["files"] = {"all": str(rain_csv), "sample": str(rain_sample)}
    return stats


def rows_from_counter(counter, key_name, value_name):
    return [{key_name: k, value_name: v} for k, v in sorted(counter.items())]


def write_excel(summary):
    if pd is None:
        return None
    xlsx = OUT_DIR / "白海豚期间水位雨量数据汇总.xlsx"
    overview = [
        ["事件名称", "白海豚"],
        ["筛选口径", "crawl_time >= 2026-08-08"],
        ["水位记录数", summary["water"]["count"]],
        ["水位站点数", summary["water"]["station_count"]],
        ["水位上报时间范围", f'{summary["water"]["min_reported_at"]} 至 {summary["water"]["max_reported_at"]}'],
        ["水位抓取时间范围", f'{summary["water"]["min_crawl_time"]} 至 {summary["water"]["max_crawl_time"]}'],
        ["雨量记录数", summary["rain"]["count"]],
        ["雨量站点数", summary["rain"]["station_count"]],
        ["雨量统计时间范围", f'{summary["rain"]["min_period_start"]} 至 {summary["rain"]["max_period_end"]}'],
        ["雨量抓取时间范围", f'{summary["rain"]["min_crawl_time"]} 至 {summary["rain"]["max_crawl_time"]}'],
        ["完整水位CSV", summary["water"]["files"]["all"]],
        ["完整雨量CSV", summary["rain"]["files"]["all"]],
    ]
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame(overview, columns=["项目", "内容"]).to_excel(writer, sheet_name="概览", index=False)
        pd.DataFrame(rows_from_counter(summary["water"]["types"], "站点类型", "水位记录数")).to_excel(writer, sheet_name="水位类型统计", index=False)
        pd.DataFrame(rows_from_counter(summary["water"]["cities"], "地级市", "水位记录数")).to_excel(writer, sheet_name="水位地市统计", index=False)
        pd.DataFrame(rows_from_counter(summary["rain"]["cities"], "地级市", "雨量记录数")).to_excel(writer, sheet_name="雨量地市统计", index=False)
        pd.DataFrame(rows_from_counter(summary["water"]["dates"], "日期", "水位记录数")).to_excel(writer, sheet_name="水位日期统计", index=False)
        pd.DataFrame(rows_from_counter(summary["rain"]["dates"], "日期", "雨量记录数")).to_excel(writer, sheet_name="雨量日期统计", index=False)
        pd.read_csv(summary["water"]["files"]["sample"], encoding="utf-8-sig").to_excel(writer, sheet_name="水位样本5000", index=False)
        pd.read_csv(summary["rain"]["files"]["sample"], encoding="utf-8-sig").to_excel(writer, sheet_name="雨量样本5000", index=False)

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for col in sheet.columns:
                max_len = min(max(len(str(cell.value)) if cell.value is not None else 0 for cell in col) + 2, 48)
                sheet.column_dimensions[col[0].column_letter].width = max(max_len, 10)
    return str(xlsx)


def write_report(summary):
    report = OUT_DIR / "白海豚期间爬虫数据简报.md"
    lines = [
        "# 白海豚期间浙江水位雨量爬虫数据简报",
        "",
        f"本次整理只保留 `crawl_time >= {START}` 的数据，不包含 7 月历史采集数据。",
        "",
        "## 数据范围",
        "",
        f"- 水位记录：{summary['water']['count']:,} 条，覆盖 {summary['water']['station_count']:,} 个站点。",
        f"- 水位上报时间：{summary['water']['min_reported_at']} 至 {summary['water']['max_reported_at']}。",
        f"- 水位抓取时间：{summary['water']['min_crawl_time']} 至 {summary['water']['max_crawl_time']}。",
        f"- 雨量记录：{summary['rain']['count']:,} 条，覆盖 {summary['rain']['station_count']:,} 个站点。",
        f"- 雨量统计时间：{summary['rain']['min_period_start']} 至 {summary['rain']['max_period_end']}。",
        f"- 雨量抓取时间：{summary['rain']['min_crawl_time']} 至 {summary['rain']['max_crawl_time']}。",
        "",
        "## 覆盖地区",
        "",
        "水位数据覆盖浙江省 11 个地级市：杭州市、宁波市、温州市、嘉兴市、湖州市、绍兴市、金华市、衢州市、舟山市、台州市、丽水市。",
        "雨量数据同样按全省范围抓取，导出时保留地级市、市县、站名、统计时段和雨量值。",
        "",
        "## 水位站点类型",
        "",
    ]
    for stype, count in sorted(summary["water"]["types"].items()):
        lines.append(f"- {stype}：{count:,} 条")
    lines.extend([
        "",
        "## 输出文件",
        "",
        f"- 完整水位 CSV：`{summary['water']['files']['all']}`",
        f"- 完整雨量 CSV：`{summary['rain']['files']['all']}`",
        f"- Excel 汇总表：`{summary['excel']}`",
        "",
        "说明：完整水位数据超过 Excel 单个工作表行数上限，因此完整明细以 CSV 保存；Excel 中放入汇总统计和前 5000 条样本，便于快速查看。",
        "",
    ])
    report.write_text("\n".join(lines), encoding="utf-8")
    return str(report)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = load_station_type_mapping()
    conn = sqlite3.connect(DB)
    try:
        summary = {"created_at": datetime.now().isoformat(timespec="seconds")}
        summary["mapping_count"] = len(mapping)
        summary["water"] = export_water(conn, mapping)
        summary["rain"] = export_rain(conn)
    finally:
        conn.close()
    summary["excel"] = write_excel(summary)
    summary["report"] = write_report(summary)
    serializable = json.loads(json.dumps(summary, ensure_ascii=False, default=lambda x: dict(x) if isinstance(x, Counter) else str(x)))
    (OUT_DIR / "white_dolphin_export_summary.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(serializable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
