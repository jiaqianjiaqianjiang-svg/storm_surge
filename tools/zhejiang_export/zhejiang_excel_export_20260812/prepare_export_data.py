import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path


OUT = Path(r"E:\AAAqian\code\storm_surge_clean\outputs\zhejiang_excel_export_20260812")
DB = Path(r"E:\AAAqian\storm_surge_runtime_data\zhejiang_water_data\zhejiang_water_data.db")
TYPHOON_DIR = Path(r"E:\AAAqian\storm_surge_runtime_data\typhoon_data\nmc")
FORMAL_EXPORT = Path(r"E:\AAAqian\storm_surge_runtime_data\zhejiang_water_data\exports\formal_export")


def write_csv(name, data):
    path = OUT / name
    if not data:
        path.write_text("", encoding="utf-8-sig")
        return str(path)
    cols = list(data[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(data)
    return str(path)


def read_csv(path, limit=None):
    if not path or not Path(path).exists():
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if limit is not None and i >= limit:
                break
            rows.append(dict(row))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    def one(sql, params=()):
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else {}

    def rows(sql, params=()):
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    water_overall = one(
        """
        SELECT COUNT(*) AS record_count,
               COUNT(DISTINCT city || '|' || city_county || '|' || station_name) AS station_count,
               MIN(reported_at) AS min_reported_at,
               MAX(reported_at) AS max_reported_at,
               MIN(crawl_time) AS min_crawl_time,
               MAX(crawl_time) AS max_crawl_time
        FROM water_level_records_v2
        """
    )
    rain_overall = one(
        """
        SELECT COUNT(*) AS record_count,
               COUNT(DISTINCT city || '|' || city_county || '|' || station_name) AS station_count,
               MIN(period_start) AS min_period_start,
               MAX(period_end) AS max_period_end,
               MIN(crawl_time) AS min_crawl_time,
               MAX(crawl_time) AS max_crawl_time
        FROM rainfall_records_v1
        """
    )

    latest_water_crawl = one("SELECT MAX(crawl_time) AS latest FROM water_level_records_v2").get("latest")
    latest_rain_crawl = one("SELECT MAX(crawl_time) AS latest FROM rainfall_records_v1").get("latest")
    latest_water_count = one("SELECT COUNT(*) AS n FROM water_level_records_v2 WHERE crawl_time = ?", (latest_water_crawl,)).get("n", 0)
    latest_rain_count = one("SELECT COUNT(*) AS n FROM rainfall_records_v1 WHERE crawl_time = ?", (latest_rain_crawl,)).get("n", 0)

    station_type_counts = rows(
        """
        SELECT COALESCE(NULLIF(TRIM(station_type), ''), '未识别') AS station_type,
               COUNT(*) AS record_count,
               COUNT(DISTINCT city || '|' || city_county || '|' || station_name) AS station_count,
               MIN(reported_at) AS min_reported_at,
               MAX(reported_at) AS max_reported_at
        FROM water_level_records_v2
        GROUP BY COALESCE(NULLIF(TRIM(station_type), ''), '未识别')
        ORDER BY record_count DESC
        """
    )
    city_water_counts = rows(
        """
        SELECT city AS 地级市,
               COUNT(*) AS 水位记录数,
               COUNT(DISTINCT city_county || '|' || station_name) AS 水位站点数,
               MIN(reported_at) AS 最早上报时间,
               MAX(reported_at) AS 最晚上报时间,
               MAX(crawl_time) AS 最新抓取时间
        FROM water_level_records_v2
        GROUP BY city
        ORDER BY 水位记录数 DESC
        """
    )
    city_rain_counts = rows(
        """
        SELECT city AS 地级市,
               COUNT(*) AS 雨量记录数,
               COUNT(DISTINCT city_county || '|' || station_name) AS 雨量站点数,
               MIN(period_start) AS 最早统计开始,
               MAX(period_end) AS 最晚统计结束,
               MAX(crawl_time) AS 最新抓取时间
        FROM rainfall_records_v1
        GROUP BY city
        ORDER BY 雨量记录数 DESC
        """
    )
    latest_water_rows = rows(
        """
        SELECT city AS 地级市,
               city_county AS 市县,
               station_name AS 站名,
               station_type AS 站点类型,
               reported_at AS 上报时间,
               water_level AS 水位_m,
               crawl_time AS 抓取时间
        FROM water_level_records_v2
        WHERE crawl_time = ?
        ORDER BY city, city_county, station_name
        """,
        (latest_water_crawl,),
    )
    latest_rain_rows = rows(
        """
        SELECT city AS 地级市,
               city_county AS 市县,
               station_name AS 站名,
               period_start AS 统计开始时间,
               period_end AS 统计结束时间,
               rainfall AS 雨量_mm,
               crawl_time AS 抓取时间
        FROM rainfall_records_v1
        WHERE crawl_time = ?
        ORDER BY city, city_county, station_name
        """,
        (latest_rain_crawl,),
    )
    recent_water_rows = latest_water_rows
    recent_rain_rows = latest_rain_rows

    tide_summary = rows(
        """
        SELECT city AS 地级市,
               city_county AS 市县,
               station_name AS 站名,
               COUNT(*) AS 记录数,
               MIN(reported_at) AS 最早上报时间,
               MAX(reported_at) AS 最晚上报时间,
               MIN(CAST(water_level AS REAL)) AS 最低水位_m,
               MAX(CAST(water_level AS REAL)) AS 最高水位_m,
               ROUND(AVG(CAST(water_level AS REAL)), 3) AS 平均水位_m
        FROM water_level_records_v2
        WHERE station_type = '潮汐'
        GROUP BY city, city_county, station_name
        ORDER BY 记录数 DESC, 地级市, 市县, 站名
        """
    )

    typhoon_files = []
    for path in sorted(TYPHOON_DIR.glob("*白海豚*")):
        typhoon_files.append(
            {
                "文件名": path.name,
                "路径": str(path),
                "大小_字节": path.stat().st_size,
                "修改时间": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    track_path = next(iter(sorted(TYPHOON_DIR.glob("track_*白海豚*.csv"))), None)
    forecast_path = next(iter(sorted(TYPHOON_DIR.glob("forecast_*白海豚*.csv"))), None)
    track_rows = read_csv(track_path)
    forecast_rows = read_csv(forecast_path, limit=5000)

    def numeric_values(data, col):
        vals = []
        for row in data:
            try:
                vals.append(float(row.get(col, "")))
            except Exception:
                pass
        return vals

    pressures = numeric_values(track_rows, "pressure_hpa")
    winds = numeric_values(track_rows, "max_wind_ms")
    typhoon_summary = {
        "台风名称": "白海豚",
        "路径记录数": len(track_rows),
        "预报记录数": len(forecast_rows),
        "最早路径时间": min([r.get("time_bj", "") for r in track_rows if r.get("time_bj")], default=""),
        "最新路径时间": max([r.get("time_bj", "") for r in track_rows if r.get("time_bj")], default=""),
        "最低中心气压_hPa": min(pressures) if pressures else "",
        "最大风速_m_s": max(winds) if winds else "",
        "路径文件": str(track_path) if track_path else "",
        "预报文件": str(forecast_path) if forecast_path else "",
    }

    formal_files = []
    if FORMAL_EXPORT.exists():
        for path in sorted(FORMAL_EXPORT.rglob("*")):
            if path.is_file():
                formal_files.append(
                    {
                        "文件名": path.name,
                        "相对目录": str(path.parent.relative_to(FORMAL_EXPORT)),
                        "大小_字节": path.stat().st_size,
                        "修改时间": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

    csv_paths = {
        "station_type_counts": write_csv("station_type_counts.csv", station_type_counts),
        "city_water_counts": write_csv("city_water_counts.csv", city_water_counts),
        "city_rain_counts": write_csv("city_rain_counts.csv", city_rain_counts),
        "latest_water_rows": write_csv("latest_water_rows.csv", latest_water_rows),
        "latest_rain_rows": write_csv("latest_rain_rows.csv", latest_rain_rows),
        "recent_water_rows": write_csv("recent_water_rows.csv", recent_water_rows),
        "recent_rain_rows": write_csv("recent_rain_rows.csv", recent_rain_rows),
        "tide_summary": write_csv("tide_summary.csv", tide_summary),
        "typhoon_files": write_csv("typhoon_files.csv", typhoon_files),
        "typhoon_track": write_csv("typhoon_track.csv", track_rows),
        "typhoon_forecast_sample": write_csv("typhoon_forecast_sample.csv", forecast_rows),
        "formal_files": write_csv("formal_files.csv", formal_files[:500]),
    }

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "database_path": str(DB),
        "database_size_bytes": DB.stat().st_size,
        "formal_export_path": str(FORMAL_EXPORT),
        "water_overall": water_overall,
        "rain_overall": rain_overall,
        "latest_water_crawl_time": latest_water_crawl,
        "latest_water_crawl_count": latest_water_count,
        "latest_rain_crawl_time": latest_rain_crawl,
        "latest_rain_crawl_count": latest_rain_count,
        "station_type_counts": station_type_counts,
        "typhoon_summary": typhoon_summary,
        "typhoon_files": typhoon_files,
        "csv_paths": csv_paths,
        "excel_note": "Excel 工作簿包含汇总、分城市统计、最新一轮和最近样本；完整明细保留在 SQLite 数据库和正式 CSV 中。",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    con.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
