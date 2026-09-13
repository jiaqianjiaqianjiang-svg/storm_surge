import sqlite3
from pathlib import Path


DB = Path(r"E:\AAAqian\storm_surge_runtime_data\zhejiang_water_data\zhejiang_water_data.db")
START = "2026-08-27"


def print_rows(title, rows):
    print(title)
    for row in rows:
        print(row)
    print()


def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    print_rows(
        "water_since_2026_08_27_by_hour:",
        cur.execute(
            """
            SELECT substr(crawl_time, 1, 13) AS hour,
                   COUNT(*),
                   MIN(reported_at),
                   MAX(reported_at),
                   MIN(crawl_time),
                   MAX(crawl_time)
            FROM water_level_records_v2
            WHERE crawl_time >= ?
            GROUP BY hour
            ORDER BY hour
            """,
            (START,),
        ).fetchall(),
    )
    print_rows(
        "rain_since_2026_08_27_by_hour:",
        cur.execute(
            """
            SELECT substr(crawl_time, 1, 13) AS hour,
                   COUNT(*),
                   MIN(period_start),
                   MAX(period_end),
                   MIN(crawl_time),
                   MAX(crawl_time)
            FROM rainfall_records_v1
            WHERE crawl_time >= ?
            GROUP BY hour
            ORDER BY hour
            """,
            (START,),
        ).fetchall(),
    )
    print_rows(
        "latest_water_samples:",
        cur.execute(
            """
            SELECT city, city_county, station_name, station_type, reported_at, water_level, crawl_time
            FROM water_level_records_v2
            WHERE crawl_time >= ?
            ORDER BY crawl_time DESC
            LIMIT 20
            """,
            (START,),
        ).fetchall(),
    )
    print_rows(
        "latest_rain_samples:",
        cur.execute(
            """
            SELECT city, city_county, station_name, period_start, period_end, rainfall, crawl_time
            FROM rainfall_records_v1
            WHERE crawl_time >= ?
            ORDER BY crawl_time DESC
            LIMIT 20
            """,
            (START,),
        ).fetchall(),
    )
    print_rows(
        "totals:",
        [
            cur.execute(
                "SELECT COUNT(*), MIN(reported_at), MAX(reported_at), MIN(crawl_time), MAX(crawl_time) FROM water_level_records_v2 WHERE crawl_time >= ?",
                (START,),
            ).fetchone(),
            cur.execute(
                "SELECT COUNT(*), MIN(period_start), MAX(period_end), MIN(crawl_time), MAX(crawl_time) FROM rainfall_records_v1 WHERE crawl_time >= ?",
                (START,),
            ).fetchone(),
        ],
    )
    conn.close()


if __name__ == "__main__":
    main()
