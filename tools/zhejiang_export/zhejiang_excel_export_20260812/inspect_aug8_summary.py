import sqlite3
from pathlib import Path


DB = Path(r"E:\AAAqian\storm_surge_runtime_data\zhejiang_water_data\zhejiang_water_data.db")
START = "2026-08-08"


def print_rows(title, rows):
    print(title)
    for row in rows:
        print(row)
    print()


def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    print_rows(
        "water_by_crawl_period:",
        cur.execute(
            """
            SELECT
                CASE WHEN crawl_time < ? THEN 'before_2026-08-08' ELSE 'from_2026-08-08' END AS period,
                COUNT(*) AS record_count,
                MIN(reported_at),
                MAX(reported_at),
                MIN(crawl_time),
                MAX(crawl_time)
            FROM water_level_records_v2
            GROUP BY period
            ORDER BY period
            """,
            (START,),
        ).fetchall(),
    )

    print_rows(
        "water_station_type_from_aug8:",
        cur.execute(
            """
            SELECT COALESCE(NULLIF(TRIM(station_type), ''), '(空)') AS station_type, COUNT(*)
            FROM water_level_records_v2
            WHERE crawl_time >= ?
            GROUP BY COALESCE(NULLIF(TRIM(station_type), ''), '(空)')
            ORDER BY COUNT(*) DESC
            """,
            (START,),
        ).fetchall(),
    )

    print_rows(
        "rain_by_crawl_period:",
        cur.execute(
            """
            SELECT
                CASE WHEN crawl_time < ? THEN 'before_2026-08-08' ELSE 'from_2026-08-08' END AS period,
                COUNT(*) AS record_count,
                MIN(period_start),
                MAX(period_end),
                MIN(crawl_time),
                MAX(crawl_time)
            FROM rainfall_records_v1
            GROUP BY period
            ORDER BY period
            """,
            (START,),
        ).fetchall(),
    )

    print_rows(
        "rain_by_crawl_date_from_aug8:",
        cur.execute(
            """
            SELECT substr(crawl_time, 1, 10), COUNT(*), MIN(period_start), MAX(period_end)
            FROM rainfall_records_v1
            WHERE crawl_time >= ?
            GROUP BY substr(crawl_time, 1, 10)
            ORDER BY 1
            """,
            (START,),
        ).fetchall(),
    )

    conn.close()


if __name__ == "__main__":
    main()
