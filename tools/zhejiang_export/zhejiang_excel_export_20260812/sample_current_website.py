import json
from collections import Counter
from pathlib import Path


NOTEBOOK = Path(r"E:\AAAqian\code\storm_surge_clean\real_time_data_crawler\water_level\爬虫_浙江全省\浙江全省.ipynb")


def load_notebook_defs():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns = {}
    for idx in (1, 2, 3, 4):
        exec("".join(nb["cells"][idx].get("source", [])), ns)
    ns["HEADLESS"] = True
    ns["MAX_RETRIES"] = 2
    ns["RETRY_DELAY"] = 2
    return ns


def main():
    ns = load_notebook_defs()
    city = "杭州市"
    pool = ns["DriverPool"](1)
    pool.initialize()
    try:
        records = ns["fetch_city_data"](pool, city)
    finally:
        pool.close_all()

    reported = [r["reported_at"] for r in records if r.get("reported_at")]
    date_counts = Counter(x[:10] for x in reported)
    print("city:", city)
    print("record_count:", len(records))
    print("min_reported_at:", min(reported) if reported else "")
    print("max_reported_at:", max(reported) if reported else "")
    print("reported_date_counts:")
    for k, v in sorted(date_counts.items()):
        print(k, v)
    print("sample_latest:")
    for row in sorted(records, key=lambda r: r["reported_at"], reverse=True)[:20]:
        print(row["city"], row["city_county"], row["station_name"], row["reported_at"], row["water_level"])


if __name__ == "__main__":
    main()
