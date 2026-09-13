import csv
from pathlib import Path


FILES = [
    Path(r"F:\data\02_typhoon_bavi_202607\zhejiang_water_observations\formal_export\01_complete_total\water_level_all.csv"),
    Path(r"F:\data\02_typhoon_bavi_202607\zhejiang_water_observations\formal_export\01_complete_total\rainfall_all.csv"),
    Path(r"F:\data\02_typhoon_bavi_202607\zhejiang_water_observations\formal_export\02_station_type\water_level_tide.csv"),
    Path(r"F:\data\02_typhoon_bavi_202607\zhejiang_water_observations\formal_export\segments\segment_summary.csv"),
    Path(r"F:\data\03_typhoon_white_dolphin_20260808\zhejiang_water_observations\exports\white_dolphin_20260808\white_dolphin_water_level.csv"),
    Path(r"F:\data\03_typhoon_white_dolphin_20260808\zhejiang_water_observations\exports\white_dolphin_20260808\white_dolphin_rainfall.csv"),
]


def count_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in f) - 1


for path in FILES:
    print("FILE:", path)
    print("exists:", path.exists(), "size_MB:", round(path.stat().st_size / 1024 / 1024, 2) if path.exists() else "")
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            print("header:", next(reader))
            for _, row in zip(range(2), reader):
                print("sample:", row)
        print("rows:", count_rows(path))
    print()
