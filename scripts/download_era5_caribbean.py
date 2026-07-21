"""按加勒比验潮站下载 ERA5 小时级气象强迫场。

下载内容与当前风暴潮模型一致：10 m U/V 风和平均海平面气压。
每个站点、年份、变量保存为一个 NetCDF，避免单次 CDS 请求过大。
"""

from __future__ import annotations

import argparse
from pathlib import Path


STATIONS = {
    "calq": {"name": "Calliaqua", "lat": 13.130, "lon": -61.196},
    "stlu": {"name": "Ganters_Bay", "lat": 14.017, "lon": -61.000},
    "pric": {"name": "Prickly_Bay", "lat": 12.005, "lon": -61.765},
    "san_juan": {"name": "San_Juan", "lat": 18.459, "lon": -66.116},
    "charlotte_amalie": {"name": "Charlotte_Amalie", "lat": 18.336, "lon": -64.920},
}

VARIABLES = {
    "u10": ("10m_u_component_of_wind", "10u"),
    "v10": ("10m_v_component_of_wind", "v10"),
    "slp": ("mean_sea_level_pressure", "slp"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", choices=sorted(STATIONS), required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/ERA5-Caribbean"),
        help="输出根目录；默认 data/ERA5-Caribbean",
    )
    parser.add_argument(
        "--half-size-deg",
        type=float,
        default=5.0,
        help="站点四周裁剪半径；5° 对应项目现有 10°×10° 输入区域",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在且非空的文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start-year 不能晚于 end-year")
    if args.half_size_deg <= 0:
        raise ValueError("half-size-deg 必须大于 0")

    try:
        import cdsapi
    except ModuleNotFoundError as exc:
        raise SystemExit('请先安装 CDS 客户端：pip install "cdsapi>=0.7.7"') from exc

    station = STATIONS[args.station]
    station_dir = args.output_root / station["name"]
    station_dir.mkdir(parents=True, exist_ok=True)
    area = [
        station["lat"] + args.half_size_deg,
        station["lon"] - args.half_size_deg,
        station["lat"] - args.half_size_deg,
        station["lon"] + args.half_size_deg,
    ]

    client = cdsapi.Client()
    months = [f"{month:02d}" for month in range(1, 13)]
    days = [f"{day:02d}" for day in range(1, 32)]
    times = [f"{hour:02d}:00" for hour in range(24)]

    for year in range(args.start_year, args.end_year + 1):
        for logical_name, (cds_name, file_hint) in VARIABLES.items():
            target = station_dir / f"{args.station}_{file_hint}_{year}.nc"
            if target.exists() and target.stat().st_size > 0 and not args.overwrite:
                print(f"[SKIP] 已存在: {target}")
                continue
            request = {
                "product_type": ["reanalysis"],
                "variable": [cds_name],
                "year": [str(year)],
                "month": months,
                "day": days,
                "time": times,
                "data_format": "netcdf",
                "area": area,
            }
            print(f"[CDS] {station['name']} {year} {logical_name} -> {target}")
            client.retrieve("reanalysis-era5-single-levels", request, str(target))


if __name__ == "__main__":
    main()
