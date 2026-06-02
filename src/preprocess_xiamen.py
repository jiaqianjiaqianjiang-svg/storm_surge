"""论文验证站点预处理命令行入口。

示例：
python src/preprocess_xiamen.py --site xiamen --start-year 1985 --end-year 1985
python src/preprocess_xiamen.py --site lusi --all-years --split-mode first-years --validation-years 5
"""

from __future__ import annotations

import argparse
import sys


def configure_console_encoding() -> None:
    """尽量将控制台输出切到 UTF-8，避免 Windows 终端打印中文时报编码错误。"""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except OSError:
                pass


configure_console_encoding()

from config import SITES, SiteConfig, get_site_config
from dataset_builder import collect_available_samples, save_train_val_arrays
from era20c_loader import Era20cReader
from gesla_loader import read_gesla_file, restrict_years
from tide_processing import daily_maximum_surge, separate_tide_with_utide


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="预处理 GESLA + ERA20C，生成 CNN 可用数据集。")
    parser.add_argument("--site", choices=sorted(SITES), default="xiamen", help="要处理的论文验证站点")
    parser.add_argument("--all-years", action="store_true", help="使用该站点配置中的全部可用年份")
    parser.add_argument("--start-year", type=int, help="开始年份，例如 1985")
    parser.add_argument("--end-year", type=int, help="结束年份，例如 1985")
    parser.add_argument(
        "--split-mode",
        choices=["auto", "first-years", "chronological"],
        default="auto",
        help="训练/验证划分方式。多年数据建议 first-years；单年测试可用 auto 或 chronological。",
    )
    parser.add_argument("--validation-years", type=int, default=5, help="论文式划分中用于验证集的最早年份数量")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="chronological 划分时训练集比例")
    parser.add_argument(
        "--skip-missing-era",
        action="store_true",
        help="如果某一年 ERA 文件缺失，则跳过该年；默认缺失即报错。",
    )
    return parser.parse_args()


def resolve_years(args: argparse.Namespace, site: SiteConfig) -> tuple[int, int]:
    """根据命令行参数和站点配置确定年份范围。"""

    if args.all_years:
        return site.start_year, site.end_year
    if args.start_year is None or args.end_year is None:
        raise SystemExit("请使用 --all-years，或同时提供 --start-year 和 --end-year")
    if args.start_year > args.end_year:
        raise SystemExit("--start-year 不能大于 --end-year")
    return args.start_year, args.end_year


def main() -> None:
    """执行完整预处理流程。"""

    args = parse_args()
    site = get_site_config(args.site)
    start_year, end_year = resolve_years(args, site)
    years = list(range(start_year, end_year + 1))
    site_file = site.file_path
    output_dir = site.output_dir

    print("=" * 80)
    print(f"[RUN] 站点: {site.name} ({site.key})")
    print(f"[RUN] 年份范围: {start_year}-{end_year}")
    print(f"[RUN] GESLA 文件: {site_file}")
    print(f"[RUN] 输出目录: {output_dir}")
    print("=" * 80)

    gesla = read_gesla_file(site_file)
    gesla = restrict_years(gesla, start_year, end_year)

    surge = separate_tide_with_utide(gesla, lat=site.lat)
    output_dir.mkdir(parents=True, exist_ok=True)
    surge.to_csv(output_dir / "cleaned_surge.csv", encoding="utf-8-sig")
    print(f"[SAVE] cleaned_surge.csv: {output_dir / 'cleaned_surge.csv'}")

    daily = daily_maximum_surge(surge)
    daily.to_csv(output_dir / "daily_max_surge.csv", encoding="utf-8-sig")
    print(f"[SAVE] daily_max_surge.csv: {output_dir / 'daily_max_surge.csv'}")

    era_reader = Era20cReader(site_key=site.key, site_lat=site.lat, site_lon=site.lon)
    available_years = era_reader.available_years(years, skip_missing=args.skip_missing_era)
    era_reader.compute_standardization(available_years)

    sample_dates, y = collect_available_samples(daily, era_reader)
    save_train_val_arrays(
        sample_dates,
        y,
        era_reader,
        output_dir,
        train_ratio=args.train_ratio,
        split_mode=args.split_mode,
        validation_years=args.validation_years,
    )

    print("=" * 80)
    print(f"[DONE] {site.name} 预处理完成。")
    print(f"[DONE] 输出目录: {output_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
