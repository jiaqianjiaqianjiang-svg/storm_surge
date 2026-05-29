"""CNN 可用数据集构建与训练/验证划分。

本文件负责把 daily maximum storm surge 标签和 ERA-20C 气象场对齐，
并保存成训练脚本可以直接读取的 .npy 文件。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import GRID_SIZE, INPUT_CHANNELS
from era20c_loader import Era20cReader


def collect_available_samples(
    daily_max: pd.DataFrame,
    era_reader: Era20cReader,
) -> tuple[list[pd.Timestamp], np.ndarray]:
    """遍历标签日期，筛选出 ERA 时间片齐全的样本日期。

    对某一天 D 的 daily maximum storm surge，论文使用 D-1 和 D 两天的 ERA-20C
    气象场作为输入。这里会真实尝试构建一次样本，确保两天共 16 个 3 小时时间片完整，
    并且插值后的 U10、V10、SLP 不包含 NaN/Inf。
    """

    sample_dates: list[pd.Timestamp] = []
    sample_y: list[float] = []
    year_counts: dict[int, int] = {}
    skipped_count = 0
    skipped_reasons: dict[str, int] = {}

    for date, row in daily_max.iterrows():
        y_value = row["daily_max_surge"]
        if pd.isna(y_value):
            skipped_count += 1
            skipped_reasons["y 缺失"] = skipped_reasons.get("y 缺失", 0) + 1
            continue

        try:
            sample = era_reader.build_predictor_for_day(pd.Timestamp(date))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            skipped_count += 1
            reason = str(exc).splitlines()[0]
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue

        if sample is None:
            skipped_count += 1
            reason = era_reader.explain_missing_for_day(pd.Timestamp(date))
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue

        ts = pd.Timestamp(date).normalize()
        sample_dates.append(ts)
        sample_y.append(float(y_value))
        year_counts[ts.year] = year_counts.get(ts.year, 0) + 1

    for year in sorted(year_counts):
        print(f"[DATASET] {year} 年生成样本数: {year_counts[year]:,}")
    print(f"[DATASET] 跳过日期数: {skipped_count:,}")
    for reason, count in sorted(skipped_reasons.items(), key=lambda item: item[1], reverse=True):
        print(f"[DATASET] 跳过原因: {reason} -> {count:,} 天")

    if not sample_dates:
        raise ValueError("没有生成任何 CNN 样本，请检查 GESLA 标签和 ERA 时间范围是否重叠")

    return sample_dates, np.asarray(sample_y, dtype="float32")


def split_train_val_indices(
    sample_dates: list[pd.Timestamp],
    train_ratio: float,
    split_mode: str,
    validation_years: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """根据日期生成训练集和验证集索引。

    论文中厦门等站点的独立验证更接近“前五年作为验证集，其余年份训练”。
    这个函数把该逻辑集中在一起，避免训练脚本里再偷偷重新划分。
    """

    if split_mode not in {"auto", "first-years", "chronological"}:
        raise ValueError("--split-mode 只能是 auto、first-years 或 chronological")
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio 必须在 0 到 1 之间")
    if validation_years <= 0:
        raise ValueError("--validation-years 必须大于 0")

    years = np.asarray([pd.Timestamp(date).year for date in sample_dates], dtype=int)
    unique_years = np.asarray(sorted(set(years.tolist())), dtype=int)

    active_mode = split_mode
    if split_mode == "auto":
        if unique_years.size > validation_years:
            active_mode = "first-years"
        else:
            active_mode = "chronological"
            print(
                "[DATASET] 年份不足以使用“前 5 年验证、其余训练”，"
                "自动改用 80/20 时间顺序划分，仅用于快速流程测试。"
            )

    if active_mode == "first-years":
        val_years = set(unique_years[:validation_years].tolist())
        val_indices = np.asarray([i for i, year in enumerate(years) if int(year) in val_years], dtype=int)
        train_indices = np.asarray([i for i, year in enumerate(years) if int(year) not in val_years], dtype=int)
        if train_indices.size == 0 or val_indices.size == 0:
            raise ValueError(
                "论文式前若干年验证划分失败：训练集或验证集为空。"
                "单年测试请使用 --split-mode auto 或 --split-mode chronological。"
            )

        train_years = sorted(set(years[train_indices].tolist()))
        val_years_sorted = sorted(val_years)
        print(
            f"[DATASET] 论文式划分: 验证年份 {val_years_sorted[0]}-{val_years_sorted[-1]}，"
            f"训练年份 {train_years[0]}-{train_years[-1]}"
        )
        return train_indices, val_indices, f"first-{validation_years}-years-validation"

    split = int(len(sample_dates) * train_ratio)
    if split <= 0 or split >= len(sample_dates):
        raise ValueError("样本数量太少，无法按时间顺序 80/20 划分训练集和验证集")
    train_indices = np.arange(0, split, dtype=int)
    val_indices = np.arange(split, len(sample_dates), dtype=int)
    print(f"[DATASET] 时间顺序划分: 前 {train_ratio:.0%} 训练，后 {1 - train_ratio:.0%} 验证")
    return train_indices, val_indices, "chronological-ratio"


def save_train_val_arrays(
    sample_dates: list[pd.Timestamp],
    y: np.ndarray,
    era_reader: Era20cReader,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    split_mode: str = "auto",
    validation_years: int = 5,
) -> None:
    """划分训练/验证集，并保存为 .npy 文件。

    split_mode 支持三种写法：
    - auto：多年完整数据默认用论文式“最早 5 年验证，其余训练”；年份不足时退回 80/20，
      方便单年快速测试流程。
    - first-years：强制使用最早 validation_years 年作为验证集。
    - chronological：按时间顺序前 train_ratio 训练、后面验证。

    为了避免完整年份时内存占用过高，X_train/X_val 使用 open_memmap 逐样本写入磁盘。
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(sample_dates)
    if n_samples < 2:
        raise ValueError("样本数量太少，无法划分训练集和验证集")

    dates_np = np.asarray([d.strftime("%Y-%m-%d") for d in sample_dates], dtype="datetime64[D]")
    train_indices, val_indices, active_split_mode = split_train_val_indices(
        sample_dates=sample_dates,
        train_ratio=train_ratio,
        split_mode=split_mode,
        validation_years=validation_years,
    )

    train_shape = (len(train_indices), INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)
    val_shape = (len(val_indices), INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)
    print(f"[DATASET] 总样本数: {n_samples:,}")
    print(f"[DATASET] X 总 shape: ({n_samples}, {INPUT_CHANNELS}, {GRID_SIZE}, {GRID_SIZE})")
    print(f"[DATASET] y 总 shape: {y.shape}")
    print(f"[DATASET] 划分方式: {active_split_mode}")

    y_train_original = y[train_indices]
    y_val_original = y[val_indices]
    y_mean = float(np.mean(y_train_original))
    y_std = float(np.std(y_train_original))
    if y_std == 0:
        y_std = 1.0
    y_train = ((y_train_original - y_mean) / y_std).astype("float32")
    y_val = ((y_val_original - y_mean) / y_std).astype("float32")

    print(f"[DATASET] 训练集 shape: X{train_shape}, y{y_train.shape}")
    print(f"[DATASET] 验证集 shape: X{val_shape}, y{y_val.shape}")
    print(f"[DATASET] y 标准化参数: mean={y_mean:.6f}, std={y_std:.6f}")
    print(f"[DATASET] 训练集日期: {dates_np[train_indices][0]} -> {dates_np[train_indices][-1]}")
    print(f"[DATASET] 验证集日期: {dates_np[val_indices][0]} -> {dates_np[val_indices][-1]}")

    x_train = np.lib.format.open_memmap(output_dir / "X_train.npy", mode="w+", dtype="float32", shape=train_shape)
    x_val = np.lib.format.open_memmap(output_dir / "X_val.npy", mode="w+", dtype="float32", shape=val_shape)

    for write_i, sample_i in enumerate(train_indices):
        date = sample_dates[int(sample_i)]
        sample = era_reader.build_predictor_for_day(date)
        if sample is None:
            raise RuntimeError(f"第二次构建训练样本时失败: {date}")
        x_train[write_i] = sample
        if (write_i + 1) % 100 == 0 or write_i + 1 == len(train_indices):
            print(f"[DATASET] 已写入训练样本 {write_i + 1:,}/{len(train_indices):,}")

    for write_i, sample_i in enumerate(val_indices):
        date = sample_dates[int(sample_i)]
        sample = era_reader.build_predictor_for_day(date)
        if sample is None:
            raise RuntimeError(f"第二次构建验证样本时失败: {date}")
        x_val[write_i] = sample
        if (write_i + 1) % 100 == 0 or write_i + 1 == len(val_indices):
            print(f"[DATASET] 已写入验证样本 {write_i + 1:,}/{len(val_indices):,}")

    # 显式释放 memmap 句柄，确保 Windows 上文件写入完成。
    del x_train
    del x_val

    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "dates_train.npy", dates_np[train_indices])
    np.save(output_dir / "dates_val.npy", dates_np[val_indices])
    np.save(output_dir / "y_original.npy", y)
    np.save(output_dir / "y_train_original.npy", y_train_original.astype("float32"))
    np.save(output_dir / "y_val_original.npy", y_val_original.astype("float32"))
    np.save(output_dir / "dates_all.npy", dates_np)
    (output_dir / "y_scaler.json").write_text(
        json.dumps({"mean": y_mean, "std": y_std}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "split_metadata.json").write_text(
        json.dumps(
            {
                "requested_split_mode": split_mode,
                "active_split_mode": active_split_mode,
                "validation_years": validation_years,
                "train_ratio": train_ratio,
                "n_total": int(n_samples),
                "n_train": int(len(train_indices)),
                "n_val": int(len(val_indices)),
                "train_start": str(dates_np[train_indices][0]),
                "train_end": str(dates_np[train_indices][-1]),
                "val_start": str(dates_np[val_indices][0]),
                "val_end": str(dates_np[val_indices][-1]),
                "note": "first-years 表示论文式最早若干年验证；chronological 表示按时间前后 80/20 快速测试。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[DATASET] 输出文件保存位置: {output_dir.resolve()}")
