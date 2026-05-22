"""CNN 可用数据集构建与时间顺序划分。"""

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

    这里会真正尝试构建一次样本，以确保 D-1 和 D 两天共 16 个时间片完整。
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


def save_train_val_arrays(
    sample_dates: list[pd.Timestamp],
    y: np.ndarray,
    era_reader: Era20cReader,
    output_dir: str | Path,
    train_ratio: float = 0.8,
) -> None:
    """按时间顺序划分训练/验证集，并保存为 .npy 文件。

    为了避免完整年份时内存占用过高，X_train/X_val 使用 open_memmap
    逐样本写入磁盘。
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(sample_dates)
    split = int(n_samples * train_ratio)
    if split <= 0 or split >= n_samples:
        raise ValueError("样本数量太少，无法按 80/20 划分训练集和验证集")

    dates_np = np.asarray([d.strftime("%Y-%m-%d") for d in sample_dates], dtype="datetime64[D]")

    train_shape = (split, INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)
    val_shape = (n_samples - split, INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)
    print(f"[DATASET] 总样本数: {n_samples:,}")
    print(f"[DATASET] X 总 shape: ({n_samples}, {INPUT_CHANNELS}, {GRID_SIZE}, {GRID_SIZE})")
    print(f"[DATASET] y 总 shape: {y.shape}")
    y_train_original = y[:split]
    y_val_original = y[split:]
    y_mean = float(np.mean(y_train_original))
    y_std = float(np.std(y_train_original))
    if y_std == 0:
        y_std = 1.0
    y_train = ((y_train_original - y_mean) / y_std).astype("float32")
    y_val = ((y_val_original - y_mean) / y_std).astype("float32")

    print(f"[DATASET] 训练集 shape: X{train_shape}, y{y_train.shape}")
    print(f"[DATASET] 验证集 shape: X{val_shape}, y{y_val.shape}")
    print(f"[DATASET] y 标准化参数: mean={y_mean:.6f}, std={y_std:.6f}")

    x_train = np.lib.format.open_memmap(output_dir / "X_train.npy", mode="w+", dtype="float32", shape=train_shape)
    x_val = np.lib.format.open_memmap(output_dir / "X_val.npy", mode="w+", dtype="float32", shape=val_shape)

    for i, date in enumerate(sample_dates):
        sample = era_reader.build_predictor_for_day(date)
        if sample is None:
            raise RuntimeError(f"第二次构建样本时失败: {date}")
        if i < split:
            x_train[i] = sample
        else:
            x_val[i - split] = sample

        if (i + 1) % 100 == 0 or i + 1 == n_samples:
            print(f"[DATASET] 已写入样本 {i + 1:,}/{n_samples:,}")

    # 显式释放 memmap 句柄，确保 Windows 上文件写入完成。
    del x_train
    del x_val

    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "dates_train.npy", dates_np[:split])
    np.save(output_dir / "dates_val.npy", dates_np[split:])
    np.save(output_dir / "y_original.npy", y)
    np.save(output_dir / "dates_all.npy", dates_np)
    (output_dir / "y_scaler.json").write_text(
        json.dumps({"mean": y_mean, "std": y_std}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[DATASET] 输出文件保存位置: {output_dir.resolve()}")
