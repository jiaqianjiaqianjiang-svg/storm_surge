"""GESLA-3 潮位文件读取与清洗。

目标是尽量兼容 GESLA 文本文件中常见的日期/时间格式，并保留足够宽松的
异常检测策略，避免误删真实风暴潮极值。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import MISSING_VALUE_MARKERS, OBS_MAD_THRESHOLD, SEA_LEVEL_ABS_LIMIT


@dataclass
class ParsedGeslaRow:
    """单行 GESLA 数据解析结果。"""

    datetime: pd.Timestamp
    sea_level: float
    qc_flag: float | None
    use_flag: float | None


def _is_number(value: str) -> bool:
    """判断字符串能否转成浮点数。"""

    try:
        float(value)
        return True
    except ValueError:
        return False


def _parse_date_time(tokens: list[str]) -> tuple[pd.Timestamp | None, int]:
    """从一行 token 开头解析时间，并返回已消耗的 token 数。

    GESLA 文件在不同站点可能出现几种格式：
    - YYYY-MM-DD HH:MM sea_level ...
    - YYYY/MM/DD HH:MM sea_level ...
    - YYYYMMDD HHMM sea_level ...
    - YYYY MM DD HH MM sea_level ...

    本函数只负责识别时间，不负责解析潮位值。
    """

    if len(tokens) < 2:
        return None, 0

    # 情况 1：日期和时间分别放在前两个字段。
    candidate = f"{tokens[0]} {tokens[1]}"
    dt = pd.to_datetime(candidate, errors="coerce")
    if pd.notna(dt):
        return pd.Timestamp(dt), 2

    # 情况 2：前五列分别是 year/month/day/hour/minute。
    if len(tokens) >= 5 and all(_is_number(t) for t in tokens[:5]):
        year, month, day, hour, minute = [int(float(t)) for t in tokens[:5]]
        dt = pd.to_datetime(
            {
                "year": [year],
                "month": [month],
                "day": [day],
                "hour": [hour],
                "minute": [minute],
            },
            errors="coerce",
        )[0]
        if pd.notna(dt):
            return pd.Timestamp(dt), 5

    # 情况 3：前四列是 year/month/day/hour，分钟默认为 0。
    if len(tokens) >= 4 and all(_is_number(t) for t in tokens[:4]):
        year, month, day, hour = [int(float(t)) for t in tokens[:4]]
        dt = pd.to_datetime(
            {"year": [year], "month": [month], "day": [day], "hour": [hour]},
            errors="coerce",
        )[0]
        if pd.notna(dt):
            return pd.Timestamp(dt), 4

    return None, 0


def _parse_data_line(line: str) -> ParsedGeslaRow | None:
    """尝试解析一行 GESLA 数据。

    如果该行不是数据行，返回 None。这样可以自动跳过元数据和表头。
    """

    tokens = line.strip().split()
    if not tokens:
        return None

    dt, consumed = _parse_date_time(tokens)
    if dt is None or consumed >= len(tokens):
        return None

    if not _is_number(tokens[consumed]):
        return None

    sea_level = float(tokens[consumed])
    qc_flag = float(tokens[consumed + 1]) if consumed + 1 < len(tokens) and _is_number(tokens[consumed + 1]) else None
    use_flag = float(tokens[consumed + 2]) if consumed + 2 < len(tokens) and _is_number(tokens[consumed + 2]) else None

    return ParsedGeslaRow(dt, sea_level, qc_flag, use_flag)


def _robust_mad_filter(series: pd.Series, threshold: float) -> pd.Series:
    """使用 MAD 做宽松异常值检测，返回 True/False 掩码。

    MAD 对极端值不敏感，适合先粗略排除明显坏点。这里阈值设置得很宽松，
    用于保留真实风暴潮极值。
    """

    values = series.to_numpy(dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(True, index=series.index)

    robust_z = 0.6745 * (values - median) / mad
    return pd.Series(np.abs(robust_z) <= threshold, index=series.index)


def read_gesla_file(path: str | Path, keep_only_use_flag: bool = True) -> pd.DataFrame:
    """读取并清洗 GESLA 潮位文件。

    Parameters
    ----------
    path:
        GESLA 站点文件路径。
    keep_only_use_flag:
        如果文件中存在 use_flag，是否只保留 use_flag 不为 0 的记录。

    Returns
    -------
    pandas.DataFrame
        索引为 datetime，包含 sea_level、qc_flag、use_flag 三列。
    """

    path = Path(path)
    print(f"[GESLA] 读取文件: {path}")

    rows: list[ParsedGeslaRow] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = _parse_data_line(line)
            if parsed is not None:
                rows.append(parsed)

    if not rows:
        raise ValueError(f"没有在 GESLA 文件中识别到数据行: {path}")

    df = pd.DataFrame(
        {
            "datetime": [row.datetime for row in rows],
            "sea_level": [row.sea_level for row in rows],
            "qc_flag": [row.qc_flag for row in rows],
            "use_flag": [row.use_flag for row in rows],
        }
    )
    print(f"[GESLA] 原始识别记录数: {len(df):,}")

    # 统一时间索引，删除重复时间点。
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    before = len(df)
    df = df.drop_duplicates(subset=["datetime"], keep="first")
    print(f"[GESLA] 删除重复时间: {before - len(df):,}")

    # 缺测值处理。
    df["sea_level"] = pd.to_numeric(df["sea_level"], errors="coerce")
    missing_mask = df["sea_level"].isin(MISSING_VALUE_MARKERS) | df["sea_level"].isna()
    before = len(df)
    df = df.loc[~missing_mask].copy()
    print(f"[GESLA] 删除缺测值: {before - len(df):,}")

    # 如果 use_flag 明确给出，通常 0 表示不建议使用。
    if keep_only_use_flag and df["use_flag"].notna().any():
        before = len(df)
        df = df.loc[df["use_flag"].fillna(1) != 0].copy()
        print(f"[GESLA] 根据 use_flag 删除记录: {before - len(df):,}")

    # 先用非常宽的物理范围删除明显坏值，再用宽松 MAD 过滤。
    before = len(df)
    df = df.loc[df["sea_level"].abs() <= SEA_LEVEL_ABS_LIMIT].copy()
    print(f"[GESLA] 删除超出宽松物理范围的记录: {before - len(df):,}")

    before = len(df)
    df = df.loc[_robust_mad_filter(df["sea_level"], OBS_MAD_THRESHOLD)].copy()
    print(f"[GESLA] 宽松 MAD 删除观测异常值: {before - len(df):,}")

    df = df.set_index("datetime").sort_index()
    print(f"[GESLA] 清洗后记录数: {len(df):,}")
    print(f"[GESLA] 时间范围: {df.index.min()} -> {df.index.max()}")
    return df


def restrict_years(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """按年份裁剪 GESLA 数据。"""

    start = pd.Timestamp(year=start_year, month=1, day=1)
    end = pd.Timestamp(year=end_year, month=12, day=31, hour=23, minute=59, second=59)
    out = df.loc[(df.index >= start) & (df.index <= end)].copy()
    print(f"[GESLA] 使用年份 {start_year}-{end_year}，记录数: {len(out):,}")
    if out.empty:
        raise ValueError(f"年份范围 {start_year}-{end_year} 内没有 GESLA 记录")
    return out
