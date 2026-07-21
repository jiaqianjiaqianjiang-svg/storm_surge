"""潮汐分离与 daily maximum storm surge 标签生成。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.dates import date2num
from utide import reconstruct, solve

from config import SITE_LAT, SURGE_MAD_THRESHOLD
from gesla_loader import _robust_mad_filter


def separate_tide_with_utide(df: pd.DataFrame, lat: float = SITE_LAT) -> pd.DataFrame:
    """使用 UTide 从观测潮位中分离预测潮汐和风暴潮。

    输入 df 需要以 datetime 为索引，并包含 sea_level 列。
    输出包含 observed_sea_level、predicted_tide、storm_surge 三列。
    """

    if "sea_level" not in df.columns:
        raise KeyError("输入 DataFrame 必须包含 sea_level 列")

    work = df[["sea_level"]].dropna().copy()
    if len(work) < 24 * 30:
        raise ValueError("可用于 UTide 调和分析的记录太少，建议至少包含 1 个月以上数据")

    print("[TIDE] 开始 UTide 调和分析，这一步在完整年份上可能需要一些时间...")
    time_num = date2num(work.index.to_pydatetime())
    observed = work["sea_level"].to_numpy(dtype=float)

    # trend=False：此处按用户要求直接计算 predicted tide，然后用 observed - predicted tide。
    coef = solve(
        time_num,
        observed,
        lat=lat,
        method="ols",
        # 预处理只需要调和常数和重建潮汐，不需要置信区间。
        # 关闭置信区间可避免 UTide 在规则小时数据上额外计算 periodogram 时
        # 产生无关的 divide-by-zero warning。
        conf_int="none",
        trend=False,
        verbose=False,
    )
    tide = reconstruct(time_num, coef, verbose=False).h

    out = pd.DataFrame(
        {
            "observed_sea_level": observed,
            "predicted_tide": tide,
            "storm_surge": observed - tide,
        },
        index=work.index,
    )

    before = len(out)
    out = out.loc[_robust_mad_filter(out["storm_surge"], SURGE_MAD_THRESHOLD)].copy()
    print(f"[TIDE] 宽松 MAD 删除 storm surge 异常值: {before - len(out):,}")
    print(f"[TIDE] storm surge 记录数: {len(out):,}")
    return out


def daily_maximum_surge(surge_df: pd.DataFrame) -> pd.DataFrame:
    """按天提取 daily maximum storm surge，作为 CNN 标签 y。"""

    if "storm_surge" not in surge_df.columns:
        raise KeyError("输入 DataFrame 必须包含 storm_surge 列")

    daily = surge_df["storm_surge"].resample("D").max().dropna()
    out = daily.rename("daily_max_surge").to_frame()
    out.index.name = "date"
    print(f"[TIDE] daily maximum storm surge 天数: {len(out):,}")
    print(f"[TIDE] daily 标签范围: {out.index.min().date()} -> {out.index.max().date()}")
    return out
