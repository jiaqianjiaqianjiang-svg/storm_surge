# -*- coding: utf-8 -*-
"""????????????????????

?????
python 01_xiamen_preprocess_all_in_one.py --start-year 1985 --end-year 1985
python 01_xiamen_preprocess_all_in_one.py --all-years --split-mode first-years --validation-years 5
"""
from __future__ import annotations


# ==================== config.py ====================
"""项目集中配置。

本文件只保存路径、站点信息、网格大小等配置项，不读取真实数据。
在实验室远程电脑运行时，通常只需要检查这里的路径是否正确。
"""

from pathlib import Path


# =========================
# 1. 真实数据路径
# =========================

# ERA-20C 根目录。目录下应包含 10U、10V、SLP 三个子目录。
ERA20C_DIR = Path(r"F:\ERA20C")

# GESLA-3 根目录。
GESLA_DIR = Path(r"F:\GESLA\GESLA3")


# =========================
# 2. 厦门站信息
# =========================

SITE_NAME = "Xiamen"
SITE_FILE = Path(r"F:\GESLA\GESLA3\xiamen-376a-chn-uhslc")
SITE_LAT = 24.45
SITE_LON = 118.067

# GESLA 厦门站大致可用年份。--all-years 会使用这个范围。
XIAMEN_START_YEAR = 1954
XIAMEN_END_YEAR = 1997


# =========================
# 3. ERA-20C 变量目录
# =========================

ERA20C_VARIABLE_DIRS = {
    "u10": ERA20C_DIR / "10U",
    "v10": ERA20C_DIR / "10V",
    "slp": ERA20C_DIR / "SLP",
}

# cfgrib/xarray 读入后，不同文件里的变量名可能略有不同。
# 这里按优先级列出候选名称。
ERA20C_VARIABLE_CANDIDATES = {
    "u10": ("u10", "10u", "u", "var165"),
    "v10": ("v10", "10v", "v", "var166"),
    "slp": ("msl", "slp", "sp", "var151"),
}


# =========================
# 4. CNN 输入参数
# =========================

# 站点周围 10°×10° 区域，即经纬度各向外扩展 5°。
REGION_HALF_SIZE_DEG = 5.0

# 插值后的空间网格。
GRID_SIZE = 40

# ERA-20C 3 小时一个时间片，一天 8 个，两天 16 个。
HOURS_PER_STEP = 3
STEPS_PER_DAY = 8
INPUT_DAYS = 2
STEPS_PER_SAMPLE = STEPS_PER_DAY * INPUT_DAYS

# 三个变量拼接后，单个样本通道数为 16 × 3 = 48。
VARIABLE_ORDER = ("u10", "v10", "slp")
TIME_TILE_ROWS = 4
TIME_TILE_COLS = 4
INPUT_CHANNELS = len(VARIABLE_ORDER)
MODEL_GRID_SIZE = GRID_SIZE * TIME_TILE_ROWS


# =========================
# 5. 输出目录
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
XIAMEN_OUTPUT_DIR = OUTPUT_ROOT / "xiamen"
CACHE_ROOT = PROJECT_ROOT / "cache"
ERA20C_CACHE_DIR = CACHE_ROOT / "xiamen" / "era20c_yearly"


# =========================
# 6. 清洗参数
# =========================

# GESLA 常见缺测标记。
MISSING_VALUE_MARKERS = {-99, -999, -9999, 9999, 99999}

# 宽松物理范围，单位沿用原始 GESLA 文件。这里只用于去除明显坏值，
# 不应删除真实极端风暴潮。
SEA_LEVEL_ABS_LIMIT = 10_000.0

# MAD 异常检测阈值。阈值较大，目标是只去掉明显离群坏点。
OBS_MAD_THRESHOLD = 15.0
SURGE_MAD_THRESHOLD = 15.0


# ==================== gesla_loader.py ====================
"""GESLA-3 潮位文件读取与清洗。

目标是尽量兼容 GESLA 文本文件中常见的日期/时间格式，并保留足够宽松的
异常检测策略，避免误删真实风暴潮极值。
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd



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


# ==================== tide_processing.py ====================
"""潮汐分离与 daily maximum storm surge 标签生成。"""


import numpy as np
import pandas as pd
from utide import reconstruct, solve



TIDAL_CONSTITUENTS = (
    "M2",
    "S2",
    "N2",
    "K2",
    "K1",
    "O1",
    "P1",
    "Q1",
    "M4",
    "MS4",
    "MN4",
    "2N2",
    "MU2",
    "NU2",
    "L2",
    "T2",
    "J1",
    "OO1",
    "M6",
    "M8",
)


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
    # UTide 的时间单位是“天”。这里使用相对第一个观测时刻的天数，而不是 Matplotlib
    # date number。绝对日期数值很大时，重建潮汐可能退化成近似常数，导致 storm surge 标签错误。
    time_num = (work.index - work.index[0]).total_seconds() / 86400.0
    time_num = np.asarray(time_num, dtype=float)
    epoch = work.index[0].to_pydatetime()
    observed = work["sea_level"].to_numpy(dtype=float)

    # trend=False：此处按用户要求直接计算 predicted tide，然后用 observed - predicted tide。
    coef = solve(
        time_num,
        observed,
        lat=lat,
        method="ols",
        epoch=epoch,
        constit=TIDAL_CONSTITUENTS,
        # 预处理只需要调和常数和重建潮汐，不需要置信区间。
        # 关闭置信区间可避免 UTide 在规则小时数据上额外计算 periodogram 时
        # 产生无关的 divide-by-zero warning。
        conf_int="none",
        trend=False,
        verbose=False,
    )
    print(f"[TIDE] UTide 解出的分潮数量: {len(coef.name)}")
    print(f"[TIDE] UTide 分潮: {', '.join(str(name) for name in coef.name)}")
    if hasattr(coef, "A"):
        amp_preview = sorted(
            zip([str(name) for name in coef.name], np.asarray(coef.A, dtype=float)),
            key=lambda item: item[1],
            reverse=True,
        )[:8]
        print("[TIDE] UTide 主要分潮振幅: " + ", ".join(f"{name}={amp:.6f}" for name, amp in amp_preview))

    # 显式传入 constit=coef.name，避免 reconstruct 默认按 SNR/PE 过滤后只剩均值项。
    tide = reconstruct(time_num, coef, epoch=epoch, constit=coef.name, verbose=False).h

    out = pd.DataFrame(
        {
            "observed_sea_level": observed,
            "predicted_tide": tide,
            "storm_surge": observed - tide,
        },
        index=work.index,
    )

    print("[TIDE] predicted_tide describe:")
    print(out["predicted_tide"].describe())
    print("[TIDE] storm_surge describe:")
    print(out["storm_surge"].describe())

    before = len(out)
    out = out.loc[_robust_mad_filter(out["storm_surge"], SURGE_MAD_THRESHOLD)].copy()
    print(f"[TIDE] 宽松 MAD 删除 storm surge 异常值: {before - len(out):,}")
    print(f"[TIDE] storm surge 记录数: {len(out):,}")
    print(
        "[TIDE] storm_surge 范围: "
        f"min={out['storm_surge'].min():.6f}, "
        f"max={out['storm_surge'].max():.6f}, "
        f"mean={out['storm_surge'].mean():.6f}"
    )
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
    print(
        "[TIDE] daily_max_surge 范围: "
        f"min={out['daily_max_surge'].min():.6f}, "
        f"max={out['daily_max_surge'].max():.6f}, "
        f"mean={out['daily_max_surge'].mean():.6f}"
    )
    return out


# ==================== era20c_loader.py ====================
"""ERA-20C GRIB 文件读取、区域裁剪、插值和标准化。"""


from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr



def _find_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    """在 xarray Dataset 中查找坐标名。"""

    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"无法识别坐标名，候选: {candidates}，实际坐标: {list(ds.coords)}")


def _find_variable_name(ds: xr.Dataset, logical_name: str) -> str:
    """根据候选名称自动识别 ERA-20C 变量名。"""

    candidates = ERA20C_VARIABLE_CANDIDATES[logical_name]
    for name in candidates:
        if name in ds.data_vars:
            return name
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise KeyError(f"无法识别 {logical_name} 变量名，文件变量: {list(ds.data_vars)}")


def _to_numpy(da: xr.DataArray, dtype: str | np.dtype | None = None) -> np.ndarray:
    """兼容不同 xarray 版本的 DataArray 转 numpy。

    你实验室环境中的 xarray 版本不支持 ``DataArray.to_numpy(dtype=...)``，
    所以统一先调用 ``to_numpy()``，再用 numpy 的 ``astype`` 转类型。
    """

    arr = da.to_numpy()
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _as_datetime_index(values: xr.DataArray | np.ndarray) -> pd.DatetimeIndex:
    """把 xarray 时间坐标统一转换成 pandas.DatetimeIndex。

    cfgrib 读出的时间坐标有时是 numpy datetime64，有时带有 valid_time 坐标。
    后续构建样本时统一使用 pandas 时间，能避免精确 reindex 因类型差异全匹配失败。
    """

    raw_values = _to_numpy(values) if isinstance(values, xr.DataArray) else values
    return pd.DatetimeIndex(pd.to_datetime(raw_values))


def find_year_file(variable: str, year: int) -> Path | None:
    """在变量目录中自动寻找某一年的 GRIB 文件。"""

    folder = ERA20C_VARIABLE_DIRS[variable]
    if not folder.exists():
        return None
    patterns = (f"*{year}*.grb", f"*{year}*.grib", f"*{year}*")
    for pattern in patterns:
        matches = sorted(p for p in folder.glob(pattern) if p.is_file())
        if matches:
            return matches[0]
    return None


def cache_path_for_year(variable: str, year: int) -> Path:
    """返回裁剪插值后 ERA 年文件的本地缓存路径。"""

    return ERA20C_CACHE_DIR / f"xiamen_{variable}_{year}_{GRID_SIZE}x{GRID_SIZE}.nc"


def open_era20c_grib(path: Path, variable: str) -> xr.DataArray:
    """读取单个 ERA-20C GRIB 文件并返回目标变量 DataArray。"""

    try:
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )
    except Exception as exc:
        raise RuntimeError(
            f"无法读取 GRIB 文件: {path}\n"
            "请确认 cfgrib/eccodes 可用，且该 GRIB 文件没有损坏。"
        ) from exc

    var_name = _find_variable_name(ds, variable)
    da = ds[var_name]

    lat_name = _find_coord_name(ds, ("latitude", "lat"))
    lon_name = _find_coord_name(ds, ("longitude", "lon"))
    time_name = _find_coord_name(ds, ("time", "valid_time"))

    print(f"[ERA] 变量 {variable} 自动识别为: {var_name}")
    print(f"[ERA] 原始维度: {dict(da.sizes)}")

    # 统一坐标名，后续处理更简单。这里只重命名实际存在且不同名的坐标，
    # 避免 valid_time 与 time 同时存在时发生命名冲突。
    rename_map = {}
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if time_name != "time":
        rename_map[time_name] = "time"
    if rename_map:
        da = da.rename(rename_map)
    if "time" not in da.dims:
        raise ValueError(f"ERA 变量 {variable} 中没有 time 维度，实际维度: {da.dims}")

    # 如果文件中存在 valid_time，并且它沿 time 维度变化，则它通常是更明确的
    # 有效时间坐标。优先用 valid_time 覆盖 time，避免后续窗口筛选失败。
    if "valid_time" in da.coords and "time" in da["valid_time"].dims:
        da = da.assign_coords(time=_as_datetime_index(da["valid_time"]))
    else:
        da = da.assign_coords(time=_as_datetime_index(da["time"]))

    print(f"[ERA] 时间范围: {pd.Timestamp(da.time.values[0])} -> {pd.Timestamp(da.time.values[-1])}")
    return da


def subset_and_interp_station_region(
    da: xr.DataArray,
    site_lat: float = SITE_LAT,
    site_lon: float = SITE_LON,
    half_size: float = REGION_HALF_SIZE_DEG,
    grid_size: int = GRID_SIZE,
) -> xr.DataArray:
    """提取站点周围 10°×10° 区域，并插值到 40×40 网格。"""

    lon = da["lon"]
    # 如果 ERA 经度是 0-360，而站点经度也是正值，通常无需转换；
    # 这里仍保留兼容逻辑，避免遇到 -180~180 文件时报错。
    site_lon_for_data = site_lon
    if float(lon.max()) > 180 and site_lon < 0:
        site_lon_for_data = site_lon % 360

    lat_min = site_lat - half_size
    lat_max = site_lat + half_size
    lon_min = site_lon_for_data - half_size
    lon_max = site_lon_for_data + half_size

    # xarray slice 在纬度降序时需要反向。
    lat_values = _to_numpy(da["lat"])
    if lat_values[0] > lat_values[-1]:
        region = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    else:
        region = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    if region.sizes.get("lat", 0) < 2 or region.sizes.get("lon", 0) < 2:
        raise ValueError("站点周围区域格点过少，请检查经纬度或 ERA 文件坐标")

    print(
        "[ERA] 裁剪区域: "
        f"lat {lat_min:.3f}~{lat_max:.3f}, lon {lon_min:.3f}~{lon_max:.3f}; "
        f"原区域格点=({region.sizes.get('lat')}, {region.sizes.get('lon')})"
    )

    # 直接用理论边界插值时，若目标点略超出原始 ERA 网格范围，xarray 会在
    # 边缘产生 NaN。为了保证 CNN 输入不含 NaN，这里在已裁剪出的真实 ERA
    # 区域内部生成 40×40 目标网格。区域仍然是站点周围约 10°×10°。
    region_lat = _to_numpy(region["lat"], dtype="float64")
    region_lon = _to_numpy(region["lon"], dtype="float64")
    target_lat = np.linspace(float(np.min(region_lat)), float(np.max(region_lat)), grid_size)
    target_lon = np.linspace(float(np.min(region_lon)), float(np.max(region_lon)), grid_size)
    interpolated = region.interp(lat=target_lat, lon=target_lon)
    return interpolated.transpose("time", "lat", "lon")


@dataclass
class EraStats:
    """某个 ERA 变量在研究期内的均值和标准差。"""

    mean: float
    std: float


class Era20cReader:
    """按年份读取 ERA-20C，并为 CNN 样本提供标准化后的数组。"""

    def __init__(self, max_cache_items: int = 6) -> None:
        self.stats: dict[str, EraStats] = {}
        self.max_cache_items = max_cache_items
        self._cache: OrderedDict[tuple[str, int], xr.DataArray] = OrderedDict()
        self.missing_files: list[str] = []

    def _put_cache(self, key: tuple[str, int], value: xr.DataArray) -> None:
        """保存到 LRU 缓存，限制内存占用。"""

        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cache_items:
            self._cache.popitem(last=False)

    def available_years(self, years: range | list[int], skip_missing: bool) -> list[int]:
        """检查三种变量文件是否齐全，返回可用年份。"""

        available: list[int] = []
        for year in years:
            missing = [var for var in VARIABLE_ORDER if find_year_file(var, year) is None]
            if missing:
                msg = f"[ERA] {year} 缺少变量文件: {', '.join(missing)}"
                self.missing_files.append(msg)
                if skip_missing:
                    print(msg + "，跳过该年")
                    continue
                raise FileNotFoundError(msg)
            available.append(year)
        if not available:
            raise FileNotFoundError("没有找到任何 ERA-20C 可用年份")
        print(f"[ERA] 可用年份: {available[0]}-{available[-1]}，共 {len(available)} 年")
        return available

    def _load_raw_year(self, variable: str, year: int) -> xr.DataArray:
        """读取并插值某变量某一年，结果不做标准化。"""

        cache_key = (variable, year)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        cache_path = cache_path_for_year(variable, year)
        if cache_path.exists():
            print(f"[ERA] 读取缓存 {year} {variable}: {cache_path}")
            da = xr.open_dataarray(cache_path).load().astype("float32")
            self._put_cache(cache_key, da)
            return da

        path = find_year_file(variable, year)
        if path is None:
            raise FileNotFoundError(f"{year} 年 {variable} 文件不存在")

        print(f"[ERA] 读取 {year} {variable}: {path}")
        da = open_era20c_grib(path, variable)
        da = subset_and_interp_station_region(da)
        da = da.astype("float32")
        print(f"[ERA] {year} {variable} 插值后 shape: {tuple(da.shape)}")
        da.name = variable
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        da.to_netcdf(cache_path)
        print(f"[ERA] 写入缓存 {year} {variable}: {cache_path}")
        self._put_cache(cache_key, da)
        return da

    def compute_standardization(self, years: list[int]) -> None:
        """对 U10、V10、SLP 分别计算研究期均值和标准差。"""

        print("[ERA] 开始计算标准化参数...")
        for variable in VARIABLE_ORDER:
            total = 0.0
            total_sq = 0.0
            count = 0
            for year in years:
                arr = _to_numpy(self._load_raw_year(variable, year), dtype="float64")
                valid = np.isfinite(arr)
                values = arr[valid]
                if values.size == 0:
                    raise ValueError(f"{year} 年 {variable} 插值后没有任何有效值")
                total += values.sum()
                total_sq += np.square(values).sum()
                count += values.size
            if count == 0:
                raise ValueError(f"{variable} 没有任何有效 ERA 数据，无法标准化")
            mean = total / count
            variance = max(total_sq / count - mean * mean, 1e-12)
            std = float(np.sqrt(variance))
            self.stats[variable] = EraStats(float(mean), std)
            print(f"[ERA] {variable} mean={mean:.6f}, std={std:.6f}")

    def get_normalized_year(self, variable: str, year: int) -> xr.DataArray:
        """返回标准化后的某变量某一年 DataArray。"""

        if variable not in self.stats:
            raise RuntimeError("请先调用 compute_standardization() 计算标准化参数")
        raw = self._load_raw_year(variable, year)
        stats = self.stats[variable]
        return ((raw - stats.mean) / stats.std).astype("float32")

    def build_predictor_for_day(self, date: pd.Timestamp) -> np.ndarray | None:
        """为某一天 D 构建 CNN 输入，shape=(48, 40, 40)。

        通道顺序为：
        U10 的 16 个 3 小时时间片，随后 V10 的 16 个时间片，最后 SLP 的 16 个时间片。
        如果 D-1 到 D 的 ERA 时间片不足 16 个，返回 None。
        """

        date = pd.Timestamp(date).normalize()
        start = date - pd.Timedelta(days=1)
        end = date + pd.Timedelta(hours=21)

        channels: list[np.ndarray] = []
        for variable in VARIABLE_ORDER:
            pieces: list[xr.DataArray] = []
            for year in sorted({start.year, date.year}):
                pieces.append(self.get_normalized_year(variable, year))
            da = xr.concat(pieces, dim="time").sortby("time")
            _, unique_index = np.unique(_as_datetime_index(da["time"]), return_index=True)
            if len(unique_index) != da.sizes["time"]:
                da = da.isel(time=np.sort(unique_index)).sortby("time")

            # 对某一天 D，使用 D-1 00:00 到 D 21:00，共 16 个 3 小时时间片。
            # 这里用时间窗口筛选，比精确 reindex 更兼容不同 xarray/cfgrib 时间类型。
            selected = da.sel(time=slice(start, end))
            if selected.sizes.get("time", 0) != 16:
                return None
            selected_arr = _to_numpy(selected, dtype="float32")
            if not np.isfinite(selected_arr).all():
                return None
            channels.append(selected_arr)

        sample = np.concatenate(channels, axis=0)
        if sample.shape != (48, GRID_SIZE, GRID_SIZE):
            raise ValueError(f"样本 shape 异常: {sample.shape}")
        return sample

    def explain_missing_for_day(self, date: pd.Timestamp) -> str:
        """给出某一天无法构建样本时最常见的原因，便于日志排查。"""

        date = pd.Timestamp(date).normalize()
        start = date - pd.Timedelta(days=1)
        required_years = sorted({start.year, date.year})
        missing_files: list[str] = []
        for year in required_years:
            for variable in VARIABLE_ORDER:
                if find_year_file(variable, year) is None:
                    missing_files.append(f"{year}-{variable}")
        if missing_files:
            return "缺少 ERA 文件: " + ", ".join(missing_files)
        return "ERA 时间片不足或插值后包含 NaN"


# ==================== dataset_builder.py ====================
"""CNN 可用数据集构建与训练/验证划分。

本文件负责把 daily maximum storm surge 标签和 ERA-20C 气象场对齐，
并保存成训练脚本可以直接读取的 .npy 文件。
"""


import json
from pathlib import Path

import numpy as np
import pandas as pd



def tile_time_slices(variable_slices: np.ndarray) -> np.ndarray:
    """把某个变量的 16 个 40×40 时间片拼成 160×160 大图。

    作者 notebook 的 ``array_reorganization`` 使用 4×4 拼图：前 4 个时间片横向拼成第一行，
    5-8 拼成第二行，依次类推。这样时间信息进入空间维度，CNN 输入通道数就从 48 变为 3。
    """

    expected_shape = (STEPS_PER_SAMPLE, GRID_SIZE, GRID_SIZE)
    if variable_slices.shape != expected_shape:
        raise ValueError(f"变量时间片 shape 异常: {variable_slices.shape}, 期望 {expected_shape}")
    rows = []
    for row in range(TIME_TILE_ROWS):
        start = row * TIME_TILE_COLS
        end = start + TIME_TILE_COLS
        rows.append(np.concatenate(variable_slices[start:end], axis=1))
    return np.concatenate(rows, axis=0).astype("float32")


def reorganize_sample_to_notebook_layout(sample: np.ndarray) -> np.ndarray:
    """把 ERA 样本从 (48,40,40) 转为作者 notebook 风格的 (3,160,160)。"""

    expected_channels = STEPS_PER_SAMPLE * len(VARIABLE_ORDER)
    if sample.shape != (expected_channels, GRID_SIZE, GRID_SIZE):
        raise ValueError(f"原始样本 shape 异常: {sample.shape}")
    tiled_variables = []
    for var_i, _ in enumerate(VARIABLE_ORDER):
        start = var_i * STEPS_PER_SAMPLE
        end = start + STEPS_PER_SAMPLE
        tiled_variables.append(tile_time_slices(sample[start:end]))
    return np.stack(tiled_variables, axis=0).astype("float32")


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

    train_shape = (len(train_indices), INPUT_CHANNELS, MODEL_GRID_SIZE, MODEL_GRID_SIZE)
    val_shape = (len(val_indices), INPUT_CHANNELS, MODEL_GRID_SIZE, MODEL_GRID_SIZE)
    print(f"[DATASET] 总样本数: {n_samples:,}")
    print(f"[DATASET] X 总 shape: ({n_samples}, {INPUT_CHANNELS}, {MODEL_GRID_SIZE}, {MODEL_GRID_SIZE})")
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
        x_train[write_i] = reorganize_sample_to_notebook_layout(sample)
        if (write_i + 1) % 100 == 0 or write_i + 1 == len(train_indices):
            print(f"[DATASET] 已写入训练样本 {write_i + 1:,}/{len(train_indices):,}")

    for write_i, sample_i in enumerate(val_indices):
        date = sample_dates[int(sample_i)]
        sample = era_reader.build_predictor_for_day(date)
        if sample is None:
            raise RuntimeError(f"第二次构建验证样本时失败: {date}")
        x_val[write_i] = reorganize_sample_to_notebook_layout(sample)
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
                "x_layout": "notebook-style tiled time slices, shape=(N, 3, 160, 160)",
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


# ==================== preprocess_xiamen.py ====================
"""厦门站 Xiamen 预处理命令行入口。

示例：
python src/preprocess_xiamen.py --start-year 1985 --end-year 1985
python src/preprocess_xiamen.py --all-years
"""


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



def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="预处理厦门站 GESLA + ERA20C，生成 CNN 可用数据集。")
    parser.add_argument("--all-years", action="store_true", help="使用厦门站全部可用年份 1954-1997")
    parser.add_argument("--start-year", type=int, help="开始年份，例如 1985")
    parser.add_argument("--end-year", type=int, help="结束年份，例如 1985")
    parser.add_argument(
        "--split-mode",
        choices=["auto", "first-years", "chronological"],
        default="auto",
        help=(
            "训练/验证划分方式。auto 会在多年数据上使用论文式前 5 年验证，"
            "单年测试时自动退回 80/20；first-years 强制前若干年验证；"
            "chronological 强制按时间顺序 80/20。"
        ),
    )
    parser.add_argument("--validation-years", type=int, default=5, help="论文式划分中用于验证集的最早年份数量")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="chronological 划分时训练集比例")
    parser.add_argument(
        "--skip-missing-era",
        action="store_true",
        help="如果某一年 ERA 文件缺失，则跳过该年；默认缺失即报错。",
    )
    return parser.parse_args()


def resolve_years(args: argparse.Namespace) -> tuple[int, int]:
    """根据命令行参数确定年份范围。"""

    if args.all_years:
        return XIAMEN_START_YEAR, XIAMEN_END_YEAR
    if args.start_year is None or args.end_year is None:
        raise SystemExit("请使用 --all-years，或同时提供 --start-year 和 --end-year")
    if args.start_year > args.end_year:
        raise SystemExit("--start-year 不能大于 --end-year")
    return args.start_year, args.end_year


def main() -> None:
    """执行完整预处理流程。"""

    args = parse_args()
    start_year, end_year = resolve_years(args)
    years = list(range(start_year, end_year + 1))

    print("=" * 80)
    print(f"[RUN] 站点: {SITE_NAME}")
    print(f"[RUN] 年份范围: {start_year}-{end_year}")
    print(f"[RUN] GESLA 文件: {SITE_FILE}")
    print(f"[RUN] 输出目录: {XIAMEN_OUTPUT_DIR}")
    print("=" * 80)

    # 1. GESLA 读取与按年份裁剪。
    gesla = read_gesla_file(SITE_FILE)
    gesla = restrict_years(gesla, start_year, end_year)

    # 2. UTide 潮汐分离，得到 hourly/sub-hourly storm surge。
    surge = separate_tide_with_utide(gesla, lat=SITE_LAT)
    XIAMEN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    surge.to_csv(XIAMEN_OUTPUT_DIR / "cleaned_surge.csv", encoding="utf-8-sig")
    print(f"[SAVE] cleaned_surge.csv: {XIAMEN_OUTPUT_DIR / 'cleaned_surge.csv'}")

    # 3. 按天取 daily maximum storm surge，得到标签 y。
    daily = daily_maximum_surge(surge)
    daily.to_csv(XIAMEN_OUTPUT_DIR / "daily_max_surge.csv", encoding="utf-8-sig")
    print(f"[SAVE] daily_max_surge.csv: {XIAMEN_OUTPUT_DIR / 'daily_max_surge.csv'}")

    # 4. ERA-20C 可用年份检查、标准化参数计算。
    era_reader = Era20cReader()
    available_years = era_reader.available_years(years, skip_missing=args.skip_missing_era)
    era_reader.compute_standardization(available_years)

    # 5. 构建 CNN 样本并按时间顺序划分。
    sample_dates, y = collect_available_samples(daily, era_reader)
    save_train_val_arrays(
        sample_dates,
        y,
        era_reader,
        XIAMEN_OUTPUT_DIR,
        train_ratio=args.train_ratio,
        split_mode=args.split_mode,
        validation_years=args.validation_years,
    )

    print("=" * 80)
    print("[DONE] 厦门站预处理完成。")
    print(f"[DONE] 输出目录: {XIAMEN_OUTPUT_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
