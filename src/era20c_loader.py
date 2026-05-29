"""ERA-20C GRIB 文件读取、区域裁剪、插值和标准化。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from config import (
    ERA20C_VARIABLE_CANDIDATES,
    ERA20C_CACHE_DIR,
    ERA20C_VARIABLE_DIRS,
    GRID_SIZE,
    REGION_HALF_SIZE_DEG,
    SITE_LAT,
    SITE_LON,
    VARIABLE_ORDER,
)


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
