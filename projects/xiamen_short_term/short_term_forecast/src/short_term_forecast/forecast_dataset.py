"""厦门站短时风暴潮预报数据集构建。

默认实现 hourly forecast：
前 t 个小时的 U10/V10/SLP 40×40 网格 + 前 t 个小时的 storm surge
预测下一小时 storm surge。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

import config


def _path_exists(path: Path) -> bool:
    return path.exists() if isinstance(path, Path) else Path(path).exists()


def frequency_to_pandas_rule(frequency: str) -> str:
    """把命令行频率转换成 pandas/xarray 可用的重采样频率。"""

    if frequency == "hourly":
        return "1h"
    raise ValueError("短时预报模块固定使用 hourly 频率")


def frequency_to_timedelta(frequency: str) -> pd.Timedelta:
    """返回单步预报对应的时间间隔。"""

    return pd.Timedelta(frequency_to_pandas_rule(frequency))


def resolve_site_file() -> Path:
    """自动寻找厦门站 GESLA 文件，并打印实际使用路径。"""

    candidates = [config.SITE_FILE, config.LOCAL_SITE_FILE]
    for path in candidates:
        if _path_exists(path):
            print(f"[DATA] 使用 GESLA 站点文件: {path}")
            return Path(path)
    raise FileNotFoundError(
        "没有找到厦门站 GESLA 文件。已检查:\n"
        + "\n".join(f"- {p}" for p in candidates)
    )


def resolve_era_root(data_source: str) -> Path:
    """根据数据源自动选择 ERA 根目录。"""

    source = data_source.upper()
    if source == "ERA5":
        # ERA5 优先使用已经按厦门站整理好的站点级 NetCDF。
        # 实验室当前目录中 F:\ERA5-NEW\Xiamen 下有
        # xiamen_10u_1970_1997.nc / xiamen_v10_1970_1997.nc /
        # xiamen_slp_1970_1997.nc，这类文件通常已经是 40×40 网格。
        candidates = [
            config.ERA5_NEW_DIR / "Xiamen",
            config.ERA5_NEW_DIR / "xiamen",
            config.ERA5_DIR / "xiamen",
            config.ERA5_DIR / "Xiamen",
            config.ERA5_NEW_DIR,
            config.ERA5_DIR,
            config.ERA5_ALL_DIR,
        ]
    else:
        raise ValueError("短时预报模块固定使用 ERA5")

    print(f"[DATA] 请求数据源: {source}")
    for path in candidates:
        print(f"[DATA] 检查 ERA 目录: {path}")
        if _path_exists(path):
            print(f"[DATA] 使用 ERA 数据目录: {path}")
            return Path(path)
    raise FileNotFoundError(
        f"没有找到 {source} 数据目录。已检查:\n"
        + "\n".join(f"- {p}" for p in candidates)
    )


def read_gesla_file(path: Path) -> pd.DataFrame:
    """读取 GESLA 文本文件，尽量兼容多种日期格式。"""

    print(f"[GESLA] 读取文件: {path}")
    try:
        df = pd.read_csv(
            path,
            comment="#",
            sep=r"\s+",
            header=None,
            names=["date", "time", "sea_level", "qc_flag", "use_flag"],
            usecols=[0, 1, 2, 3, 4],
            engine="python",
        )
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            format="%Y/%m/%d %H:%M:%S",
            errors="coerce",
        )
        use_flag = pd.to_numeric(df["use_flag"], errors="coerce")
        if use_flag.notna().any():
            before = len(df)
            df = df.loc[use_flag == 1].copy()
            print(f"[GESLA] use_flag 过滤无效记录: {before - len(df):,}")
        df = df[["datetime", "sea_level"]]
    except Exception:
        rows: list[tuple[pd.Timestamp, float, float | None]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                tokens = raw.strip().split()
                if not tokens or raw.lstrip().startswith("#") or not tokens[0][0].isdigit():
                    continue
                parsed = _parse_gesla_line(tokens)
                if parsed is not None:
                    rows.append(parsed)
        if not rows:
            raise ValueError(f"没有在 GESLA 文件中识别到数据行: {path}")
        df = pd.DataFrame(rows, columns=["datetime", "sea_level", "use_flag"])
        use_flag = pd.to_numeric(df["use_flag"], errors="coerce")
        if use_flag.notna().any():
            before = len(df)
            df = df.loc[use_flag == 1].copy()
            print(f"[GESLA] use_flag 过滤无效记录: {before - len(df):,}")
        df = df[["datetime", "sea_level"]]

    df = df.dropna().drop_duplicates("datetime").sort_values("datetime")
    df["sea_level"] = pd.to_numeric(df["sea_level"], errors="coerce")
    bad = df["sea_level"].isin(config.MISSING_VALUE_MARKERS) | df["sea_level"].isna()
    df = df.loc[~bad & (df["sea_level"].abs() <= config.SEA_LEVEL_ABS_LIMIT)].copy()
    df = df.set_index("datetime").sort_index()
    print(f"[GESLA] 清洗后记录数: {len(df):,}")
    print(f"[GESLA] 时间范围: {df.index.min()} -> {df.index.max()}")
    return df


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _parse_gesla_line(tokens: list[str]) -> tuple[pd.Timestamp, float, float | None] | None:
    """解析一行 GESLA 数据。"""

    patterns = []
    if len(tokens) >= 3:
        patterns.append((f"{tokens[0]} {tokens[1]}", 2))
    if len(tokens) >= 6 and all(_is_float(t) for t in tokens[:5]):
        y, m, d, h, minute = [int(float(t)) for t in tokens[:5]]
        patterns.append((dict(year=y, month=m, day=d, hour=h, minute=minute), 5))
    if len(tokens) >= 5 and all(_is_float(t) for t in tokens[:4]):
        y, m, d, h = [int(float(t)) for t in tokens[:4]]
        patterns.append((dict(year=y, month=m, day=d, hour=h), 4))

    for candidate, consumed in patterns:
        dt = pd.to_datetime(candidate, errors="coerce")
        if pd.notna(dt) and consumed < len(tokens) and _is_float(tokens[consumed]):
            # GESLA 常见格式为 date time sea_level qc_flag use_flag。
            # 对数字拆分日期格式，sea_level 后通常也是 qc_flag、use_flag。
            use_flag_index = consumed + 2
            use_flag = float(tokens[use_flag_index]) if use_flag_index < len(tokens) and _is_float(tokens[use_flag_index]) else None
            return pd.Timestamp(dt), float(tokens[consumed]), use_flag
    return None


def _robust_mad_filter(series: pd.Series, threshold: float) -> pd.Series:
    values = series.to_numpy(dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(True, index=series.index)
    robust_z = 0.6745 * (values - median) / mad
    return pd.Series(np.abs(robust_z) <= threshold, index=series.index)


def _separate_tide_with_utide(df: pd.DataFrame) -> pd.DataFrame:
    """用 UTide 生成 storm surge；UTide 不可用时给出明确报错。"""

    try:
        from utide import reconstruct, solve
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("需要安装 utide 才能从 GESLA 原始潮位生成 storm surge") from exc

    work = df[["sea_level"]].dropna().copy()
    if len(work) < 24 * 30:
        raise ValueError("可用于 UTide 的记录太少，至少建议 1 个月以上")

    print("[TIDE] 未发现已清洗 CSV，开始用 UTide 从 GESLA 生成 storm surge...")
    time_num = (work.index - work.index[0]).total_seconds() / 86400.0
    coef = solve(
        np.asarray(time_num, dtype=float),
        work["sea_level"].to_numpy(dtype=float),
        lat=config.SITE_LAT,
        method="ols",
        epoch=work.index[0].to_pydatetime(),
        conf_int="none",
        trend=False,
        verbose=False,
    )
    tide = reconstruct(
        np.asarray(time_num, dtype=float),
        coef,
        epoch=work.index[0].to_pydatetime(),
        constit=coef.name,
        verbose=False,
    ).h
    out = pd.DataFrame(
        {
            "observed_sea_level": work["sea_level"].to_numpy(dtype=float),
            "predicted_tide": tide,
            "storm_surge": work["sea_level"].to_numpy(dtype=float) - tide,
        },
        index=work.index,
    )
    before = len(out)
    out = out.loc[_robust_mad_filter(out["storm_surge"], config.SURGE_MAD_THRESHOLD)].copy()
    print(f"[TIDE] 宽松 MAD 删除 storm surge 异常值: {before - len(out):,}")
    return out


def load_surge_series(start_year: int, end_year: int, frequency: str = config.FORECAST_FREQUENCY) -> pd.DataFrame:
    """读取或生成小时级 storm surge 序列。

    hourly：保留 GESLA/UTide 得到的小时级 storm surge，用前 t 小时预测下一小时。
    """

    frequency_to_pandas_rule(frequency)
    cleaned_candidates = [
        config.FORECAST_OUTPUT_ROOT / "cleaned_surge.csv",
        config.XIAMEN_OUTPUT_DIR / "cleaned_surge.csv",
        config.REPO_ROOT / "code_my" / "xiamen" / "outputs" / "xiamen" / "cleaned_surge.csv",
    ]
    csv_candidates = cleaned_candidates
    for path in csv_candidates:
        if path.exists():
            print(f"[SURGE] 使用已有 CSV: {path}")
            df = pd.read_csv(path)
            date_col = "date" if "date" in df.columns else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
            if "storm_surge" not in df.columns:
                continue
            series = df["storm_surge"].resample(frequency_to_pandas_rule(frequency)).mean()
            out = series.dropna().rename("storm_surge").to_frame()
            return _restrict_surge_years(out, start_year, end_year, frequency)

    site_file = resolve_site_file()
    raw = read_gesla_file(site_file)
    raw = raw.loc[
        (raw.index >= pd.Timestamp(start_year, 1, 1))
        & (raw.index <= pd.Timestamp(end_year, 12, 31, 23, 59, 59))
    ].copy()
    if raw.empty:
        raise ValueError(f"GESLA 在 {start_year}-{end_year} 内没有记录")
    surge = _separate_tide_with_utide(raw)
    series = surge["storm_surge"].resample(frequency_to_pandas_rule(frequency)).mean()
    out = series.dropna().rename("storm_surge").to_frame()
    return _restrict_surge_years(out, start_year, end_year, frequency)


def _restrict_surge_years(df: pd.DataFrame, start_year: int, end_year: int, frequency: str) -> pd.DataFrame:
    out = df.loc[
        (df.index >= pd.Timestamp(start_year, 1, 1))
        & (df.index <= pd.Timestamp(end_year, 12, 31, 23, 59, 59))
    ].copy()
    if out.empty:
        raise ValueError(f"{start_year}-{end_year} 没有 {frequency} storm surge 标签")
    out.index = pd.DatetimeIndex(out.index)
    print(f"[SURGE] {frequency} 标签数: {len(out):,}")
    print(f"[SURGE] 时间范围: {out.index.min()} -> {out.index.max()}")
    return out


def _to_datetime_index(values: xr.DataArray | np.ndarray) -> pd.DatetimeIndex:
    raw = values.to_numpy() if isinstance(values, xr.DataArray) else values
    return pd.DatetimeIndex(pd.to_datetime(raw))


def _find_coord_name(ds: xr.Dataset | xr.DataArray, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"无法识别坐标名，候选 {tuple(candidates)}，实际坐标 {list(ds.coords)}")


def _find_variable_name(ds: xr.Dataset, logical_name: str) -> str:
    for name in config.ERA_VARIABLE_CANDIDATES[logical_name]:
        if name in ds.data_vars:
            return name
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise KeyError(f"无法识别 {logical_name} 变量名，文件变量: {list(ds.data_vars)}")


def _filename_covers_year(filename: str, year: int) -> bool:
    """判断文件名是否包含目标年份，或包含覆盖目标年份的起止年份范围。"""

    if str(year) in filename:
        return True
    years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", filename)]
    if len(years) >= 2:
        start_year = min(years)
        end_year = max(years)
        return start_year <= year <= end_year
    return False


def _open_era_file(path: Path, variable: str) -> xr.DataArray:
    """打开 NetCDF 文件并返回目标变量。"""

    suffix = path.suffix.lower()
    try:
        if suffix not in {".nc", ".netcdf"}:
            raise ValueError("短时预报模块只读取 ERA5 NetCDF 文件")
        ds = xr.open_dataset(path)
    except Exception as exc:
        raise RuntimeError(f"无法读取 ERA 文件: {path}") from exc

    var_name = _find_variable_name(ds, variable)
    da = ds[var_name]
    lat_name = _find_coord_name(da, ("latitude", "lat"))
    lon_name = _find_coord_name(da, ("longitude", "lon"))
    time_name = _find_coord_name(da, ("time", "valid_time"))

    rename = {}
    if lat_name != "lat":
        rename[lat_name] = "lat"
    if lon_name != "lon":
        rename[lon_name] = "lon"
    if time_name != "time":
        rename[time_name] = "time"
    if rename:
        da = da.rename(rename)
    if "valid_time" in da.coords and "time" in da["valid_time"].dims:
        da = da.assign_coords(time=_to_datetime_index(da["valid_time"]))
    else:
        da = da.assign_coords(time=_to_datetime_index(da["time"]))
    print(f"[ERA] {path.name} 变量 {variable} 自动识别为 {var_name}，原始维度 {dict(da.sizes)}")
    return da


def subset_and_resample_grid(da: xr.DataArray, grid_size: int = config.GRID_SIZE) -> xr.DataArray:
    """裁剪站点周围区域，并按需插值/重采样到 40×40。"""

    lon = da["lon"]
    site_lon = config.SITE_LON
    if float(lon.max()) > 180 and site_lon < 0:
        site_lon = site_lon % 360
    lat_min = config.SITE_LAT - config.REGION_HALF_SIZE_DEG
    lat_max = config.SITE_LAT + config.REGION_HALF_SIZE_DEG
    lon_min = site_lon - config.REGION_HALF_SIZE_DEG
    lon_max = site_lon + config.REGION_HALF_SIZE_DEG

    lat_values = da["lat"].to_numpy()
    if lat_values[0] > lat_values[-1]:
        region = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    else:
        region = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    original_shape = (int(region.sizes.get("lat", 0)), int(region.sizes.get("lon", 0)))
    print(f"[ERA] 裁剪后原始网格大小: {original_shape}")
    if original_shape[0] < 2 or original_shape[1] < 2:
        raise ValueError("站点周围区域格点过少，请检查经纬度或 ERA 文件坐标")

    if original_shape == (grid_size, grid_size):
        print("[ERA] 是否执行插值: 否，裁剪后已经是 40×40")
        out = region
    else:
        region_lat = region["lat"].to_numpy().astype("float64")
        region_lon = region["lon"].to_numpy().astype("float64")
        target_lat = np.linspace(float(np.min(region_lat)), float(np.max(region_lat)), grid_size)
        target_lon = np.linspace(float(np.min(region_lon)), float(np.max(region_lon)), grid_size)
        print("[ERA] 是否执行插值: 是，重采样到 40×40")
        out = region.interp(lat=target_lat, lon=target_lon)

    out = out.transpose("time", "lat", "lon").astype("float32")
    print(f"[ERA] 最终网格大小: ({out.sizes['lat']}, {out.sizes['lon']})")
    return out


@dataclass
class EraHourlyFieldStore:
    """缓存 ERA5 小时级网格，供样本构建和滚动预测复用。"""

    data_source: str
    start_year: int
    end_year: int
    variables: list[str]
    frequency: str = config.FORECAST_FREQUENCY
    era_root: Path | None = None

    def __post_init__(self) -> None:
        self.era_root = self.era_root or resolve_era_root(self.data_source)
        self.fields: dict[str, xr.DataArray] = {}

    def load(self) -> None:
        for variable in self.variables:
            paths: list[Path] = []
            for year in range(self.start_year, self.end_year + 1):
                path = self.find_year_file(variable, year)
                if path is None:
                    raise FileNotFoundError(f"[ERA] {year} 年缺少 {variable} 文件，数据源目录: {self.era_root}")
                paths.append(path)

            unique_paths = list(dict.fromkeys(paths))
            pieces: list[xr.DataArray] = []
            for path in unique_paths:
                covered_years = [
                    str(year)
                    for year in range(self.start_year, self.end_year + 1)
                    if paths[year - self.start_year] == path
                ]
                year_text = covered_years[0] if len(covered_years) == 1 else f"{covered_years[0]}-{covered_years[-1]}"
                print(f"[ERA] 读取 {year_text} {variable}: {path}")
                pieces.append(subset_and_resample_grid(_open_era_file(path, variable)))

            merged = xr.concat(pieces, dim="time").sortby("time") if len(pieces) > 1 else pieces[0].sortby("time")

            # 站点级 ERA5 文件可能一次覆盖 1970-1997。这里只保留实验所需时间段，
            # 避免多年训练时把同一个大文件重复拼接进内存。
            start_time = pd.Timestamp(self.start_year, 1, 1)
            end_time = pd.Timestamp(self.end_year, 12, 31, 23, 59, 59)
            merged = merged.sel(time=slice(start_time, end_time))

            _, unique_index = np.unique(_to_datetime_index(merged["time"]), return_index=True)
            if len(unique_index) != merged.sizes["time"]:
                merged = merged.isel(time=np.sort(unique_index)).sortby("time")
            rule = frequency_to_pandas_rule(self.frequency)
            field = merged.resample(time=rule).mean(skipna=True).dropna("time", how="any")
            self.fields[variable] = field.astype("float32")
            print(f"[ERA] {variable} {self.frequency} 场 shape: {tuple(field.shape)}")

    def find_year_file(self, variable: str, year: int) -> Path | None:
        """自动探测某变量某年的 ERA 文件。"""

        assert self.era_root is not None
        search_roots = [self.era_root]

        extensions = (".nc", ".netcdf")
        hints = config.ERA_FILE_HINTS[variable]
        matches: list[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in extensions:
                    continue
                lower = path.name.lower()
                if any(h in lower for h in hints) and _filename_covers_year(lower, year):
                    matches.append(path)
        if matches:
            return sorted(set(matches), key=lambda p: (len(str(p)), str(p)))[0]
        return None

    def build_atmosphere_window(self, dates: list[pd.Timestamp]) -> np.ndarray:
        """返回 shape=(t*3, 40, 40) 的大气场窗口。"""

        channels: list[np.ndarray] = []
        for date in dates:
            date = pd.Timestamp(date)
            for variable in self.variables:
                da = self.fields[variable].sel(time=date)
                arr = da.to_numpy().astype("float32")
                if arr.shape != (config.GRID_SIZE, config.GRID_SIZE):
                    raise ValueError(f"{date.date()} {variable} 网格 shape 异常: {arr.shape}")
                channels.append(arr)
        sample = np.stack(channels, axis=0)
        if not np.isfinite(sample).all():
            raise ValueError("大气场窗口包含 NaN/Inf")
        return sample


def build_forecast_arrays(
    data_source: str,
    start_year: int,
    end_year: int,
    input_steps: int,
    horizon: int = config.FORECAST_HORIZON,
    frequency: str = config.FORECAST_FREQUENCY,
    variables: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """构建指定频率的 forecast 数组。"""

    if horizon != 1:
        raise ValueError("当前方案B短时滚动预报训练阶段请使用 horizon=1，多步预报请用 rolling_forecast.py")

    variables = variables or list(config.VARIABLES)
    surge = load_surge_series(start_year, end_year, frequency=frequency)
    era = EraHourlyFieldStore(data_source, start_year, end_year, variables, frequency=frequency)
    era.load()

    x_atm: list[np.ndarray] = []
    x_surge: list[np.ndarray] = []
    y: list[float] = []
    target_dates: list[pd.Timestamp] = []
    dates = pd.DatetimeIndex(surge.index).sort_values()
    freq_rule = frequency_to_pandas_rule(frequency)

    for i in range(input_steps, len(dates) - horizon + 1):
        hist_dates = [pd.Timestamp(d) for d in dates[i - input_steps : i]]
        target_date = pd.Timestamp(dates[i + horizon - 1])
        expected = pd.date_range(hist_dates[0], target_date, freq=freq_rule)
        if len(expected) != input_steps + horizon or not expected.equals(pd.DatetimeIndex(hist_dates + [target_date])):
            continue
        try:
            atm = era.build_atmosphere_window(hist_dates)
        except Exception as exc:
            print(f"[DATASET] 跳过 {target_date}: {str(exc).splitlines()[0]}")
            continue
        surge_hist = surge.loc[hist_dates, "storm_surge"].to_numpy(dtype="float32")
        target = float(surge.loc[target_date, "storm_surge"])
        if not np.isfinite(surge_hist).all() or not np.isfinite(target):
            continue
        x_atm.append(atm)
        x_surge.append(surge_hist)
        y.append(target)
        target_dates.append(target_date)

    if not y:
        raise ValueError("没有生成任何短时预报样本，请检查 ERA 与 GESLA 时间范围是否重叠")

    print(f"[DATASET] 样本数: {len(y):,}")
    print(f"[DATASET] atmosphere shape: ({len(y)}, {input_steps * len(variables)}, 40, 40)")
    print(f"[DATASET] surge_history shape: ({len(y)}, {input_steps})")
    print(f"[DATASET] y shape: ({len(y)},)")
    return (
        np.stack(x_atm).astype("float32"),
        np.stack(x_surge).astype("float32"),
        np.asarray(y, dtype="float32"),
        np.asarray(target_dates, dtype="datetime64[ns]"),
        pd.DatetimeIndex(target_dates),
    )


def chronological_split(n_samples: int, train_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """按时间顺序划分 train/val。"""

    split = int(n_samples * train_ratio)
    if split <= 0 or split >= n_samples:
        raise ValueError("样本数量太少，无法按时间顺序划分训练集和验证集")
    return np.arange(split), np.arange(split, n_samples)


def standardize_train_val(
    x_atm: np.ndarray,
    x_surge: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """用训练集统计量标准化 X 和 y。"""

    atm_mean = x_atm[train_idx].mean(axis=(0, 2, 3), keepdims=True)
    atm_std = x_atm[train_idx].std(axis=(0, 2, 3), keepdims=True)
    atm_std = np.where(atm_std < 1e-6, 1.0, atm_std)
    surge_mean = float(x_surge[train_idx].mean())
    surge_std = float(x_surge[train_idx].std() or 1.0)
    y_mean = float(y[train_idx].mean())
    y_std = float(y[train_idx].std() or 1.0)

    x_atm_std = ((x_atm - atm_mean) / atm_std).astype("float32")
    x_surge_std = ((x_surge - surge_mean) / surge_std).astype("float32")
    y_std_arr = ((y - y_mean) / y_std).astype("float32")

    scalers = {
        "surge_history_mean": surge_mean,
        "surge_history_std": surge_std,
        "target_mean": y_mean,
        "target_std": y_std,
        "atmosphere_channel_mean": atm_mean.reshape(-1).astype(float).tolist(),
        "atmosphere_channel_std": atm_std.reshape(-1).astype(float).tolist(),
    }
    print(f"[DATASET] y 标准化: mean={y_mean:.6f}, std={y_std:.6f}")
    return x_atm_std, x_surge_std, y_std_arr, scalers
