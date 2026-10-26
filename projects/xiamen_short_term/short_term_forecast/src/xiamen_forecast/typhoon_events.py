"""历史台风最佳路径筛选与固定起报事件评价。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .station_config import StationConfig


IBTRACS_SOURCE = "https://www.ncei.noaa.gov/products/international-best-track-archive"
IBTRACS_WP_CSV = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.WP.list.v04r01.csv"
)
ORIGIN_OFFSETS_HOURS = (-72, -48, -24)
EVALUATION_HOURS = (24, 48, 72)
TRACK_COLUMNS = (
    "SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON", "BASIN",
    "WMO_WIND", "WMO_PRES", "TRACK_TYPE",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ibtracs_wp(path: Path, year: int | tuple[int, int] = 1997) -> pd.DataFrame:
    """读取NOAA CSV，跳过官方文件的第二行单位说明。"""
    start_year, end_year = (year, year) if isinstance(year, int) else year
    if start_year > end_year:
        raise ValueError("IBTrACS year range must be increasing")
    if not path.is_file():
        raise FileNotFoundError(f"IBTrACS best-track CSV not found: {path}")
    header = pd.read_csv(path, nrows=0).columns
    required = {"SID", "NAME", "ISO_TIME", "LAT", "LON", "BASIN"}
    if missing := required - set(header):
        raise ValueError(f"IBTrACS CSV lacks columns: {sorted(missing)}")
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=[name for name in TRACK_COLUMNS if name in header],
        skiprows=[1],
        chunksize=50000,
        low_memory=False,
    ):
        chunk["time"] = pd.to_datetime(chunk["ISO_TIME"], utc=True, errors="coerce")
        chunk = chunk.loc[
            chunk.time.dt.year.between(start_year, end_year)
            & (chunk.BASIN.astype(str).str.strip() == "WP")
        ].copy()
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        raise ValueError(f"No western North Pacific best-track points in {year}: {path}")
    tracks = pd.concat(parts, ignore_index=True)
    if "TRACK_TYPE" in tracks:
        tracks = tracks.loc[tracks.TRACK_TYPE.astype(str).str.upper().eq("MAIN")].copy()
    for field in ("LAT", "LON", "WMO_WIND", "WMO_PRES"):
        if field in tracks:
            tracks[field] = pd.to_numeric(tracks[field], errors="coerce")
    tracks = tracks.dropna(subset=["SID", "time", "LAT", "LON"])
    tracks = tracks.loc[tracks.LAT.between(-90, 90) & tracks.LON.between(-360, 360)]
    tracks["time"] = tracks.time.dt.tz_convert("UTC").dt.tz_localize(None)
    tracks = tracks.sort_values(["SID", "time"]).drop_duplicates(["SID", "time"])
    if tracks.empty:
        raise ValueError(f"No usable main-track points in {year}: {path}")
    return tracks.reset_index(drop=True)


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """两点球面距离，允许经度使用0–360或-180–180。"""
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(np.asarray(lat2, dtype=float))
    dp = p2 - p1
    dl = np.deg2rad((np.asarray(lon2, dtype=float) - lon1 + 180) % 360 - 180)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def closest_hourly_approach(track: pd.DataFrame, station: StationConfig) -> tuple[pd.Timestamp, float]:
    """在相邻最佳路径点之间逐小时线性插值；不跨越12小时以上缺口。"""
    ordered = track.sort_values("time").reset_index(drop=True)
    sample_times: list[pd.Timestamp] = []
    sample_lat: list[float] = []
    sample_lon: list[float] = []
    for position, row in ordered.iterrows():
        sample_times.append(pd.Timestamp(row.time))
        sample_lat.append(float(row.LAT))
        sample_lon.append(float(row.LON))
        if position + 1 == len(ordered):
            continue
        nxt = ordered.iloc[position + 1]
        hours = (pd.Timestamp(nxt.time) - pd.Timestamp(row.time)).total_seconds() / 3600
        if not 1 < hours <= 12:
            continue
        longitude_change = (float(nxt.LON) - float(row.LON) + 180) % 360 - 180
        for step in range(1, int(np.floor(hours)) + 1):
            timestamp = pd.Timestamp(row.time) + pd.Timedelta(hours=step)
            if timestamp >= pd.Timestamp(nxt.time):
                break
            fraction = step / hours
            sample_times.append(timestamp)
            sample_lat.append(float(row.LAT) + fraction * (float(nxt.LAT) - float(row.LAT)))
            sample_lon.append(float(row.LON) + fraction * longitude_change)
    distances = haversine_km(
        station.latitude, station.longitude, np.asarray(sample_lat), np.asarray(sample_lon)
    )
    nearest = int(np.argmin(distances))
    return sample_times[nearest], float(distances[nearest])


def build_candidates(
    tracks: pd.DataFrame,
    stations: list[StationConfig],
    max_distance_km: float = 500.0,
) -> pd.DataFrame:
    """路径距离仅用于初筛，候选事件仍须人工核验归因。"""
    if max_distance_km <= 0:
        raise ValueError("max_distance_km must be positive")
    rows = []
    for sid, track in tracks.groupby("SID", sort=True):
        names = track.NAME.dropna().astype(str).str.strip()
        names = names.loc[~names.str.upper().isin(("", "NOT_NAMED", "UNNAMED"))]
        name = names.mode().iloc[0] if not names.empty else "UNNAMED"
        for station in stations:
            nearest_time, distance = closest_hourly_approach(track, station)
            if distance > max_distance_km:
                continue
            t0 = nearest_time.round("h")
            window_start = t0 - pd.Timedelta(hours=72)
            window_end = t0 + pd.Timedelta(hours=48)
            wind = track.WMO_WIND.max() if "WMO_WIND" in track else np.nan
            pressure = track.WMO_PRES.min() if "WMO_PRES" in track else np.nan
            rows.append({
                "event_id": f"{station.station_id}_{sid}",
                "station": station.station_id,
                "storm_name": name,
                "storm_sid": str(sid),
                "track_first_utc": track.time.min().isoformat(),
                "track_last_utc": track.time.max().isoformat(),
                "nearest_track_time_utc": nearest_time.isoformat(),
                "closest_time_utc": t0.isoformat(),
                "closest_distance_km": round(distance, 2),
                "window_start_utc": window_start.isoformat(),
                "window_end_exclusive_utc": window_end.isoformat(),
                "max_wmo_wind_kt": wind,
                "min_wmo_pressure_mb": pressure,
                "track_source": IBTRACS_SOURCE,
                "review_decision": "pending",
            })
    columns = (
        "event_id", "station", "storm_name", "storm_sid", "track_first_utc",
        "track_last_utc", "nearest_track_time_utc", "closest_time_utc", "closest_distance_km",
        "window_start_utc", "window_end_exclusive_utc", "max_wmo_wind_kt",
        "min_wmo_pressure_mb", "track_source", "review_decision",
    )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["station", "closest_time_utc", "storm_sid"]
    ).reset_index(drop=True)


def inspect_event_coverage(
    times: pd.DatetimeIndex,
    surge: np.ndarray,
    atmosphere: np.ndarray,
    t0: pd.Timestamp,
    year: int,
) -> dict[str, object]:
    """检查事先固定的120小时事件窗口及3个72小时起报。"""
    t0 = pd.Timestamp(t0)
    event_times = pd.date_range(t0 - pd.Timedelta(hours=72), periods=120, freq="h")
    event_positions = times.get_indexer(event_times)
    observed = np.full(len(event_times), np.nan, dtype=float)
    present = event_positions >= 0
    observed[present] = np.asarray(surge[event_positions[present]], dtype=float)
    atmosphere_valid = np.zeros(len(event_times), dtype=bool)
    if present.any():
        atmosphere_valid[present] = np.isfinite(
            np.asarray(atmosphere[event_positions[present]])
        ).all(axis=(1, 2, 3))
    valid_origins: list[int] = []
    invalid_origins: dict[str, str] = {}
    for offset in ORIGIN_OFFSETS_HOURS:
        origin_time = t0 + pd.Timedelta(hours=offset)
        complete_times = pd.date_range(
            origin_time - pd.Timedelta(hours=24), periods=96, freq="h"
        )
        positions = times.get_indexer(complete_times)
        reason = ""
        if complete_times[24].year != year or complete_times[-1].year != year:
            reason = "forecast crosses evaluation-year boundary"
        elif (positions < 0).any():
            reason = "hourly timeline missing"
        else:
            history = np.asarray(surge[positions[:24]], dtype=float)
            truth = np.asarray(surge[positions[24:]], dtype=float)
            weather = np.asarray(atmosphere[positions], dtype=float)
            if not np.isfinite(history).all():
                reason = "24-hour observed history has gaps"
            elif not np.isfinite(truth).all():
                reason = "72-hour verification has gaps"
            elif not np.isfinite(weather).all():
                reason = "ERA5 input has gaps"
        if reason:
            invalid_origins[str(offset)] = reason
        else:
            valid_origins.append(int(positions[24]))
    finite_observed = observed[np.isfinite(observed)]
    return {
        "window_hours": 120,
        "valid_observed_hours": int(np.isfinite(observed).sum()),
        "valid_era5_hours": int(atmosphere_valid.sum()),
        "max_observed_surge_cm": (
            float(np.max(finite_observed) * 100) if len(finite_observed) else np.nan
        ),
        "valid_origin_count": len(valid_origins),
        "valid_origin_positions": valid_origins,
        "invalid_origins": invalid_origins,
        "data_complete": (
            event_times[0].year == year
            and event_times[-1].year == year
            and np.isfinite(observed).all()
            and atmosphere_valid.all()
            and len(valid_origins) == len(ORIGIN_OFFSETS_HOURS)
        ),
    }


def score_forecast_window(
    observed_m: np.ndarray,
    predicted_m: np.ndarray,
    origin_time: pd.Timestamp,
    horizon_hours: int,
) -> dict[str, object]:
    """在同一起报后的前H小时比较过程与正向峰值。"""
    if horizon_hours not in EVALUATION_HOURS:
        raise ValueError(f"Unsupported event horizon: {horizon_hours}")
    observed = np.asarray(observed_m[:horizon_hours], dtype=float) * 100
    predicted = np.asarray(predicted_m[:horizon_hours], dtype=float) * 100
    if len(observed) != horizon_hours or len(predicted) != horizon_hours:
        raise ValueError("Forecast and observation do not cover the requested horizon")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Event metrics require complete finite forecast and observation")
    error = predicted - observed
    observed_peak = float(observed.max())
    predicted_peak = float(predicted.max())
    observed_step = int(observed.argmax())
    predicted_step = int(predicted.argmax())
    peak_available = observed_peak > 0
    return {
        "horizon_hours": horizon_hours,
        "n_hours": horizon_hours,
        "rmse_cm": float(np.sqrt(np.mean(error**2))),
        "mae_cm": float(np.mean(np.abs(error))),
        "bias_cm": float(np.mean(error)),
        "endpoint_error_cm": float(error[-1]),
        "observed_peak_cm": observed_peak if peak_available else np.nan,
        "predicted_peak_cm": predicted_peak if peak_available else np.nan,
        "peak_height_error_cm": predicted_peak - observed_peak if peak_available else np.nan,
        "absolute_peak_height_error_cm": abs(predicted_peak - observed_peak) if peak_available else np.nan,
        "peak_time_error_hours": predicted_step - observed_step if peak_available else np.nan,
        "observed_peak_time_utc": (
            (origin_time + pd.Timedelta(hours=observed_step)).isoformat()
            if peak_available else ""
        ),
        "predicted_peak_time_utc": (
            (origin_time + pd.Timedelta(hours=predicted_step)).isoformat()
            if peak_available else ""
        ),
        "peak_underestimated": bool(predicted_peak < observed_peak) if peak_available else None,
        "peak_underestimation_cm": max(0.0, observed_peak - predicted_peak) if peak_available else np.nan,
    }
