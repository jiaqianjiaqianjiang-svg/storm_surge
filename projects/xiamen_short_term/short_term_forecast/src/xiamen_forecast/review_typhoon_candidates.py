"""只读核验1997候选台风：输出路径、增水、ERA5风场与气压证据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from src.short_term_forecast.journal_figures.style import apply_axis_style, setup_journal_style

from .era5_loader import load_era5_files
from .station_config import get_station_config, validate_dataset_identity
from .train_xiamen import load_prepared
from .typhoon_events import inspect_event_coverage, read_ibtracs_wp, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CANDIDATES = REPO_ROOT / "reports/experiment_results/typhoon_events_1997_seed42_compact/candidate_events.csv"
DEFAULT_OUTPUT = REPO_ROOT / "reports/experiment_results/typhoon_candidate_review_1997"
TARGET_EVENTS = ("xiamen_1997232N14136", "lianyungang_1997210N15120")


def overlap_hours(start_a: pd.Timestamp, end_a: pd.Timestamp,
                  start_b: pd.Timestamp, end_b: pd.Timestamp) -> float:
    """两个左闭右开事件窗口的重叠小时数。"""
    return max(0.0, (min(end_a, end_b) - max(start_a, start_b)).total_seconds() / 3600)


def wind_from_degrees(u: float, v: float) -> float:
    """气象学风向：风从哪个方向吹来，北为0度。"""
    return float(np.degrees(np.arctan2(-u, -v)) % 360)


def oscillation_summary(values: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """线性去趋势后的描述性ACF/频谱，不判断振荡来源。"""
    observed = np.asarray(values, dtype=float)
    if len(observed) < 48 or not np.isfinite(observed).all():
        raise ValueError("At least 48 finite hourly observations are needed")
    trend = np.polyval(np.polyfit(np.arange(len(observed)), observed, 1), np.arange(len(observed)))
    residual = observed - trend
    acf = []
    for lag in range(1, 37):
        left, right = residual[:-lag], residual[lag:]
        correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 0 and np.std(right) > 0 else np.nan
        acf.append({"lag_hours": lag, "autocorrelation": correlation})
    frequency = np.fft.rfftfreq(len(residual), d=1.0)
    power = np.abs(np.fft.rfft(residual)) ** 2
    selected = (frequency > 0) & (frequency >= 1 / 36) & (frequency <= 1 / 2)
    spectrum = pd.DataFrame({"period_hours": 1 / frequency[selected], "relative_power": power[selected] / max(power[selected].sum(), 1e-12)})
    strongest = spectrum.iloc[int(spectrum.relative_power.argmax())]
    summary = {
        "acf_lag12": acf[11]["autocorrelation"],
        "acf_lag24": acf[23]["autocorrelation"],
        "largest_2_to_36h_period_hours": float(strongest.period_hours),
        "largest_period_relative_power": float(strongest.relative_power),
        "max_abs_hourly_change_cm": float(np.max(np.abs(np.diff(observed)))),
    }
    return pd.DataFrame(acf), spectrum.sort_values("period_hours"), summary


def weather_table(raw: xr.DataArray, processed: np.ndarray, times: pd.DatetimeIndex,
                  surge_m: np.ndarray) -> pd.DataFrame:
    """用真实网格坐标读天气；中央2×2仅是站点附近网格代理。"""
    values = np.asarray(raw.values, dtype=float)
    if values.shape != processed.shape or values.shape[1:] != (3, 40, 40):
        raise ValueError(f"Raw/processed ERA5 shapes differ: {values.shape}, {processed.shape}")
    if len(times) != len(values) or len(surge_m) != len(times):
        raise ValueError("Weather, surge and timestamps must have the same length")
    if not np.isfinite(values).all() or not np.isfinite(surge_m).all():
        raise ValueError("Candidate window contains missing ERA5 or surge values")
    if not np.allclose(values, processed, atol=0.01, rtol=1e-5):
        raise ValueError("Source ERA5 values do not match the aligned dataset")
    center = values[:, :, 19:21, 19:21].mean(axis=(2, 3))
    region = values.mean(axis=(2, 3))
    frame = pd.DataFrame({
        "time_utc": times.astype(str),
        "observed_surge_cm": np.asarray(surge_m, dtype=float) * 100,
        "center_u10_mps": center[:, 0], "center_v10_mps": center[:, 1],
        "center_msl_hpa": center[:, 2] / 100,
        "region_mean_u10_mps": region[:, 0], "region_mean_v10_mps": region[:, 1],
        "region_mean_msl_hpa": region[:, 2] / 100,
    })
    frame["center_wind_speed_mps"] = np.hypot(frame.center_u10_mps, frame.center_v10_mps)
    frame["region_mean_wind_speed_mps"] = np.hypot(frame.region_mean_u10_mps, frame.region_mean_v10_mps)
    frame["center_wind_from_deg"] = [wind_from_degrees(u, v) for u, v in zip(frame.center_u10_mps, frame.center_v10_mps)]
    frame["region_mean_wind_from_deg"] = [wind_from_degrees(u, v) for u, v in zip(frame.region_mean_u10_mps, frame.region_mean_v10_mps)]
    return frame


def plot_process(frame: pd.DataFrame, t0: pd.Timestamp, peak: pd.Timestamp,
                 other_t0: pd.Timestamp | None, path: Path) -> None:
    setup_journal_style(font_size=8.5)
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 7.7), sharex=True, layout="constrained")
    time = pd.to_datetime(frame.time_utc)
    axes[0].plot(time, frame.observed_surge_cm, color="#202428", lw=1.5)
    axes[0].scatter([peak], [frame.loc[frame.time_utc.eq(str(peak)), "observed_surge_cm"].iloc[0]], color="#202428", s=25, zorder=5)
    axes[0].set_ylabel("Surge (cm)")
    axes[1].plot(time, frame.center_u10_mps, color="#20566B", label="U10")
    axes[1].plot(time, frame.center_v10_mps, color="#A74F49", linestyle="--", label="V10")
    axes[1].set_ylabel("Center 2x2 wind (m/s)")
    axes[1].legend(loc="best")
    axes[2].plot(time, frame.center_wind_speed_mps, color="#20566B", label="Center 2x2")
    axes[2].plot(time, frame.region_mean_wind_speed_mps, color="#8A8580", linestyle="--", label="40x40 vector mean")
    axes[2].set_ylabel("Wind speed (m/s)")
    axes[2].legend(loc="best")
    axes[3].plot(time, frame.center_msl_hpa, color="#765E87", label="Center 2x2")
    axes[3].plot(time, frame.region_mean_msl_hpa, color="#8A8580", linestyle="--", label="40x40 mean")
    axes[3].set_ylabel("MSL pressure (hPa)")
    axes[3].legend(loc="best")
    for ax in axes:
        ax.axvline(t0, color="#A74F49", lw=0.8, linestyle=":")
        ax.axvline(peak, color="#202428", lw=0.8, linestyle=":")
        if other_t0 is not None:
            ax.axvline(other_t0, color="#2C7C74", lw=0.8, linestyle=":")
        apply_axis_style(ax)
    markers = "Black: observed peak; red: candidate closest approach"
    if other_t0 is not None:
        markers += "; green: CASS closest approach"
    axes[0].text(0.01, 0.96, markers,
                 transform=axes[0].transAxes, va="top", fontsize=7,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[-1].set_xlabel("UTC time")
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_wind_maps(raw: xr.DataArray, tracks: dict[str, pd.DataFrame], station: object,
                   t0: pd.Timestamp, peak: pd.Timestamp, path: Path) -> None:
    setup_journal_style(font_size=8.5)
    grid_times = pd.DatetimeIndex(pd.to_datetime(raw.time.values))
    indices = [grid_times.get_loc(t0), grid_times.get_loc(peak)]
    lat = np.asarray(raw.latitude.values, dtype=float)
    lon = np.asarray(raw.longitude.values, dtype=float)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
    maximum = max(float(np.hypot(raw.values[index, 0], raw.values[index, 1]).max()) for index in indices)
    image = None
    for ax, index, label in zip(axes, indices, ("Closest approach", "Observed peak")):
        u = np.asarray(raw.values[index, 0], dtype=float)
        v = np.asarray(raw.values[index, 1], dtype=float)
        image = ax.pcolormesh(lon, lat, np.hypot(u, v), shading="auto", cmap="cividis", vmin=0, vmax=maximum)
        ax.quiver(lon_grid[::4, ::4], lat_grid[::4, ::4], u[::4, ::4], v[::4, ::4], color="#202428", scale=140, width=0.004)
        for name, track in tracks.items():
            ax.plot(track.LON, track.LAT, lw=1.2, label=name)
        ax.scatter([station.longitude], [station.latitude], marker="*", color="#A74F49", s=85, zorder=5, label=station.name)
        ax.set(xlabel="Longitude (deg E)", ylabel="Latitude (deg N)",
               title=f"{label}: {grid_times[index]:%m-%d %H:%M} UTC",
               xlim=(lon.min(), lon.max()), ylim=(lat.min(), lat.max()))
        ax.set_aspect(1 / np.cos(np.deg2rad(station.latitude)))
        apply_axis_style(ax)
    axes[0].legend(loc="best", fontsize=7)
    fig.colorbar(image, ax=axes, label="Wind speed (m/s)", shrink=0.8)
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_oscillation(acf: pd.DataFrame, spectrum: pd.DataFrame, path: Path) -> None:
    setup_journal_style(font_size=8.5)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3), layout="constrained")
    axes[0].plot(acf.lag_hours, acf.autocorrelation, color="#20566B")
    axes[0].axhline(0, color="#202428", lw=0.6)
    axes[0].set(xlabel="Lag (h)", ylabel="Detrended autocorrelation", xlim=(1, 36))
    axes[1].plot(spectrum.period_hours, spectrum.relative_power, color="#A74F49")
    axes[1].set(xlabel="Period (h)", ylabel="Relative spectral power", xlim=(2, 36))
    for ax in axes:
        apply_axis_style(ax)
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-ids", nargs="+", default=list(TARGET_EVENTS))
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite existing review: {args.output_dir}")
    candidates = pd.read_csv(args.candidate_csv, encoding="utf-8-sig")
    selected = candidates.loc[candidates.event_id.isin(args.event_ids)]
    if set(selected.event_id) != set(args.event_ids) or selected.event_id.duplicated().any():
        raise ValueError("Candidate event IDs are missing or duplicated")
    protocol_path = args.candidate_csv.parent / "source_and_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_hash = sha256_file(args.track_csv)
    if source_hash != protocol["best_track_sha256"]:
        raise ValueError("Best-track file hash differs from the original candidate scan")
    tracks = read_ibtracs_wp(args.track_csv, 1997)
    summary_rows = []
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".typhoon_review_", dir=args.output_dir.parent) as temporary:
        work = Path(temporary)
        for _, row in selected.iterrows():
            station = get_station_config(str(row.station))
            metadata = validate_dataset_identity(station.dataset_dir, station.station_id)
            if metadata.get("array_schema", {}).get("atmosphere_variable_order") != ["U10", "V10", "MSL"]:
                raise ValueError(f"Unexpected ERA5 variable order for {station.station_id}")
            if station.grid_size != 40:
                raise ValueError("Review expects the audited 40x40 station grid")
            atmosphere, surge, times = load_prepared(station.dataset_dir, 1997, 1997)
            t0 = pd.Timestamp(row.closest_time_utc)
            coverage = inspect_event_coverage(times, surge, atmosphere, t0, 1997)
            if not coverage["data_complete"]:
                raise ValueError(f"Incomplete candidate data: {row.event_id}: {coverage['invalid_origins']}")
            window = pd.date_range(t0 - pd.Timedelta(hours=72), periods=120, freq="h")
            positions = times.get_indexer(window)
            if (positions < 0).any():
                raise ValueError(f"Missing observation times: {row.event_id}")
            raw = load_era5_files(
                list(station.era5_files.values()), station.latitude, station.longitude,
                grid_size=station.grid_size, region_size_degrees=station.region_size_degrees,
                start_time=window[0], end_time=window[-1],
            )
            if not np.array_equal(pd.DatetimeIndex(pd.to_datetime(raw.time.values)), window):
                raise ValueError(f"Raw ERA5 hours differ from aligned data: {row.event_id}")
            frame = weather_table(raw, np.asarray(atmosphere[positions]), window, np.asarray(surge[positions]))
            peak_row = frame.loc[frame.observed_surge_cm.idxmax()]
            peak_time = pd.Timestamp(peak_row.time_utc)
            if not np.isclose(peak_row.observed_surge_cm, row.max_observed_surge_cm, atol=0.02):
                raise ValueError(f"Candidate peak differs from original scan: {row.event_id}")
            closest_row = frame.loc[frame.time_utc.eq(str(t0))].iloc[0]
            local_track = tracks.loc[tracks.SID.astype(str).eq(str(row.storm_sid))].copy()
            if local_track.empty:
                raise ValueError(f"No matching best-track points: {row.storm_sid}")
            shown_tracks = {str(row.storm_name): local_track}
            other_t0 = None
            overlap = 0.0
            if row.station == "xiamen":
                cass = candidates.loc[(candidates.station == "xiamen") & (candidates.storm_name == "CASS")]
                if len(cass) != 1:
                    raise ValueError("Expected one Xiamen CASS candidate for overlap check")
                other = cass.iloc[0]
                other_t0 = pd.Timestamp(other.closest_time_utc)
                overlap = overlap_hours(pd.Timestamp(row.window_start_utc), pd.Timestamp(row.window_end_exclusive_utc),
                                        pd.Timestamp(other.window_start_utc), pd.Timestamp(other.window_end_exclusive_utc))
                cass_track = tracks.loc[tracks.SID.astype(str).eq(str(other.storm_sid))].copy()
                if cass_track.empty:
                    raise ValueError("CASS best-track points are missing")
                shown_tracks["CASS"] = cass_track
            acf, spectrum, oscillation = oscillation_summary(frame.observed_surge_cm.to_numpy())
            event_dir = work / str(row.event_id)
            event_dir.mkdir()
            frame.to_csv(event_dir / "hourly_evidence.csv", index=False, encoding="utf-8-sig")
            acf.to_csv(event_dir / "detrended_acf.csv", index=False, encoding="utf-8-sig")
            spectrum.to_csv(event_dir / "detrended_periodogram.csv", index=False, encoding="utf-8-sig")
            for name, track in shown_tracks.items():
                track.to_csv(event_dir / f"best_track_{name}.csv", index=False, encoding="utf-8-sig")
            plot_process(frame, t0, peak_time, other_t0, event_dir / "surge_weather_review.png")
            plot_wind_maps(raw, shown_tracks, station, t0, peak_time, event_dir / "regional_wind_review.png")
            plot_oscillation(acf, spectrum, event_dir / "oscillation_review.png")
            summary_rows.append({
                "event_id": row.event_id, "station": row.station, "storm_name": row.storm_name,
                "review_decision": "pending", "closest_time_utc": t0.isoformat(),
                "closest_distance_km": row.closest_distance_km,
                "observed_peak_time_utc": peak_time.isoformat(),
                "observed_peak_cm": float(peak_row.observed_surge_cm),
                "peak_offset_from_closest_hours": float((peak_time - t0).total_seconds() / 3600),
                "overlap_with_cass_hours": overlap,
                "valid_observed_hours": coverage["valid_observed_hours"],
                "valid_era5_hours": coverage["valid_era5_hours"],
                "valid_origin_count": coverage["valid_origin_count"],
                "center_wind_from_at_peak_deg": float(peak_row.center_wind_from_deg),
                "region_mean_wind_from_at_peak_deg": float(peak_row.region_mean_wind_from_deg),
                "center_wind_speed_at_peak_mps": float(peak_row.center_wind_speed_mps),
                "center_msl_at_peak_hpa": float(peak_row.center_msl_hpa),
                "center_wind_speed_at_closest_mps": float(closest_row.center_wind_speed_mps),
                "center_msl_at_closest_hpa": float(closest_row.center_msl_hpa),
                **oscillation,
            })
        pd.DataFrame(summary_rows).to_csv(work / "candidate_review_summary.csv", index=False, encoding="utf-8-sig")
        (work / "review_protocol.json").write_text(json.dumps({
            "track_sha256": source_hash, "candidate_csv": str(args.candidate_csv),
            "analysis_year": 1997, "source": "read-only GESLA-derived surge and ERA5 inputs",
            "attribution_status": "pending manual review, not automatically confirmed",
            "notes": "Central 2x2 and full 40x40 means are grid proxies; map uses raw ERA5 coordinates. ACF/FFT do not establish residual tide.",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (work / "README.md").write_text(
            "# 1997候选台风人工核验材料\n\n"
            "此包只读复核对齐增水、原始ERA5和IBTrACS；`review_decision`始终为pending。"
            "请结合带真实坐标的区域风矢量图、风速/气压和路径审查归因。"
            "厦门AMBER必须与CASS分别核验，重叠窗口中的峰值不得直接归于单一台风。"
            "连云港VICTOR的ACF/频谱只描述振荡；没有原始潮位与潮汐分离证据不能称残余潮汐。"
            "人工确认前不要运行正式事件回算。\n",
            encoding="utf-8",
        )
        work.rename(args.output_dir)
    print(f"Review evidence ready (not attribution confirmation): {args.output_dir}")


if __name__ == "__main__":
    main()
