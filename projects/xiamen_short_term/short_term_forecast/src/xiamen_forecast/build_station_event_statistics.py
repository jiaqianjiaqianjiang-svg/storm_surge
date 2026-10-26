"""按站点划分统计历史台风候选及非潮增水极端样本（不训练模型）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .station_config import StationConfig, get_station_config, validate_dataset_identity
from .typhoon_events import build_candidates, read_ibtracs_wp, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[5]
RESULT_ROOT = REPO_ROOT / "reports/experiment_results"
DEFAULT_OUTPUT = RESULT_ROOT / "china_three_station_event_distribution_v1"
REVIEW_PACKAGES = (
    "typhoon_events_1997_seed42_compact",
    "typhoon_events_1997_amber_victor_seed42",
)
SPLITS = ("train", "validation", "test")
QUANTILES = (0.90, 0.95, 0.99)
GAPS_HOURS = (12, 24, 48)


def split_for_year(year: int, station: StationConfig) -> str | None:
    if station.train_start_year <= year <= station.train_end_year:
        return "train"
    if year == station.validation_year:
        return "validation"
    if year == station.test_year:
        return "test"
    return None


def _range_count(bad: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    prefix = np.r_[0, np.cumsum(bad, dtype=np.int64)]
    return prefix[right] - prefix[left]


def valid_target_mask(times: pd.DatetimeIndex, surge: np.ndarray,
                      atmosphere_valid: np.ndarray, input_steps: int) -> np.ndarray:
    """与dataset_builder.valid_targets相同的24小时历史和目标有效性规则。"""
    count = len(times)
    result = np.zeros(count, dtype=bool)
    if count <= input_steps:
        return result
    target = np.arange(input_steps, count)
    gaps = np.asarray(np.diff(times.values) != np.timedelta64(1, "h"))
    finite_surge = np.isfinite(surge)
    result[target] = (
        (_range_count(gaps, target - input_steps, target) == 0)
        & (_range_count(~atmosphere_valid, target - input_steps, target) == 0)
        & (_range_count(~finite_surge, target - input_steps, target + 1) == 0)
    )
    return result


def independent_peak_events(times: pd.DatetimeIndex, surge_m: np.ndarray,
                            threshold_m: float, gap_hours: int) -> pd.DataFrame:
    """超过阈值的小时相隔不超过gap小时合并；每簇只取一个最高峰。"""
    positions = np.flatnonzero(np.isfinite(surge_m) & (surge_m > 0) & (surge_m >= threshold_m))
    columns = ["start_utc", "end_utc", "peak_time_utc", "peak_cm", "above_hours", "gap_hours"]
    if not len(positions):
        return pd.DataFrame(columns=columns)
    clusters: list[np.ndarray] = []
    first = 0
    for i in range(1, len(positions)):
        elapsed = (times[positions[i]] - times[positions[i - 1]]).total_seconds() / 3600
        if elapsed > gap_hours:
            clusters.append(positions[first:i])
            first = i
    clusters.append(positions[first:])
    rows = []
    for positions_in_event in clusters:
        peak_index = int(positions_in_event[np.argmax(surge_m[positions_in_event])])
        rows.append({
            "start_utc": times[positions_in_event[0]].isoformat(),
            "end_utc": times[positions_in_event[-1]].isoformat(),
            "peak_time_utc": times[peak_index].isoformat(),
            "peak_cm": float(surge_m[peak_index] * 100),
            "above_hours": len(positions_in_event),
            "gap_hours": gap_hours,
        })
    return pd.DataFrame(rows, columns=columns)


def load_station_arrays(station: StationConfig) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    root = station.dataset_dir
    names = ("time.npy", "surge.npy", "atmosphere.npy", "dataset_metadata.json")
    missing = [str(root / name) for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing prepared dataset files:\n" + "\n".join(missing))
    validate_dataset_identity(root, station.station_id)
    raw_time = np.load(root / "time.npy", mmap_mode="r", allow_pickle=False)
    raw_surge = np.load(root / "surge.npy", mmap_mode="r", allow_pickle=False)
    atmosphere = np.load(root / "atmosphere.npy", mmap_mode="r", allow_pickle=False)
    if raw_time.ndim != 1 or raw_surge.ndim != 1 or len(raw_time) != len(raw_surge):
        raise ValueError(f"Invalid time/surge shapes in {root}")
    if atmosphere.shape != (len(raw_time), 3, station.grid_size, station.grid_size):
        raise ValueError(f"Invalid atmosphere shape in {root}: {atmosphere.shape}")
    times = pd.DatetimeIndex(pd.to_datetime(raw_time))
    if times.hasnans or not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError(f"Time axis must be strictly increasing without NaT or duplicates: {root}")
    if times[0] > pd.Timestamp(station.train_start_year, 1, 1) or times[-1] < pd.Timestamp(station.test_year, 12, 31, 23):
        raise ValueError(f"Dataset does not cover configured train/test endpoints: {root}: {times[0]} to {times[-1]}")
    surge = np.asarray(raw_surge, dtype=float)
    atmospheric_valid = np.zeros(len(times), dtype=bool)
    for start in range(0, len(times), 168):
        chunk = np.asarray(atmosphere[start:start + 168])
        atmospheric_valid[start:start + len(chunk)] = np.isfinite(chunk).all(axis=(1, 2, 3))
    return times, surge, atmospheric_valid


def training_thresholds(times: pd.DatetimeIndex, surge: np.ndarray,
                        target_valid: np.ndarray, station: StationConfig) -> dict[str, float]:
    train = (times.year >= station.train_start_year) & (times.year <= station.train_end_year)
    values = surge[train & target_valid]
    if not len(values):
        raise ValueError(f"No valid training targets for threshold fitting: {station.station_id}")
    raw = np.quantile(values, QUANTILES)
    if raw[-1] <= 0:
        raise ValueError(f"Training P99 is not positive: {station.station_id}")
    return {f"p{int(q * 100)}_m": float(max(0.0, value)) for q, value in zip(QUANTILES, raw)}


def station_extreme_tables(station: StationConfig, times: pd.DatetimeIndex,
                           surge: np.ndarray, atmosphere_valid: np.ndarray
                           ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    target_valid = valid_target_mask(times, surge, atmosphere_valid, station.input_steps)
    thresholds = training_thresholds(times, surge, target_valid, station)
    p99 = thresholds["p99_m"]
    peaks_by_gap = {gap: independent_peak_events(times, surge, p99, gap) for gap in GAPS_HOURS}
    peaks = peaks_by_gap[24].copy()
    peaks.insert(0, "station", station.station_id)
    peaks["split"] = [split_for_year(pd.Timestamp(value).year, station) for value in peaks.peak_time_utc]
    peaks["year"] = pd.to_datetime(peaks.peak_time_utc).dt.year
    peaks["cross_split_cluster"] = [
        split_for_year(pd.Timestamp(row.start_utc).year, station)
        != split_for_year(pd.Timestamp(row.end_utc).year, station)
        for row in peaks.itertuples()
    ]
    # 快速上涨与绝对幅度不同；阈值也只从训练期连续1小时的正向变化拟合。
    rise = np.full(len(surge), np.nan)
    contiguous = np.r_[False, np.diff(times.values) == np.timedelta64(1, "h")]
    okay = contiguous & np.isfinite(surge) & np.r_[False, np.isfinite(surge[:-1])]
    rise[okay] = surge[okay] - surge[np.flatnonzero(okay) - 1]
    train_rise = rise[(times.year <= station.train_end_year) & (times.year >= station.train_start_year)]
    positive_rise = train_rise[np.isfinite(train_rise) & (train_rise > 0)]
    rise_p90 = float(np.quantile(positive_rise, 0.90)) if len(positive_rise) else float("nan")
    rapid = independent_peak_events(times, rise, rise_p90, 24) if np.isfinite(rise_p90) else pd.DataFrame()
    thresholds["rapid_rise_p90_cm_per_h"] = rise_p90 * 100 if np.isfinite(rise_p90) else None
    thresholds["training_target_count"] = int((target_valid & (times.year >= station.train_start_year)
                                                & (times.year <= station.train_end_year)).sum())
    rows = []
    for year in range(station.train_start_year, station.test_year + 1):
        split = split_for_year(year, station)
        if split is None:
            continue
        mask = np.asarray(times.year == year)
        observed_hours = int(mask.sum())
        expected_hours = len(pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left"))
        finite = mask & np.isfinite(surge)
        row = {
            "station": station.station_id, "split": split, "year": year,
            "expected_hours": expected_hours, "observed_hours": observed_hours,
            "valid_surge_hours": int(finite.sum()),
            "valid_model_targets": int((mask & target_valid).sum()),
            "missing_rate": 1 - int(finite.sum()) / expected_hours,
            "max_positive_surge_cm": float(max(0, np.max(surge[finite]) * 100)) if finite.any() else np.nan,
            "rapid_rise_hours": int((mask & np.isfinite(rise) & (rise >= rise_p90)).sum()),
            "rapid_rise_events_gap24": int((pd.to_datetime(rapid.peak_time_utc).dt.year == year).sum()) if len(rapid) else 0,
            "training_rapid_rise_p90_cm_per_h": rise_p90 * 100,
        }
        for label, threshold in thresholds.items():
            if not label.startswith("p"):
                continue
            exceed = mask & np.isfinite(surge) & (surge > 0) & (surge >= threshold)
            row[f"training_{label[:-2]}_threshold_cm"] = threshold * 100
            row[f"{label[:-2]}_hours"] = int(exceed.sum())
            row[f"{label[:-2]}_fraction_valid"] = int(exceed.sum()) / int(finite.sum()) if finite.any() else np.nan
        for gap, events in peaks_by_gap.items():
            row[f"p99_peak_events_gap{gap}"] = int((pd.to_datetime(events.peak_time_utc).dt.year == year).sum()) if len(events) else 0
        rows.append(row)
    return pd.DataFrame(rows), peaks, thresholds


def _reviews(track_sha256: str) -> dict[str, str]:
    reviewed = {}
    for package in REVIEW_PACKAGES:
        source = RESULT_ROOT / package
        file = source / "confirmed_event_list.csv"
        protocol = source / "source_and_protocol.json"
        if file.is_file() and protocol.is_file() and json.loads(protocol.read_text(encoding="utf-8")).get("best_track_sha256") == track_sha256:
            table = pd.read_csv(file, encoding="utf-8-sig")
            for row in table.itertuples():
                if row.review_decision == "yes":
                    reviewed[row.event_id] = package
        elif file.is_file():
            warnings.warn(f"Archived preliminary reviews from {package} were not inherited: best-track SHA256 differs or protocol is missing")
    return reviewed


def station_candidate_table(station: StationConfig, candidates: pd.DataFrame,
                            times: pd.DatetimeIndex, surge: np.ndarray,
                            atmosphere_valid: np.ndarray, reviews: dict[str, str]
                            ) -> pd.DataFrame:
    rows = []
    for row in candidates.loc[candidates.station.eq(station.station_id)].itertuples():
        t0 = pd.Timestamp(row.closest_time_utc)
        split = split_for_year(t0.year, station)
        if split is None:
            continue
        start = t0 - pd.Timedelta(hours=96)
        end = t0 + pd.Timedelta(hours=48)
        expected = pd.date_range(start, periods=144, freq="h")
        positions = times.get_indexer(expected)
        present = positions >= 0
        valid_observed = present.copy()
        valid_era5 = present.copy()
        valid_observed[present] = np.isfinite(surge[positions[present]])
        valid_era5[present] = atmosphere_valid[positions[present]]
        stage_cross = split_for_year(start.year, station) != split or split_for_year((end - pd.Timedelta(hours=1)).year, station) != split
        track_cross = split_for_year(pd.Timestamp(row.track_first_utc).year, station) != split or split_for_year(pd.Timestamp(row.track_last_utc).year, station) != split
        complete = bool(valid_observed.all() and valid_era5.all() and not stage_cross)
        event_observed = valid_observed[24:]
        event_positions = positions[24:][event_observed]
        maximum = float(np.max(surge[event_positions]) * 100) if len(event_positions) else np.nan
        rows.append({**row._asdict(), "split": split, "year": t0.year,
                     "within_500km": bool(row.closest_distance_km <= 500),
                     "data_complete_same_split": complete,
                     "observed_hours_120h": int(event_observed.sum()),
                     "era5_hours_120h": int(valid_era5[24:].sum()),
                     "input_evaluation_cross_split": bool(stage_cross),
                     "track_cross_split": bool(track_cross),
                     "max_observed_surge_cm_120h": maximum,
                     "review_status": "preliminary_included" if row.event_id in reviews else "not_reviewed",
                     "review_source_package": reviews.get(row.event_id, "")})
    return pd.DataFrame(rows)


def crosswalk_peaks(peaks: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    nearby = candidates.loc[candidates.within_500km]
    for peak in peaks.itertuples():
        moment = pd.Timestamp(peak.peak_time_utc)
        same = nearby.loc[nearby.station.eq(peak.station)].copy()
        in_window = same.loc[(pd.to_datetime(same.window_start_utc) <= moment) &
                             (moment < pd.to_datetime(same.window_end_exclusive_utc))]
        rows.append({"station": peak.station, "split": peak.split, "year": peak.year,
                     "peak_time_utc": peak.peak_time_utc, "peak_cm": peak.peak_cm,
                     "candidate_count": len(in_window),
                     "candidate_event_ids": ";".join(in_window.event_id),
                     "preliminary_included_count": int(in_window.review_status.eq("preliminary_included").sum()),
                     "attribution": "overlap_only_not_causal" if len(in_window) else "no_500km_candidate_window"})
    return pd.DataFrame(rows)


def count_candidates(candidates: pd.DataFrame, stations: list[StationConfig]) -> pd.DataFrame:
    rows = []
    for station in stations:
        for split in SPLITS:
            part = candidates.loc[candidates.station.eq(station.station_id) & candidates.split.eq(split)]
            near = part.loc[part.within_500km]
            rows.append({
                "station": station.station_id, "split": split,
                "ibtracs_wp_storms": len(part), "within_500km_candidates": len(near),
                "data_complete_same_split_candidates": int(near.data_complete_same_split.sum()),
                "preliminary_manually_included": int(near.review_status.eq("preliminary_included").sum()),
                "fully_attributed_confirmed": 0,
                "not_reviewed_candidates": int(near.review_status.eq("not_reviewed").sum()),
                "attribution_unresolved_candidates": len(near),
                "incomplete_or_cross_split": int((~near.data_complete_same_split).sum()),
                "cross_split_window_candidates": int(near.input_evaluation_cross_split.sum()),
                "track_cross_split_candidates": int(near.track_cross_split.sum()),
            })
    return pd.DataFrame(rows)


def _figures(counts: pd.DataFrame, distribution: pd.DataFrame, work: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2})
    stations = list(counts.station.unique())
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), layout="constrained")
    for ax, station in zip(axes, stations):
        data = counts.loc[counts.station.eq(station)].set_index("split").reindex(SPLITS)
        x = np.arange(3)
        ax.bar(x - 0.18, data.within_500km_candidates, width=0.36, color="#40566D", label="Within 500 km")
        ax.bar(x + 0.18, data.data_complete_same_split_candidates, width=0.36, color="#3A887C", label="Complete")
        ax.set(title=station, xticks=x, xticklabels=SPLITS, ylabel="Candidate storms (count)")
    axes[0].legend(fontsize=7)
    fig.savefig(work / "fig01_typhoon_counts_by_split.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    aggregated = distribution.groupby(["station", "split"], as_index=False).agg({
        "valid_surge_hours": "sum", "p90_hours": "sum", "p95_hours": "sum", "p99_hours": "sum"})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), layout="constrained")
    for ax, station in zip(axes, stations):
        data = aggregated.loc[aggregated.station.eq(station)].set_index("split").reindex(SPLITS)
        for percentile, color in ((90, "#33699B"), (95, "#3A887C"), (99, "#A45C52")):
            ax.plot(SPLITS, 100 * data[f"p{percentile}_hours"] / data.valid_surge_hours,
                    marker="o", label=f"P{percentile}", color=color)
        ax.set(title=station, ylabel="Above training threshold (% of valid hours)")
    axes[0].legend(fontsize=7)
    fig.savefig(work / "fig02_surge_threshold_distribution.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), layout="constrained")
    for ax, station in zip(axes, stations):
        data = distribution.loc[distribution.station.eq(station)].groupby("split").sum(numeric_only=True).reindex(SPLITS)
        x = np.arange(3)
        for offset, gap, color in ((-0.25, 12, "#40566D"), (0, 24, "#3A887C"), (0.25, 48, "#A45C52")):
            ax.bar(x + offset, data[f"p99_peak_events_gap{gap}"], width=0.24,
                   color=color, label=f"Gap {gap} h")
        ax.set(title=station, xticks=x, xticklabels=SPLITS, ylabel="Independent P99 peak clusters (count)")
    axes[0].legend(fontsize=7)
    fig.savefig(work / "fig03_extreme_events_by_split.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def _documents(work: Path, track: Path, counts: pd.DataFrame,
               distribution: pd.DataFrame, thresholds: dict[str, dict[str, float]]) -> None:
    (work / "statistics_methodology.md").write_text(
        "# 统计口径\n\n"
        f"最佳路径：IBTrACS WP CSV `{track.name}`，SHA256 `{sha256_file(track)}`。"
        "按SID、站点最近逐小时插值时刻计一次；500 km仅初筛，不是归因边界。"
        "每场以最近时刻t0为中心，要求t0-96h至t0+48h的观测/ERA5齐全，"
        "其中前24h是最早起报的历史输入。跨训练/验证/测试边界不算同阶段完整候选。"
        "已归档事件只标记人工初步纳入，区域风向、最佳路径和因果归因仍待完整核验；"
        "未经核验的历史候选不自动算已确认。\n\n"
        "P90/P95/P99取训练年份全部有效的一步预测目标（有连续24h历史、有限ERA5及增水），"
        "包含正负增水，与现有极值损失的阈值样本口径一致；阈值下限保护为0 m。"
        "阈值冻结后统计各阶段全部有效观测小时中的正向超阈值小时及占比。"
        "相邻超阈值小时相隔不超过12/24/48h合并为事件，以最高点年份归属；"
        "24h为主口径。跨年簇按峰值归属，并不代表训练时使用未来数据。"
        "快速上涨另取连续1h正向差值的训练期P90，以24h合并；不与增水幅度混用。"
        "台风窗口与极端簇的重合只表示时间关联，不证明动力归因。\n",
        encoding="utf-8",
    )
    lines = ["# 三站台风候选与极端增水分布", "", "以下计数来自处理后数组和IBTrACS；不是模型预测精度。",
             "人工初步纳入不等于完成因果归因，未经人工核验的候选不能视为已确认台风影响。", "",
             "## 训练期固定阈值", ""]
    for station, value in thresholds.items():
        lines.append(f"- {station}：P90={value['p90_m']*100:.2f} cm，P95={value['p95_m']*100:.2f} cm，P99={value['p99_m']*100:.2f} cm；训练目标数={value['training_target_count']}。")
    lines.extend(["", "## 各阶段初筛与极端簇", ""])
    for row in counts.itertuples():
        part = distribution.loc[distribution.station.eq(row.station) & distribution.split.eq(row.split)]
        lines.append(f"- {row.station} / {row.split}：500 km内候选{row.within_500km_candidates}场，"
                     f"同阶段数据完整{row.data_complete_same_split_candidates}场，人工初步纳入{row.preliminary_manually_included}场，"
                     f"P99峰值簇(24h间隔){int(part.p99_peak_events_gap24.sum())}个。")
    lines.extend(["", "跨阶段候选、缺测及逐年分布见CSV。未来ERA5回算和历史路径筛选不能称为业务实时预报。", ""])
    (work / "statistics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", nargs="+", default=["xiamen", "lianyungang", "beihai"])
    parser.add_argument("--track-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    stations = [get_station_config(name) for name in args.stations]
    if len({station.station_id for station in stations}) != len(stations):
        parser.error("Duplicate station names are not allowed")
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite existing statistics: {args.output_dir}")
    if not args.track_csv.is_file():
        raise FileNotFoundError(f"IBTrACS best-track CSV not found: {args.track_csv}")
    arrays = {station.station_id: load_station_arrays(station) for station in stations}
    tracks = read_ibtracs_wp(args.track_csv, (min(s.train_start_year for s in stations) - 1,
                                               max(s.test_year for s in stations) + 1))
    candidates = build_candidates(tracks, stations, max_distance_km=float("inf"))
    reviews = _reviews(sha256_file(args.track_csv))
    distributions, peaks, catalog, thresholds = [], [], [], {}
    for station in stations:
        times, surge, atmosphere_valid = arrays[station.station_id]
        dist, station_peaks, thresholds[station.station_id] = station_extreme_tables(
            station, times, surge, atmosphere_valid)
        distributions.append(dist)
        peaks.append(station_peaks)
        catalog.append(station_candidate_table(station, candidates, times, surge,
                                               atmosphere_valid, reviews))
    distribution = pd.concat(distributions, ignore_index=True)
    all_peaks = pd.concat(peaks, ignore_index=True)
    all_candidates = pd.concat(catalog, ignore_index=True)
    counts = count_candidates(all_candidates, stations)
    crosswalk = crosswalk_peaks(all_peaks, all_candidates)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".station_event_stats_", dir=args.output_dir.parent) as temporary:
        work = Path(temporary)
        for name, table in (("station_typhoon_counts_by_split.csv", counts),
                            ("station_typhoon_event_catalog.csv", all_candidates),
                            ("station_extreme_sample_distribution.csv", distribution),
                            ("station_independent_peak_events.csv", all_peaks),
                            ("station_typhoon_extreme_crosswalk.csv", crosswalk)):
            table.to_csv(work / name, index=False, encoding="utf-8-sig")
        (work / "training_thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
        _figures(counts, distribution, work)
        _documents(work, args.track_csv, counts, distribution, thresholds)
        work.rename(args.output_dir)
    print(f"Three-station event statistics: {args.output_dir}")
    return args.output_dir


if __name__ == "__main__":
    main()
