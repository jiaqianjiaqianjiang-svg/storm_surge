"""1997年真实台风候选筛选与固定起报历史回算；不训练模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from src.short_term_forecast.journal_figures.style import DPI, MODEL_COLORS, setup_journal_style

from .evaluate_baselines import summarise_atmosphere
from .rolling_diagnostics import load_models, recursive_predictions
from .station_config import STATIONS, get_station_config, validate_dataset_identity
from .train_xiamen import load_prepared
from .typhoon_events import (
    EVALUATION_HOURS,
    IBTRACS_SOURCE,
    IBTRACS_WP_CSV,
    ORIGIN_OFFSETS_HOURS,
    build_candidates,
    inspect_event_coverage,
    read_ibtracs_wp,
    score_forecast_window,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "experiment_results" / "typhoon_events_1997_seed42"
EXTRA_NAMES = {
    "wmse_mse": "cnn_gru_extreme_wmse_rollout6_mse",
    "wmse_wmse": "cnn_gru_extreme_wmse_rollout6_wmse",
    "awmse_awmse": "cnn_gru_extreme_awmse_rollout6_awmse",
    "whuber_whuber": "cnn_gru_extreme_whuber_rollout6_whuber",
}
COLORS = {
    "observed": MODEL_COLORS["observed"],
    "ridge": MODEL_COLORS["ridge"],
    "cnn_gru": MODEL_COLORS["cnn_gru"],
    "cnn_gru_rollout6": MODEL_COLORS["cnn_gru_rollout6"],
    "wmse_mse": "#748343",
    "awmse_awmse": "#81628E",
}
MAIN_MODELS = ("cnn_gru", "cnn_gru_rollout6", "wmse_mse", "awmse_awmse")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for action in ("scan", "run"):
        command = sub.add_parser(action)
        command.add_argument("--track-csv", type=Path, required=True, help="下载的NOAA IBTrACS WP CSV或CSV.GZ；原文件只读")
        command.add_argument("--results-dir", type=Path, default=DEFAULT_OUTPUT)
        command.add_argument("--year", type=int, default=1997)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--stations", nargs="+", choices=tuple(STATIONS), default=list(STATIONS))
        if action == "scan":
            command.add_argument("--max-distance-km", type=float, default=500)
        else:
            command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return result


def model_files(station_id: str, seed: int) -> tuple[Path, Path, Path, dict[str, Path]]:
    station = get_station_config(station_id)
    formal = station.model_root / f"formal_seed{seed}"
    ridge = station.model_root / "baselines" / "ridge_pipeline.joblib"
    one_step = formal / "cnn_gru" / "best_model.pth"
    rollout = formal / "cnn_gru_rollout6" / "best_model.pth"
    extras = (
        {name: formal / folder / "best_model.pth" for name, folder in EXTRA_NAMES.items()}
        if station_id == "lianyungang" else {}
    )
    return ridge, one_step, rollout, extras


def missing_model_files(station_id: str, seed: int) -> list[str]:
    ridge, one_step, rollout, extras = model_files(station_id, seed)
    return [str(path) for path in (ridge, one_step, rollout, *extras.values()) if not path.is_file()]


def read_station_year(station_id: str, year: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    station = get_station_config(station_id)
    validate_dataset_identity(station.dataset_dir, station_id)
    return load_prepared(station.dataset_dir, year, year)


def _metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "best_track_source": IBTRACS_SOURCE,
        "best_track_download": IBTRACS_WP_CSV,
        "best_track_local_file": str(args.track_csv.resolve()),
        "best_track_sha256": sha256_file(args.track_csv),
        "year": args.year,
        "seed": args.seed,
        "stations": args.stations,
        "initial_distance_km": getattr(args, "max_distance_km", None),
        "forecast_type": "historical hindcast with known future ERA5 reanalysis forcing",
        "future_observed_surge_used_as_model_input": False,
        "test_status": "1997 additional historical case analysis; not a new blind test",
        "fixed_origin_offsets_hours": ORIGIN_OFFSETS_HOURS,
        "fixed_horizons_hours": EVALUATION_HOURS,
    }


def scan(args: argparse.Namespace) -> None:
    output = args.results_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing case package: {output}")
    tracks = read_ibtracs_wp(args.track_csv, args.year)
    candidates = build_candidates(
        tracks, [get_station_config(name) for name in args.stations], args.max_distance_km
    )
    candidates["max_observed_surge_cm"] = np.nan
    candidates["valid_observed_hours"] = 0
    candidates["valid_era5_hours"] = 0
    candidates["valid_origin_count"] = 0
    candidates["data_complete"] = False
    candidates["missing_model_count"] = 0
    candidates["screening_note"] = ""
    candidates["review_note"] = ""
    for station_id in args.stations:
        indices = candidates.index[candidates.station == station_id]
        if not len(indices):
            continue
        missing = missing_model_files(station_id, args.seed)
        try:
            atmosphere, surge, times = read_station_year(station_id, args.year)
        except (FileNotFoundError, ValueError) as error:
            for index in indices:
                candidates.at[index, "screening_note"] = f"prepared data unavailable: {error}"
            continue
        for index in indices:
            coverage = inspect_event_coverage(
                times, surge, atmosphere,
                pd.Timestamp(candidates.at[index, "closest_time_utc"]), args.year,
            )
            for column in ("max_observed_surge_cm", "valid_observed_hours", "valid_era5_hours", "valid_origin_count", "data_complete"):
                candidates.at[index, column] = coverage[column]
            candidates.at[index, "missing_model_count"] = len(missing)
            notes = list(coverage["invalid_origins"].values())
            if missing:
                notes.append("missing model files: " + "; ".join(missing))
            if not coverage["data_complete"]:
                notes.append("event window incomplete")
            if not np.isfinite(coverage["max_observed_surge_cm"]) or coverage["max_observed_surge_cm"] <= 0:
                notes.append("no positive observed surge peak in the fixed event window")
            candidates.at[index, "screening_note"] = "; ".join(dict.fromkeys(notes))
    output.mkdir(parents=True)
    candidates.to_csv(output / "candidate_events.csv", index=False, encoding="utf-8-sig")
    (output / "source_and_protocol.json").write_text(
        json.dumps(_metadata(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Candidate checklist: {output / 'candidate_events.csv'}")
    print("All review_decision values are pending. Verify cyclone influence and mark yes/no before run.")


def _device(choice: str) -> torch.device:
    selected = "cuda" if choice == "auto" and torch.cuda.is_available() else ("cpu" if choice == "auto" else choice)
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(selected)


def _plot_track(track: pd.DataFrame, station_id: str, t0: pd.Timestamp, path: Path) -> None:
    setup_journal_style()
    station = get_station_config(station_id)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(track.LON, track.LAT, color="#20566B", marker="o", markersize=2.5, label="IBTrACS best track")
    ax.scatter([station.longitude], [station.latitude], c="#A74F49", marker="*", s=100, label=station.name, zorder=4)
    every = max(1, len(track) // 12)
    for _, point in track.iloc[::every].iterrows():
        ax.annotate(pd.Timestamp(point.time).strftime("%m-%d %H"), (point.LON, point.LAT), xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    ax.set_aspect(1 / np.cos(np.deg2rad(station.latitude)))
    ax.grid(alpha=0.5)
    ax.legend(loc="best")
    ax.text(0.01, 0.01, f"Closest approach near {t0:%Y-%m-%d %H:%M} UTC", transform=ax.transAxes, fontsize=7)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_process(forecasts: pd.DataFrame, path: Path) -> None:
    setup_journal_style()
    names = [name for name in MAIN_MODELS if name in forecasts.model.unique()]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.1), sharey=True, layout="constrained")
    for axis, (offset, section) in zip(axes, forecasts.groupby("origin_offset_hours", sort=True)):
        observed = section.loc[section.model == "observed"].sort_values("valid_time_utc")
        dates = pd.to_datetime(observed.valid_time_utc)
        axis.plot(dates, observed.surge_cm, color=COLORS["observed"], linewidth=1.8, label="Observed")
        observed_peak = observed.iloc[int(observed.surge_cm.to_numpy().argmax())]
        axis.scatter([pd.Timestamp(observed_peak.valid_time_utc)], [observed_peak.surge_cm], color=COLORS["observed"], marker="o", s=19, zorder=4)
        for index, name in enumerate(names):
            model = section.loc[section.model == name].sort_values("valid_time_utc")
            axis.plot(dates, model.surge_cm, color=COLORS[name], linestyle=("-", "--", "-.", ":")[index], label=name.replace("_", " "))
            peak = model.iloc[int(model.surge_cm.to_numpy().argmax())]
            axis.scatter([pd.Timestamp(peak.valid_time_utc)], [peak.surge_cm], color=COLORS[name], s=15, zorder=4)
        axis.set_ylabel("Surge (cm)")
        axis.text(0.01, 0.97, f"Origin t0{offset:+d} h", transform=axis.transAxes, ha="left", va="top")
        axis.grid(alpha=0.45)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[0].legend(ncol=3, loc="lower right")
    axes[-1].set_xlabel("UTC valid time")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def summarise_exact_leads(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, horizon), group in metrics.groupby(["model", "horizon_hours"], sort=True):
        errors = group.endpoint_error_cm.to_numpy(dtype=float)
        rows.append({
            "model": model, "lead_hours": int(horizon), "n_origins": len(errors),
            "lead_rmse_cm": float(np.sqrt(np.mean(errors**2))),
            "lead_mae_cm": float(np.mean(np.abs(errors))),
            "lead_bias_cm": float(np.mean(errors)),
        })
    return pd.DataFrame(rows)


def _plot_errors(metrics: pd.DataFrame, lead_metrics: pd.DataFrame, path: Path) -> None:
    setup_journal_style()
    names = [name for name in ("ridge", *MAIN_MODELS) if name in metrics.model.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.15), layout="constrained")
    fields = (
        ("lead_rmse_cm", "Exact-lead RMSE (cm)"),
        ("absolute_peak_height_error_cm", "Absolute peak-height error (cm)"),
        ("peak_time_error_hours", "Peak-time error (h)"),
    )
    for panel, (axis, (field, label)) in enumerate(zip(axes, fields)):
        for index, name in enumerate(names):
            if panel == 0:
                series = lead_metrics.loc[lead_metrics.model == name].set_index("lead_hours")[field]
            else:
                series = metrics.loc[metrics.model == name].groupby("horizon_hours")[field].mean()
            axis.plot(series.index, series.values, color=COLORS[name], marker=("o", "s", "^", "d", "v")[index], linestyle=("-", "--", "-.", ":", "--")[index], label=name.replace("_", " "))
        axis.set(xlabel="Horizon (h)", ylabel=label, xticks=EVALUATION_HOURS)
        axis.grid(alpha=0.45)
    axes[0].legend(loc="best")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def run_one(row: pd.Series, track: pd.DataFrame, args: argparse.Namespace, output: Path) -> pd.DataFrame:
    final_dir = output / str(row.event_id)
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing event: {final_dir}")
    with tempfile.TemporaryDirectory(prefix=".typhoon_event_", dir=output) as temporary:
        metrics = _run_one_impl(row, track, args, Path(temporary))
        if final_dir.exists():
            raise FileExistsError(f"Event appeared during computation: {final_dir}")
        (Path(temporary) / str(row.event_id)).rename(final_dir)
    return metrics


def _run_one_impl(row: pd.Series, track: pd.DataFrame, args: argparse.Namespace, output: Path) -> pd.DataFrame:
    station_id = str(row.station)
    station = get_station_config(station_id)
    event_dir = output / str(row.event_id)
    if event_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing event: {event_dir}")
    missing = missing_model_files(station_id, args.seed)
    if missing:
        raise FileNotFoundError("Missing required checkpoints: " + "; ".join(missing))
    atmosphere, surge, times = read_station_year(station_id, args.year)
    t0 = pd.Timestamp(row.closest_time_utc)
    coverage = inspect_event_coverage(times, surge, atmosphere, t0, args.year)
    if not coverage["data_complete"]:
        raise ValueError(f"Incomplete event data: {coverage['invalid_origins']}")
    if not np.isfinite(coverage["max_observed_surge_cm"]) or coverage["max_observed_surge_cm"] <= 0:
        raise ValueError("No positive observed surge peak in the fixed event window")
    origins = np.asarray(coverage["valid_origin_positions"], dtype=np.int64)
    ridge_path, _, rollout_path, extras = model_files(station_id, args.seed)
    device = _device(args.device)
    models, checkpoints, paths = load_models(
        station.model_root / f"formal_seed{args.seed}", device,
        rollout_path, ("cnn_gru",), station_id, extras,
    )
    if "cnn_gru_rollout6" not in models:
        raise ValueError("Base rollout checkpoint does not identify as CNN-GRU rollout-6")
    ridge = joblib.load(ridge_path)
    predictions = recursive_predictions(
        atmosphere, surge, origins, summarise_atmosphere(atmosphere), ridge,
        models, next(iter(checkpoints.values()))["scalers"], 24, 72, 3, device,
    )
    forecast_rows = []
    metric_rows = []
    for origin_index, origin in enumerate(origins):
        origin_time = times[origin]
        offset = ORIGIN_OFFSETS_HOURS[origin_index]
        observed = np.asarray(surge[origin:origin + 72], dtype=float)
        for model_name, values in (("observed", observed), *[(name, prediction[origin_index]) for name, prediction in predictions.items()]):
            for lead, value in enumerate(values, start=1):
                forecast_rows.append({
                    "event_id": row.event_id, "station": station_id, "storm_sid": row.storm_sid,
                    "origin_offset_hours": offset, "origin_time_utc": origin_time.isoformat(),
                    "lead_hours": lead, "valid_time_utc": times[origin + lead - 1].isoformat(),
                    "model": model_name, "surge_cm": float(value) * 100,
                })
            if model_name != "observed":
                for horizon in EVALUATION_HOURS:
                    metric_rows.append({
                        "event_id": row.event_id, "station": station_id, "storm_name": row.storm_name,
                        "storm_sid": row.storm_sid, "origin_offset_hours": offset,
                        "origin_time_utc": origin_time.isoformat(), "model": model_name,
                        **score_forecast_window(observed, values, origin_time, horizon),
                    })
    forecasts = pd.DataFrame(forecast_rows)
    metrics = pd.DataFrame(metric_rows)
    lead_metrics = summarise_exact_leads(metrics)
    event_dir.mkdir(parents=True)
    forecasts.to_csv(event_dir / "forecasts_72h.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(event_dir / "event_metrics.csv", index=False, encoding="utf-8-sig")
    lead_metrics.to_csv(event_dir / "exact_lead_metrics.csv", index=False, encoding="utf-8-sig")
    track.to_csv(event_dir / "best_track_points.csv", index=False)
    local_track = track.loc[
        (track.time >= t0 - pd.Timedelta(hours=72))
        & (track.time < t0 + pd.Timedelta(hours=48))
    ]
    _plot_track(local_track if len(local_track) >= 2 else track, station_id, t0, event_dir / "figure1_best_track.png")
    _plot_process(forecasts, event_dir / "figure2_fixed_origin_forecasts.png")
    _plot_errors(metrics, lead_metrics, event_dir / "figure3_event_errors.png")
    (event_dir / "experiment_config.json").write_text(json.dumps({
        "station": station_id, "storm_sid": str(row.storm_sid), "t0_utc": t0.isoformat(),
        "closest_distance_km": float(row.closest_distance_km),
        "observed_peak_cm_in_120h_window": coverage["max_observed_surge_cm"],
        "origin_offsets_hours": ORIGIN_OFFSETS_HOURS, "horizons_hours": EVALUATION_HOURS,
        "ridge_path": str(ridge_path), "ridge_sha256": sha256_file(ridge_path),
        "neural_checkpoints": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "known_future_era5": True, "future_observed_surge_as_input": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    focused = metrics.loc[
        (metrics.horizon_hours == 72)
        & metrics.model.isin(("ridge", "cnn_gru", "cnn_gru_rollout6", *extras))
    ]
    lines = [
        f"# {station.name} — {row.storm_name} ({row.storm_sid})",
        "",
        f"- 最接近站点的路径时刻：{row.nearest_track_time_utc} UTC；固定整点 t0：{t0.isoformat()} UTC。",
        f"- 最近路径距离：{float(row.closest_distance_km):.1f} km。人工核验说明：{row.review_note}",
        f"- 120小时事件窗口最大实测正向增水：{coverage['max_observed_surge_cm']:.2f} cm。",
        "- 本实验为已知未来 ERA5 气象强迫的历史回算，不是业务预报。1997年为追加历史案例分析，并非全新盲测。",
        "- 三次固定起报分别独立预测72小时，以下为每次起报后前72小时的指标；重叠起报不能算作独立台风个数。",
        "- 峰值高度和时间误差均在各起报后的相同72小时窗口内定义；时间误差为预测峰值时间减实测峰值时间。",
        "- t0−72h起报的72小时窗口在t0前结束，不一定覆盖整场台风的最大峰值；不得将窗内峰值解释为整场峰值。",
        "- exact_lead_metrics.csv是3个固定起报在精确24/48/72小时处的描述性RMSE/MAE；与每条曲线前H小时累计RMSE不同。",
        "",
        "| 起报相对t0(h) | 模型 | RMSE(cm) | MAE(cm) | 峰高误差(cm) | 峰时误差(h) | 低估(cm) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, item in focused.sort_values(["origin_offset_hours", "model"]).iterrows():
        lines.append(
            f"| {item.origin_offset_hours} | {item.model} | {item.rmse_cm:.2f} | "
            f"{item.mae_cm:.2f} | {item.peak_height_error_cm:.2f} | "
            f"{item.peak_time_error_hours:+.0f} | {item.peak_underestimation_cm:.2f} |"
        )
    lines.extend([
        "", "## 解读边界", "",
        "图2分别展示三次起报的完整曲线，未将不同起报拼接。图3的点是这些固定起报的描述性平均，不代表独立样本置信区间。",
        "最佳路径靠近、风场和实测过程需联合审查；不能仅凭P99或路径距离证明增水完全由该台风导致。",
        "仅靠此事件无法判断模型是否达到实时预报要求；未来需用实际发布时可得的GFS和实测增水，并存档发布时间。",
    ])
    (event_dir / "研究报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def run(args: argparse.Namespace) -> None:
    output = args.results_dir.resolve()
    checklist = output / "candidate_events.csv"
    metadata_file = output / "source_and_protocol.json"
    if not checklist.is_file() or not metadata_file.is_file():
        raise FileNotFoundError("Run scan first, then manually review candidate_events.csv")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata["best_track_sha256"] != sha256_file(args.track_csv):
        raise ValueError("Best-track file changed since screening; run scan again")
    if metadata["year"] != args.year or metadata["seed"] != args.seed:
        raise ValueError("Year or seed differs from the screening protocol")
    candidates = pd.read_csv(checklist, encoding="utf-8-sig").fillna("")
    approved = candidates.loc[candidates.review_decision.astype(str).str.lower().str.strip() == "yes"]
    if approved.empty:
        raise ValueError("No manually confirmed typhoon events; mark review_decision=yes after review")
    if approved.event_id.duplicated().any():
        raise ValueError("Duplicate approved event_id in checklist")
    tracks = read_ibtracs_wp(args.track_csv, args.year)
    expected = build_candidates(
        tracks, [get_station_config(name) for name in metadata["stations"]],
        float(metadata["initial_distance_km"]),
    ).set_index("event_id")
    all_metrics = []
    failures = []
    for _, row in approved.iterrows():
        try:
            if row.event_id not in expected.index or row.station not in args.stations:
                raise ValueError("Event is not in the original screened best-track list")
            original = expected.loc[row.event_id]
            if str(row.station) != str(original.station) or str(row.storm_sid) != str(original.storm_sid):
                raise ValueError("Station or storm ID was edited after screening")
            if str(row.closest_time_utc) != str(original.closest_time_utc):
                raise ValueError("t0 was edited; re-screen instead of changing the fixed evaluation window")
            if str(row.review_note).strip() == "":
                raise ValueError("A manual attribution/data review note is required")
            track = tracks.loc[tracks.SID.astype(str) == str(row.storm_sid)].copy()
            all_metrics.append(run_one(row, track, args, output))
            print(f"Completed {row.event_id}")
        except Exception as error:
            failures.append({"event_id": row.event_id, "reason": str(error)})
            print(f"Skipped {row.event_id}: {error}")
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(output / "all_event_metrics.csv", index=False, encoding="utf-8-sig")
    approved.to_csv(output / "confirmed_event_list.csv", index=False, encoding="utf-8-sig")
    (output / "run_status.json").write_text(json.dumps({
        "completed_event_ids": [str(frame.event_id.iloc[0]) for frame in all_metrics],
        "failures": failures,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed {len(all_metrics)} real event(s); failed/skipped {len(failures)}. Package: {output}")


def main() -> None:
    args = parser().parse_args()
    if args.year != 1997:
        raise ValueError("This protocol is fixed to the 1997 additional historical test")
    if args.command == "scan":
        scan(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
