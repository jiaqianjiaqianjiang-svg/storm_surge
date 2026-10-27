"""Reuse the established station pipeline with explicit validation/test gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

from .station_config import MODULE_ROOT, get_station_config
from .run_validation_workflow import require_passed_audit


def repository_root() -> Path:
    return next(parent for parent in (MODULE_ROOT, *MODULE_ROOT.parents) if (parent / ".git").exists())


def result_package(station_id: str, year: int, seed: int) -> Path:
    return repository_root() / "reports" / "experiment_results" / f"{station_id}_short_term_{year}_seed{seed}"


def model_paths(station_id: str, seed: int) -> dict[str, Path]:
    station = get_station_config(station_id)
    formal = station.model_root / f"formal_seed{seed}"
    return {
        "ridge": station.model_root / "baselines" / "ridge_pipeline.joblib",
        "cnn_gru": formal / "cnn_gru" / "best_model.pth",
        "cnn_gru_rollout6": formal / "cnn_gru_rollout6" / "best_model.pth",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_validation_lock(station_id: str, seed: int) -> Path:
    station = get_station_config(station_id)
    require_passed_audit(station_id)
    package = result_package(station_id, station.validation_year, seed)
    if not (package / "RESULTS_SUMMARY.md").is_file():
        raise FileNotFoundError(f"Complete validation result package first: {package}")
    paths = model_paths(station_id, seed)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing selected model files: " + "; ".join(missing))
    lock = station.model_root / f"formal_seed{seed}" / "validation_model_lock.json"
    if lock.exists():
        raise FileExistsError(f"Validation model selection was already locked: {lock}")
    data = {
        "station_id": station_id,
        "seed": seed,
        "validation_year": station.validation_year,
        "test_year": station.test_year,
        "models": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()},
        "selection_rule": "Core models fixed using validation results only; test results must not tune weights or hyperparameters.",
    }
    lock.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return lock


def require_validation_lock(station_id: str, seed: int) -> None:
    station = get_station_config(station_id)
    lock = station.model_root / f"formal_seed{seed}" / "validation_model_lock.json"
    if not lock.is_file():
        raise FileNotFoundError(f"Lock validated models before test/event hindcast: {lock}")
    data = json.loads(lock.read_text(encoding="utf-8"))
    if data.get("station_id") != station_id or data.get("seed") != seed:
        raise ValueError(f"Model lock belongs to another station/seed: {lock}")
    for name, item in data["models"].items():
        path = model_paths(station_id, seed)[name]
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"Locked {name} checkpoint changed or is missing: {path}")


def build_command(stage: str, station_id: str, seed: int, device: str, language: str,
                  track_csv: Path | None, events_dir: Path | None) -> list[str]:
    station = get_station_config(station_id)
    def module_command(name: str) -> list[str]:
        return [sys.executable, "-m", f"src.xiamen_forecast.{name}"]

    if stage == "prepare":
        return [*module_command("prepare_xiamen"), "--station", station_id]
    if stage == "audit":
        return [*module_command("audit_prepared_dataset"), "--station", station_id]
    if stage == "validation":
        return [*module_command("run_validation_workflow"), "--station", station_id,
                "--profile", "core", "--seed", str(seed), "--device", device, "--language", language]
    if stage == "test":
        return [*module_command("run_final_test"), "--station", station_id,
                "--profile", "core", "--seed", str(seed), "--device", device, "--language", language]
    if stage in {"scan-events", "run-events"}:
        if track_csv is None or events_dir is None:
            raise ValueError("--track-csv and --events-dir are required for event stages")
        command = [*module_command("run_typhoon_events"),
                   "scan" if stage == "scan-events" else "run",
                   "--track-csv", str(track_csv), "--results-dir", str(events_dir),
                   "--year", str(station.test_year), "--seed", str(seed),
                   "--stations", station_id]
        if stage == "run-events":
            command.extend(["--device", device])
        return command
    raise ValueError(f"Unknown workflow stage: {stage}")


def write_core_review(station_id: str, seed: int, events_dir: Path | None) -> Path:
    station = get_station_config(station_id)
    source = result_package(station_id, station.test_year, seed)
    rolling_file = source / "rolling_metrics_long.csv"
    if not rolling_file.is_file():
        raise FileNotFoundError(f"Missing compact test results: {rolling_file}")
    destination = repository_root() / "reports" / "experiment_results" / f"{station_id}_core_review_{station.test_year}_seed{seed}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite a previous review: {destination}")
    rolling = pd.read_csv(rolling_file)
    required_columns = {
        "lead_hours", "model", "rmse_cm", "mae_cm", "positive_p99_n",
        "positive_p99_threshold_cm", "positive_p99_rmse_cm",
        "positive_p99_bias_cm", "positive_p99_underestimation_ratio",
    }
    if not required_columns.issubset(rolling.columns):
        raise ValueError(
            "Compact rolling results lack core/P99 metrics: "
            + ", ".join(sorted(required_columns - set(rolling.columns)))
        )
    selected = rolling.loc[
        rolling.lead_hours.isin([1, 24, 48, 72])
        & rolling.model.isin(["persistence", "ridge", "cnn_gru", "cnn_gru_rollout6"])
    ].copy()
    if selected.empty:
        raise ValueError("No selected core-model rolling metrics in compact result package")
    event_metrics = None
    if events_dir is not None and (events_dir / "all_event_metrics.csv").is_file():
        event_metrics = pd.read_csv(events_dir / "all_event_metrics.csv")
    destination.mkdir(parents=True)
    selected.to_csv(destination / "core_rolling_metrics.csv", index=False, encoding="utf-8-sig")
    if event_metrics is not None:
        event_metrics.to_csv(destination / "event_metrics_by_origin.csv", index=False, encoding="utf-8-sig")
        case_summary = (
            event_metrics.groupby(["event_id", "station", "model", "horizon_hours"], as_index=False)
            .agg(
                n_origins=("origin_time_utc", "nunique"),
                mean_window_rmse_cm=("rmse_cm", "mean"),
                mean_window_mae_cm=("mae_cm", "mean"),
                mean_peak_height_error_cm=("peak_height_error_cm", "mean"),
                mean_absolute_peak_height_error_cm=("absolute_peak_height_error_cm", "mean"),
                mean_peak_time_error_hours=("peak_time_error_hours", "mean"),
            )
        )
        case_summary.to_csv(destination / "event_case_summary.csv", index=False, encoding="utf-8-sig")
    figure_sources = {
        "one_step_model_comparison.png": source / "test_model_comparison.png",
        "rolling_rmse.png": source / "rmse_vs_lead.png",
    }
    for name, path in figure_sources.items():
        if path.is_file():
            shutil.copy2(path, destination / name)
    if events_dir is not None:
        for path in sorted(events_dir.glob("*/figure2_fixed_origin_forecasts.png")):
            shutil.copy2(path, destination / f"{path.parent.name}_fixed_origin_forecasts.png")
    peak_file = source / "journal_figures" / "peak_metrics.json"
    p99 = None
    if peak_file.is_file():
        entries = json.loads(peak_file.read_text(encoding="utf-8"))
        p99 = next((item for item in entries if float(item.get("percentile", -1)) == 99), None)
    lines = [
        f"# {station.name} 短期非潮增水核心实验",
        "",
        f"- 训练期：{station.train_start_year}–{station.train_end_year}；验证：{station.validation_year}；测试：{station.test_year}。",
        "- 输入历史24小时，滚动时效1–72小时；模型为Persistence、Ridge、CNN-GRU及Rollout-6。",
        "- 使用未来ERA5再分析场，属于历史回算，不是实时业务预报。测试集不用于重新调参。",
        "- core_rolling_metrics.csv记录全年相同提前量的RMSE等指标。",
        "",
        "## 全年滚动回算",
        "",
        "| 模型 | 提前量(h) | RMSE(cm) | MAE(cm) |",
        "|---|---:|---:|---:|",
    ]
    for _, row in selected.sort_values(["model", "lead_hours"]).iterrows():
        lines.append(f"| {row.model} | {int(row.lead_hours)} | {row.rmse_cm:.2f} | {row.mae_cm:.2f} |")
    lines.append("")
    lines.extend([
        "## 正向强增水P99（相同提前量的共同起报样本）",
        "",
        "此P99阈值由该提前量测试观测样本计算，仅用于事后分层描述，不用于训练、早停或模型选择。",
        "",
        "| 模型 | 提前量(h) | 强增水样本n | P99阈值(cm) | P99 RMSE(cm) | 偏差(cm) | 低估比例 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in selected.loc[selected.lead_hours.isin([24, 48, 72])].sort_values(["model", "lead_hours"]).iterrows():
        lines.append(
            f"| {row.model} | {int(row.lead_hours)} | {int(row.positive_p99_n)} | "
            f"{row.positive_p99_threshold_cm:.2f} | {row.positive_p99_rmse_cm:.2f} | "
            f"{row.positive_p99_bias_cm:+.2f} | {row.positive_p99_underestimation_ratio:.1%} |"
        )
    lines.append("")
    if p99 is not None:
        lines.append(
            f"- 另列CNN-GRU一步预测P99描述性统计：阈值{p99['threshold_cm']:.2f} cm，"
            f"RMSE {p99['peak_rmse']:.2f} cm，偏差{p99['peak_bias']:.2f} cm。"
            "该分位数来自一步测试结果，与上表滚动共同起报样本口径不同，不用于模型选择。"
        )
    else:
        lines.append("- 一步预测P99指标未在精简结果包中找到，暂不填写数值。")
    if event_metrics is None:
        lines.append("- 尚无经人工核验的合格台风事件结果；候选事件不能冒充正式台风案例。")
    else:
        lines.extend([
            f"- 已纳入{event_metrics.event_id.nunique()}场经人工核验的事件；同场的多个起报并非独立台风样本。",
            "- event_metrics_by_origin.csv包含各起报的24/48/72小时窗口累计RMSE/MAE、峰高与峰时误差。",
            "- event_case_summary.csv仅对同场固定起报做描述性均值，不把它们算作独立台风。",
            "- 单次台风曲线按起报时刻分别绘制，不能拼接为一条业务预报。",
            "",
            "## 台风事件（各场固定起报描述性汇总）",
            "",
            "| 事件 | 模型 | 窗口(h) | 起报数 | 平均RMSE(cm) | 平均峰高偏差(cm) | 平均峰时偏差(h) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for _, row in case_summary.sort_values(["event_id", "model", "horizon_hours"]).iterrows():
            lines.append(
                f"| {row.event_id} | {row.model} | {int(row.horizon_hours)} | "
                f"{int(row.n_origins)} | {row.mean_window_rmse_cm:.2f} | "
                f"{row.mean_peak_height_error_cm:+.2f} | {row.mean_peak_time_error_hours:+.1f} |"
            )
    lines.extend(["", "## 结论边界", "", "请结合各站点气象与最佳路径人工核验报告解释峰值归因；不能仅凭路径距离或P99认定台风致灾。", ""])
    (destination / "结论报告.md").write_text("\n".join(lines), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "validation", "lock", "test", "scan-events", "run-events", "report"))
    parser.add_argument("--station", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--track-csv", type=Path)
    parser.add_argument("--events-dir", type=Path)
    args = parser.parse_args()
    station = get_station_config(args.station)
    if args.stage == "prepare":
        if station.processed_root.exists() and any(station.processed_root.iterdir()):
            raise FileExistsError(f"Refusing to overwrite prepared station data: {station.processed_root}")
        stages = ("prepare", "audit")
    elif args.stage == "lock":
        print(f"Validation models locked: {create_validation_lock(station.station_id, args.seed)}")
        return
    elif args.stage == "report":
        print(f"Core review: {write_core_review(station.station_id, args.seed, args.events_dir)}")
        return
    else:
        if args.stage == "validation":
            require_passed_audit(station.station_id)
        if args.stage in {"test", "scan-events", "run-events"}:
            require_validation_lock(station.station_id, args.seed)
        if args.stage in {"scan-events", "run-events"} and not (
            result_package(station.station_id, station.test_year, args.seed) / "RESULTS_SUMMARY.md"
        ).is_file():
            raise FileNotFoundError("Complete the independent-year test before screening typhoon cases")
        if args.stage == "test" and result_package(station.station_id, station.test_year, args.seed).exists():
            raise FileExistsError("Test result package already exists; refusing to overwrite it")
        if args.stage == "scan-events" and args.events_dir is not None and args.events_dir.exists():
            raise FileExistsError(f"Event scan output already exists: {args.events_dir}")
        stages = (args.stage,)
    for stage in stages:
        command = build_command(stage, station.station_id, args.seed, args.device,
                                args.language, args.track_csv, args.events_dir)
        print("[RUN] " + subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=MODULE_ROOT, check=True)


if __name__ == "__main__":
    main()
