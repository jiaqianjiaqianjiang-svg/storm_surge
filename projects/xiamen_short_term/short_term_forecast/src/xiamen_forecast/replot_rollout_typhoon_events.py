"""只读归档预测，生成测试期台风的双Rollout模型对照。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.short_term_forecast.journal_figures.style import DPI, setup_journal_style


ROOT = Path(__file__).resolve().parents[5] / "reports" / "experiment_results"
SOURCES = (
    ROOT / "typhoon_events_1997_mse_whuber_comparison_seed42",
    ROOT / "kanmen_1996_1997_seed42_compact" / "typhoon_1997",
)
DESTINATION = ROOT / "historical_typhoon_rollout_only_review"
MODELS = ("cnn_gru_rollout6", "whuber_rollout6")
LABELS = {MODELS[0]: "MSE Rollout-6", MODELS[1]: "WHuber Rollout-6"}
COLORS = {MODELS[0]: "#A64B45", MODELS[1]: "#66518A"}
LINESTYLES = {MODELS[0]: "-", MODELS[1]: "--"}
ORIGINS = (-72, -48, -24)


def load_thresholds(station: str, root: Path = ROOT) -> tuple[dict[str, float], str]:
    training = root / "china_three_station_event_distribution_v1" / "training_thresholds.json"
    if training.is_file():
        values = json.loads(training.read_text(encoding="utf-8-sig"))
        if station in values:
            return {q: float(values[station][f"{q}_m"]) * 100 for q in ("p95", "p99")}, "training_target_all_valid"
    validation = root / "kanmen_1996_1997_seed42_compact" / "1996" / "rolling_metrics_long.csv"
    if station == "kanmen" and validation.is_file():
        table = pd.read_csv(validation)
        row = table.loc[(table.model == MODELS[0]) & (table.lead_hours == 1)]
        if len(row) == 1:
            print("warning: kanmen has no archived training P95/P99; using fixed 1996 validation lead-1 positive thresholds")
            return {q: float(row.iloc[0][f"positive_{q}_threshold_cm"]) for q in ("p95", "p99")}, "validation_1996_lead1_positive"
    return {}, "event_positive_descriptive"


def aligned_forecasts(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    required = {"event_id", "station", "origin_offset_hours", "lead_hours", "valid_time_utc", "model", "surge_cm"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing forecast columns in {path}: {sorted(required - set(raw.columns))}")
    raw = raw.loc[raw.model.isin(("observed", *MODELS))].copy()
    keys = ["event_id", "station", "origin_offset_hours", "lead_hours", "valid_time_utc"]
    if raw.duplicated(keys + ["model"]).any():
        raise ValueError(f"Duplicate model/lead rows: {path}")
    wide = raw.pivot(index=keys, columns="model", values="surge_cm").reset_index()
    wide.columns.name = None
    if set(("observed", *MODELS)) - set(wide.columns):
        raise ValueError(f"Missing rollout or observed series: {path}")
    wide = wide.sort_values(["origin_offset_hours", "lead_hours"])
    if set(wide.origin_offset_hours) != set(ORIGINS):
        raise ValueError(f"Expected three fixed origins: {path}")
    for origin, group in wide.groupby("origin_offset_hours"):
        if group.lead_hours.tolist() != list(range(1, 73)) or group[["observed", *MODELS]].isna().any().any():
            raise ValueError(f"Incomplete paired 72-hour forecast at origin {origin}: {path}")
    return raw, wide


def extreme_rows(wide: pd.DataFrame, storm: str, thresholds: dict[str, float], source: str) -> pd.DataFrame:
    output = []
    for origin, group in wide.groupby("origin_offset_hours", sort=True):
        observed = group.observed.to_numpy(dtype=float)
        # Event fallback uses unique valid hours, never duplicated overlapping origins.
        for model in MODELS:
            error = group[model].to_numpy(dtype=float) - observed
            for name in ("all", "p95", "p99"):
                threshold = thresholds.get(name, np.nan)
                mask = np.ones(len(group), dtype=bool) if name == "all" else observed >= threshold
                values = error[mask]
                negative = values[values < 0]
                output.append({
                    "event_id": group.event_id.iloc[0], "station": group.station.iloc[0],
                    "storm_name": storm, "origin_offset_hours": origin, "model": model,
                    "sample_group": name, "threshold_cm": threshold, "threshold_source": source,
                    "n_samples": len(values),
                    "rmse_cm": float(np.sqrt(np.mean(values**2))) if len(values) else np.nan,
                    "mae_cm": float(np.mean(np.abs(values))) if len(values) else np.nan,
                    "bias_cm": float(np.mean(values)) if len(values) else np.nan,
                    "underestimation_ratio": float(np.mean(values < 0)) if len(values) else np.nan,
                    "mean_underestimation_cm": float(-np.mean(negative)) if len(negative) else 0.0 if len(values) else np.nan,
                })
    return pd.DataFrame(output)


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def plot_series(wide: pd.DataFrame, output: Path, kind: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for ax, origin in zip(axes, ORIGINS):
        group = wide.loc[wide.origin_offset_hours == origin]
        hours = group.lead_hours.to_numpy()
        observed = group.observed.to_numpy()
        if kind == "process":
            x = pd.to_datetime(group.valid_time_utc, utc=True)
            ax.plot(x, observed, color="black", label="Observed", lw=1.8)
            ax.plot(x.iloc[int(np.argmax(observed))], np.max(observed), "o", color="black", ms=5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=mdates.UTC))
            ax.set_xlabel("Valid time (UTC)")
            ax.set_ylabel("Surge (cm)")
        else:
            x = hours
            ax.set_xlabel("Lead time (h)")
            ax.set_ylabel("Error (cm)" if kind == "error" else "Cumulative RMSE (cm)")
            if kind == "error":
                ax.axhline(0, color="0.4", lw=0.8)
        for model in MODELS:
            error = group[model].to_numpy() - observed
            values = group[model] if kind == "process" else error if kind == "error" else np.sqrt(np.cumsum(error**2) / hours)
            ax.plot(x, values, color=COLORS[model], ls=LINESTYLES[model], label=LABELS[model], lw=1.5)
            if kind == "process":
                peak = int(np.argmax(group[model].to_numpy()))
                ax.plot(x.iloc[peak], group[model].iloc[peak], "o", color=COLORS[model], ms=5)
        ax.set_title(f"Origin: t0{origin}h", loc="left", fontsize=10)
        ax.grid(color="0.9", lw=0.6)
    axes[0].legend(ncol=3, frameon=False, fontsize=9)
    name = {"process": "figure2_rollout_process_only.png", "error": "figure3_rollout_hourly_errors.png", "cumulative": "figure4_rollout_cumulative_rmse.png"}[kind]
    save_plot(fig, output / name)


def plot_extremes(table: pd.DataFrame, output: Path) -> None:
    groups = ("all", "p95", "p99")
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True)
    x = np.arange(3)
    for ax, metric, title in zip(axes, ("rmse_cm", "mean_underestimation_cm"), ("RMSE (cm)", "Mean underestimation (cm)")):
        for offset, model in ((-0.15, MODELS[0]), (0.15, MODELS[1])):
            values = [table.loc[(table.model == model) & (table.sample_group == group) & (table.n_samples > 0), metric].mean() for group in groups]
            ax.bar(x + offset, values, width=0.29, color=COLORS[model], hatch="///" if model == MODELS[1] else None, label=LABELS[model])
        ax.set_xticks(x, ("All", "P95", "P99"))
        ax.set_ylabel(title)
        ax.grid(axis="y", color="0.9", lw=0.6)
    axes[0].legend(frameon=False)
    save_plot(fig, output / "figure5_extreme_error_comparison.png")


def event_table(metrics: pd.DataFrame, extremes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (origin, model), group in metrics.loc[metrics.model.isin(MODELS)].groupby(["origin_offset_hours", "model"]):
        values = {int(row.horizon_hours): row for row in group.itertuples()}
        if set(values) != {24, 48, 72}:
            raise ValueError("Incomplete event metric horizons")
        peak = values[72]
        selected = extremes.loc[(extremes.origin_offset_hours == origin) & (extremes.model == model)].set_index("sample_group")
        rows.append({"event_id": peak.event_id, "station": peak.station, "storm_name": peak.storm_name,
                     "origin_offset_hours": origin, "model": model,
                     **{f"rmse{h}_cm": float(values[h].rmse_cm) for h in (24, 48, 72)},
                     "peak_height_error_cm": peak.peak_height_error_cm,
                     "absolute_peak_height_error_cm": peak.absolute_peak_height_error_cm,
                     "peak_time_error_hours": peak.peak_time_error_hours,
                     **{f"{q}_rmse_cm": selected.loc[q, "rmse_cm"] for q in ("p95", "p99")},
                     **{f"{q}_underestimation_ratio": selected.loc[q, "underestimation_ratio"] for q in ("p95", "p99")},
                     "threshold_source": selected.loc["p95", "threshold_source"]})
    return pd.DataFrame(rows)


def write_event_report(table: pd.DataFrame, extremes: pd.DataFrame, output: Path) -> None:
    first = table.iloc[0]
    lines = [f"# {first.station} {first.storm_name}：滚动模型事件分析", "",
             "已知未来ERA5再分析强迫下的1997追加历史回算，不是实时预报或新的盲测。",
             "三次固定起报属于同一场台风；下面数值按起报分列，不当作三个独立事件。", "",
             f"P95/P99阈值来源：`{first.threshold_source}`。按每次起报的72小时观测筛选，重叠时次不用于独立样本推断。", "",
             "| 起报 | 模型 | 前24h RMSE | 前48h RMSE | 前72h RMSE | 72h窗口峰高偏差 | 峰时偏差 | P95 RMSE | P99 RMSE |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table.sort_values(["origin_offset_hours", "model"]).itertuples():
        fmt = lambda v: f"{v:.2f}" if pd.notna(v) else "无样本"
        lines.append(f"| t0{row.origin_offset_hours}h | {LABELS[row.model]} | {fmt(row.rmse24_cm)} | {fmt(row.rmse48_cm)} | {fmt(row.rmse72_cm)} | {fmt(row.peak_height_error_cm)} | {row.peak_time_error_hours:+d} h | {fmt(row.p95_rmse_cm)} | {fmt(row.p99_rmse_cm)} |")
    lines += ["", "## 极端样本与低估", "", "| 起报 | 模型 | 组别 | 样本数 | RMSE (cm) | 低估比例 | 平均低估幅度 (cm) |", "|---|---|---|---:|---:|---:|---:|"]
    for row in extremes.sort_values(["origin_offset_hours", "sample_group", "model"]).itertuples():
        lines.append(f"| t0{row.origin_offset_hours}h | {LABELS[row.model]} | {row.sample_group.upper()} | {row.n_samples} | {row.rmse_cm:.2f} | {row.underestimation_ratio:.1%} | {row.mean_underestimation_cm:.2f} |" if row.n_samples else f"| t0{row.origin_offset_hours}h | {LABELS[row.model]} | {row.sample_group.upper()} | 0 | 无样本 | 无样本 | 无样本 |")
    lines += ["", "P95/P99零样本不补造指标。峰时误差仅指同一72小时窗口内的最大值时间差，不等于整个过程的系统性相位滞后。",
              "过程图、逐小时误差和累计RMSE图分别保留原始波动，不平滑或拼接起报。", ""]
    (output / "研究报告.md").write_text("\n".join(lines), encoding="utf-8")


def run(sources: tuple[Path, ...] = SOURCES, destination: Path = DESTINATION) -> pd.DataFrame:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing review: {destination}")
    events = [p for source in sources for p in sorted(source.iterdir()) if p.is_dir() and (p / "forecasts_72h.csv").is_file()]
    if not events:
        raise FileNotFoundError("No archived event forecasts found")
    prepared = []
    for event in events:
        print(f"Processing {event.name}")
        _, wide = aligned_forecasts(event / "forecasts_72h.csv")
        metrics = pd.read_csv(event / "event_metrics.csv")
        station = str(wide.station.iloc[0])
        storm = str(metrics.storm_name.iloc[0])
        thresholds, source = load_thresholds(station, ROOT)
        if not thresholds:
            print(f"warning: {event.name} lacks fixed P95/P99; event-based positive thresholds are descriptive only")
            unique = wide.drop_duplicates("valid_time_utc")
            positive = unique.loc[unique.observed > 0, "observed"]
            if positive.empty:
                raise ValueError(f"No positive observations for fallback: {event}")
            thresholds = {q: float(positive.quantile(p)) for q, p in (("p95", .95), ("p99", .99))}
        extremes = extreme_rows(wide, storm, thresholds, source)
        table = event_table(metrics, extremes)
        prepared.append((event, wide, extremes, table))
    setup_journal_style()
    destination.mkdir(parents=True)
    all_tables = []
    for event, wide, extremes, table in prepared:
        out = destination / event.name
        out.mkdir()
        for kind in ("process", "error", "cumulative"):
            plot_series(wide, out, kind)
        plot_extremes(extremes, out)
        extremes.to_csv(out / "extreme_error_summary.csv", index=False, encoding="utf-8-sig")
        print(f"  {out / 'extreme_error_summary.csv'}")
        write_event_report(table, extremes, out)
        print(f"  {out / '研究报告.md'}")
        all_tables.append(table)
    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(destination / "all_events_rollout_comparison.csv", index=False, encoding="utf-8-sig")
    lines = ["# 测试期台风：双Rollout模型对比", "", "仅使用归档预测；原图和旧结果未改动。未来气象为ERA5再分析，1997为追加历史案例分析。", "阈值来源逐行记录；没有样本的P95/P99指标保持空白。不同起报重叠，不能作为独立台风样本。", ""]
    for event_id, group in combined.groupby("event_id", sort=True):
        lines.append(f"## {event_id}")
        lines.append("")
        for model in MODELS:
            subset = group.loc[group.model == model]
            lines.append(f"- {LABELS[model]}：三次起报前72h窗口RMSE均值 {subset.rmse72_cm.mean():.2f} cm；峰高绝对误差均值 {subset.absolute_peak_height_error_cm.mean():.2f} cm。")
        for q in ("p95", "p99"):
            mse = group.loc[group.model == MODELS[0], f"{q}_rmse_cm"].mean()
            huber = group.loc[group.model == MODELS[1], f"{q}_rmse_cm"].mean()
            verdict = "WHuber较低" if huber < mse else "MSE较低" if mse < huber else "相同"
            lines.append(f"- {q.upper()} RMSE（有样本起报描述性均值）：MSE {mse:.2f} cm，WHuber {huber:.2f} cm；{verdict}。" if pd.notna(mse) and pd.notna(huber) else f"- {q.upper()}：至少一个模型缺少可评价样本。")
            ratios = group.groupby("model")[f"{q}_underestimation_ratio"].mean()
            if ratios.notna().all():
                lines.append(f"- {q.upper()}低估比例（有样本起报描述性均值）：MSE {ratios[MODELS[0]]:.1%}，WHuber {ratios[MODELS[1]]:.1%}。")
        lines.append("")
    lines += ["每场的峰时偏差和24/48/72小时累计RMSE见总表和事件报告。不同站点阈值来源可能不同，不对事件样本作独立显著性推断。", ""]
    (destination / "rollout_only_summary.md").write_text("\n".join(lines), encoding="utf-8")
    readme = ["# 滚动模型台风事件后处理", "", "本目录是只读归档预测的派生分析，原始事件目录和旧图保持原样。", "只比较 MSE Rollout-6 与 WHuber Rollout-6；每场三次起报均单独评价72小时，不拼接，也不视为三场独立事件。", "", "- `all_events_rollout_comparison.csv`：前24/48/72小时累计窗口RMSE、72小时窗口峰高/峰时误差和P95/P99统计。", "- 各事件的 `extreme_error_summary.csv`：逐起报、逐模型的全部/P95/P99样本误差与低估统计。", "- 各事件的四张PNG：过程、逐小时有符号误差、累计RMSE、极端误差。旧图仍在原归档目录。", "- `rollout_only_summary.md` 与各事件 `研究报告.md`：中文描述性分析。", "", "阈值优先使用训练期固定阈值。厦门、连云港、北海使用训练期全部有效目标的正向尾部阈值；坎门缺少归档训练阈值，使用已固定的1996验证集1小时正向阈值，并发出警告。若其他站缺少固定阈值，才使用事件正值分位数，仅作描述性比较。阈值来源在CSV中逐行记录。", "", "从 `projects/xiamen_short_term/short_term_forecast` 运行：", "", "```powershell", "python -m src.xiamen_forecast.replot_rollout_typhoon_events", "```", "", "程序拒绝覆盖已存在的目标目录；重绘时指定新的 `--output-dir`。全部结果是已知未来ERA5强迫下的追加历史回算，不是实时预报或新的盲测。", ""]
    (destination / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    args = parser.parse_args()
    run(destination=args.output_dir)


if __name__ == "__main__":
    main()
