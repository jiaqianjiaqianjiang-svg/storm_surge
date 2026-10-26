"""核验并汇总四场1997年历史台风的既有回算结果。"""

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

from src.short_term_forecast.journal_figures.style import apply_axis_style, setup_journal_style

from .analyze_typhoon_stage1 import COLORS, LABELS, recompute_case


REPO_ROOT = Path(__file__).resolve().parents[5]
RESULT_ROOT = REPO_ROOT / "reports/experiment_results"
DEFAULT_OUTPUT = RESULT_ROOT / "historical_typhoon_four_case_review"
CASES = (
    ("typhoon_events_1997_seed42_compact", "beihai_1997232N17115", "Beihai / ZITA"),
    ("typhoon_events_1997_seed42_compact", "lianyungang_1997217N06168", "Lianyungang / WINNIE"),
    ("typhoon_events_1997_amber_victor_seed42", "xiamen_1997232N14136", "Xiamen / AMBER"),
    ("typhoon_events_1997_amber_victor_seed42", "lianyungang_1997210N15120", "Lianyungang / VICTOR"),
)
MAIN_MODELS = ("cnn_gru", "cnn_gru_rollout6", "wmse_wmse")
BASELINES = ("persistence", "ridge")
ORIGINS = (-72, -48, -24)
HORIZONS = (24, 48, 72)
LIMITATIONS = {
    "beihai_1997232N17115": "ZITA与CASS窗口重叠；不得把重叠时段全部归因于ZITA。",
    "lianyungang_1997217N06168": "WINNIE部分模型在长提前量和主峰处明显失败，应保留失败案例。",
    "xiamen_1997232N14136": "AMBER主峰后存在CASS影响，120小时过程不能单独归因于AMBER。",
    "lianyungang_1997210N15120": "VICTOR增水有约12小时周期，不能直接判为残余潮汐或全部台风动力增水。",
}


def load_verified_case(source: Path, event_id: str) -> dict[str, object]:
    """原预测不改动；逐行核验累计与精确时效指标。"""
    case = source / event_id
    required = ["forecasts_72h.csv", "event_metrics.csv", "exact_lead_metrics.csv",
                "experiment_config.json", "研究报告.md", "figure1_best_track.png",
                "figure2_fixed_origin_forecasts.png", "figure3_event_errors.png"]
    missing = [str(case / name) for name in required if not (case / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing archived case files: {missing}")
    forecasts = pd.read_csv(case / "forecasts_72h.csv", encoding="utf-8-sig")
    windows = pd.read_csv(case / "event_metrics.csv", encoding="utf-8-sig")
    exact = pd.read_csv(case / "exact_lead_metrics.csv", encoding="utf-8-sig")
    config = json.loads((case / "experiment_config.json").read_text(encoding="utf-8"))
    if set(forecasts.event_id) != {event_id} or set(forecasts.origin_offset_hours) != set(ORIGINS):
        raise ValueError(f"Unexpected event ID or fixed origins: {event_id}")
    if forecasts.duplicated(["origin_offset_hours", "model", "lead_hours"]).any():
        raise ValueError(f"Duplicate forecast rows: {event_id}")
    for (_, _), group in forecasts.groupby(["origin_offset_hours", "model"]):
        if sorted(group.lead_hours.tolist()) != list(range(1, 73)):
            raise ValueError(f"Incomplete 72-hour forecast: {event_id}")
    observations = forecasts.loc[forecasts.model.eq("observed")]
    if observations.empty or observations.groupby("valid_time_utc").surge_cm.agg(lambda x: x.max() - x.min()).max() > 1e-6:
        raise ValueError(f"Overlapping observed values differ: {event_id}")
    if not np.isfinite(forecasts.surge_cm.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite forecast values: {event_id}")
    expected = set(MAIN_MODELS[:2]) | set(BASELINES)
    if not expected.issubset(set(forecasts.model)):
        raise ValueError(f"Required baseline or core model missing: {event_id}")
    verified_windows, _, verified_exact, coverage = recompute_case(forecasts, windows, exact, config)
    if len(verified_windows) != len(windows) or len(verified_exact) != len(exact):
        raise ValueError(f"Archived metrics differ in row count: {event_id}")
    return {"forecasts": forecasts, "windows": windows, "exact": exact,
            "coverage": coverage, "config": config, "source": case}


def compile_tables(cases: list[tuple[str, dict[str, object]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, comparison = [], []
    for label, data in cases:
        config = data["config"]
        forecasts = data["forecasts"]
        windows = data["windows"]
        exact = data["exact"]
        coverage = data["coverage"]
        event_id = str(forecasts.event_id.iloc[0])
        observed = forecasts.loc[forecasts.model.eq("observed")].drop_duplicates("valid_time_utc")
        observed_peak = observed.loc[observed.surge_cm.idxmax()]
        observed_peak_time = pd.Timestamp(observed_peak.valid_time_utc)
        t0 = pd.Timestamp(config["t0_utc"])
        summary_rows.append({
            "event_id": event_id, "station": str(windows.station.iloc[0]),
            "storm_name": str(windows.storm_name.iloc[0]), "storm_sid": str(config["storm_sid"]),
            "event_label": label, "window_start_utc": (t0 - pd.Timedelta(hours=72)).isoformat(),
            "window_end_exclusive_utc": (t0 + pd.Timedelta(hours=48)).isoformat(),
            "closest_time_utc": t0.isoformat(), "closest_distance_km": config["closest_distance_km"],
            "observed_event_peak_cm": float(observed_peak.surge_cm),
            "observed_event_peak_time_utc": observed_peak.valid_time_utc,
            "origin_count": len(coverage),
            "origins_covering_event_peak": int(coverage.contains_120h_event_peak.sum()),
            "interpretation_limit": LIMITATIONS[event_id],
        })
        for row in windows.itertuples():
            comparison.append({
                "event_id": event_id, "station": row.station, "storm_name": row.storm_name,
                "metric_scope": "window_cumulative", "origin_offset_hours": row.origin_offset_hours,
                "origin_time_utc": row.origin_time_utc, "horizon_hours": row.horizon_hours,
                "n_hours": row.n_hours, "n_origins": 1, "model": row.model,
                "rmse_cm": row.rmse_cm, "mae_cm": row.mae_cm, "bias_cm": row.bias_cm,
                "observed_window_peak_cm": row.observed_peak_cm,
                "predicted_window_peak_cm": row.predicted_peak_cm,
                "peak_height_error_cm": row.peak_height_error_cm,
                "peak_time_error_hours": row.peak_time_error_hours,
                "observed_window_peak_time_utc": row.observed_peak_time_utc,
                "predicted_window_peak_time_utc": row.predicted_peak_time_utc,
                "contains_event_peak": bool(pd.Timestamp(row.origin_time_utc) < observed_peak_time
                                            <= pd.Timestamp(row.origin_time_utc) + pd.Timedelta(hours=row.horizon_hours)),
            })
        for row in exact.itertuples():
            comparison.append({
                "event_id": event_id, "station": str(windows.station.iloc[0]),
                "storm_name": str(windows.storm_name.iloc[0]),
                "metric_scope": "exact_lead_aggregate", "origin_offset_hours": None,
                "origin_time_utc": None, "horizon_hours": row.lead_hours,
                "n_hours": row.n_origins, "n_origins": row.n_origins, "model": row.model,
                "rmse_cm": row.lead_rmse_cm, "mae_cm": row.lead_mae_cm,
                "bias_cm": row.lead_bias_cm, "observed_window_peak_cm": None,
                "predicted_window_peak_cm": None, "peak_height_error_cm": None,
                "peak_time_error_hours": None, "observed_window_peak_time_utc": None,
                "predicted_window_peak_time_utc": None, "contains_event_peak": None,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(comparison)


def _main_models(case: dict[str, object]) -> list[str]:
    present = set(case["forecasts"].model)
    return [name for name in MAIN_MODELS if name in present]


def plot_overview(cases: list[tuple[str, dict[str, object]]], path: Path) -> None:
    setup_journal_style(font_size=8.5)
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 10.0), layout="constrained")
    for ax, (label, data) in zip(axes, cases):
        part = data["forecasts"].loc[data["forecasts"].origin_offset_hours.eq(-24)]
        for model in ("observed", *_main_models(data)):
            series = part.loc[part.model.eq(model)].sort_values("lead_hours")
            ax.plot(pd.to_datetime(series.valid_time_utc), series.surge_cm,
                    color=COLORS[model], linewidth=1.6 if model == "observed" else 1.2,
                    linestyle="-" if model == "observed" else "--" if model == "cnn_gru_rollout6" else "-.",
                    label=LABELS[model])
        obs = part.loc[part.model.eq("observed")]
        peak = obs.loc[obs.surge_cm.idxmax()]
        ax.scatter(pd.Timestamp(peak.valid_time_utc), peak.surge_cm, color=COLORS["observed"], s=24, zorder=5)
        ax.set(title=f"{label} | origin t0-24 h", ylabel="Surge (cm)")
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 12)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        apply_axis_style(ax)
    axes[0].legend(ncol=4, fontsize=7, loc="best")
    axes[-1].set_xlabel("UTC valid time")
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_leads(cases: list[tuple[str, dict[str, object]]], path: Path) -> None:
    setup_journal_style(font_size=8.5)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.7), layout="constrained")
    for ax, (label, data) in zip(axes.flat, cases):
        windows = data["windows"]
        for model in (*BASELINES, *_main_models(data)):
            part = windows.loc[windows.model.eq(model)].groupby("horizon_hours").rmse_cm.mean()
            ax.plot(part.index, part.values, label=LABELS[model], color=COLORS[model],
                    marker="o", linestyle="--" if model == "cnn_gru_rollout6" else "-")
        ax.set(title=label, xlabel="Window horizon (h)", ylabel="Mean cumulative RMSE (cm)",
               xticks=HORIZONS)
        apply_axis_style(ax)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_peaks(cases: list[tuple[str, dict[str, object]]], path: Path) -> None:
    setup_journal_style(font_size=8.0)
    fig, axes = plt.subplots(4, 2, figsize=(9.2, 10.0), layout="constrained")
    markers = {-72: "o", -48: "s", -24: "^"}
    for row_index, (label, data) in enumerate(cases):
        windows = data["windows"]
        coverage = data["coverage"].set_index("origin_offset_hours")
        covered = set(coverage.index[coverage.contains_120h_event_peak])
        models = _main_models(data)
        for model_index, model in enumerate(models):
            selected = windows.loc[(windows.horizon_hours == 72) & windows.model.eq(model) &
                                   windows.origin_offset_hours.isin(covered)]
            for result in selected.itertuples():
                for column, field in enumerate(("peak_height_error_cm", "peak_time_error_hours")):
                    axes[row_index, column].scatter(model_index + result.origin_offset_hours / 720,
                                                    getattr(result, field), marker=markers[result.origin_offset_hours],
                                                    color=COLORS[model], s=35,
                                                    label=f"t0{result.origin_offset_hours:+d} h" if model_index == 0 else None)
        for column, unit in enumerate(("Peak-height error (cm)", "Peak-time error (h)")):
            ax = axes[row_index, column]
            ax.axhline(0, color="#555E66", lw=0.7)
            ax.set(title=f"{label} | {unit}", ylabel=unit, xticks=range(len(models)),
                   xticklabels=[LABELS[model] for model in models], xlim=(-0.5, len(models) - 0.5))
            apply_axis_style(ax)
        axes[row_index, 0].legend(fontsize=7, loc="best")
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(cases: list[tuple[str, dict[str, object]]], summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# 四场1997年历史台风追加回算汇总", "",
        "本汇总只读取两个既有结果包；未来气象是已知ERA5再分析，故为历史回算而非实时预报。",
        "1997年全年结果此前已查看，这四场也不是新的盲测。三次重叠起报只属于同一事件，不作为三个独立样本。", "",
        "## 指标口径", "",
        "`window_cumulative`：单次起报后前24/48/72小时的RMSE、MAE、Bias及该固定窗口内峰值误差。",
        "`exact_lead_aggregate`：三次起报在恰好24/48/72小时处的误差描述性RMSE、MAE、Bias（n=3）。",
        "两者不可混用；图2展示三次窗口累计RMSE的描述性均值，图3仅展示包含整场观测主峰的72小时起报窗口。", "",
        "## 逐事件结果", "",
    ]
    for label, data in cases:
        event_id = str(data["forecasts"].event_id.iloc[0])
        info = summary.loc[summary.event_id.eq(event_id)].iloc[0]
        means = data["windows"].loc[data["windows"].horizon_hours.eq(72)].groupby("model").rmse_cm.mean()
        summary_text = "; ".join(f"{LABELS[model]} {means[model]:.2f}" for model in (*BASELINES, *_main_models(data)))
        core = data["windows"].loc[data["windows"].horizon_hours.eq(72)]
        paired = core.pivot(index="origin_offset_hours", columns="model", values="rmse_cm")
        improved = int((paired.cnn_gru_rollout6 < paired.cnn_gru).sum())
        covered_origins = set(data["coverage"].loc[data["coverage"].contains_120h_event_peak, "origin_offset_hours"])
        peak_lines = []
        for model in _main_models(data):
            selected = core.loc[core.model.eq(model) & core.origin_offset_hours.isin(covered_origins)]
            if len(selected):
                peak_lines.append(f"{LABELS[model]}峰高偏差范围{selected.peak_height_error_cm.min():+.1f}至"
                                  f"{selected.peak_height_error_cm.max():+.1f} cm，峰时误差范围"
                                  f"{selected.peak_time_error_hours.min():+.0f}至"
                                  f"{selected.peak_time_error_hours.max():+.0f} h")
        lines.extend([
            f"### {label}", "",
            f"事件窗口：{info.window_start_utc}至{info.window_end_exclusive_utc}（右端不含）；最近距离{info.closest_distance_km:.2f} km。",
            f"120小时最大观测增水{info.observed_event_peak_cm:.2f} cm，发生于{info.observed_event_peak_time_utc}；"
            f"{info.origins_covering_event_peak}/3次起报覆盖此峰。",
            f"72小时窗口累计RMSE三起报均值（cm）：{summary_text}。",
            f"Rollout-6在{improved}/3次起报的72小时累计RMSE低于原CNN-GRU。",
            "覆盖整场观测主峰的72小时窗口：" + "；".join(peak_lines) + "。",
            f"解释限制：{info.interpretation_limit}",
            "",
        ])
    lines.extend([
        "## 综合判断", "",
        "1. 四场表现并不一致。Rollout-6的72小时窗口累计RMSE并非每个起报都低于CNN-GRU，也并非总能提高峰值幅度；必须同时报告逐起报表。",
        "2. 连云港极值损失在真实案例中可改善部分强增水峰，但所有损失组合仍在部分主峰出现低估或选错局部峰；不能凭1997单场重新选择损失函数。",
        "3. AMBER案例中Ridge可优于深度模型，不能把全年平均排序无条件移植到单事件；WINNIE长提前量仍有明显失败。",
        "4. ZITA与CASS、AMBER与CASS的窗口重叠，以及VICTOR周期波动都限制归因。四场样本太少，且起报相关，不能据此推断普遍台风预测能力。",
        "",
        "图1只展示每场同一个t0-24 h起报的72小时曲线；没有把不同起报拼成连续预报。图3峰时误差是两个窗口最大值的时间差，不等于整段过程系统性相位滞后。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite existing review: {args.output_dir}")
    cases = [(label, load_verified_case(args.result_root / package, event_id))
             for package, event_id, label in CASES]
    summary, comparison = compile_tables(cases)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".four_typhoon_", dir=args.output_dir.parent) as temporary:
        work = Path(temporary)
        summary.to_csv(work / "historical_typhoon_case_summary.csv", index=False, encoding="utf-8-sig")
        comparison.to_csv(work / "historical_typhoon_model_comparison.csv", index=False, encoding="utf-8-sig")
        plot_overview(cases, work / "fig01_four_case_overview.png")
        plot_leads(cases, work / "fig02_model_lead_comparison.png")
        plot_peaks(cases, work / "fig03_peak_error_comparison.png")
        write_report(cases, summary, work / "historical_typhoon_case_review.md")
        work.rename(args.output_dir)
    print(f"Four verified archived cases summarized: {args.output_dir}")
    return args.output_dir


if __name__ == "__main__":
    main()
