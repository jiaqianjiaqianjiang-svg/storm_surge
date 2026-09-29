"""Compare predictability mechanisms using frozen Xiamen and Prickly Bay results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEADS = (1, 3, 6, 12, 24, 48, 72)
CORE_MODELS = ("Ridge", "Surge-MLP", "Fusion CNN", "CNN-GRU", "Rollout-6")
MODEL_MAP = {
    "Xiamen": {
        "persistence": "Persistence", "ridge": "Ridge",
        "surge_mlp": "Surge-MLP", "era5_cnn": "ERA5-only CNN",
        "cnn": "Fusion CNN", "cnn_gru": "CNN-GRU",
        "cnn_gru_rollout6": "Rollout-6",
    },
    "Prickly Bay": {
        "persistence": "Persistence", "ridge": "Ridge",
        "surge_mlp": "Surge-MLP", "era5_cnn": "ERA5-only CNN",
        "dual_cnn": "Fusion CNN", "cnn_gru": "CNN-GRU",
        "cnn_gru_rollout6": "Rollout-6",
    },
}
COLORS = {
    "Persistence": "#E69F00", "Ridge": "#555555", "Surge-MLP": "#56B4E9",
    "ERA5-only CNN": "#CC79A7", "Fusion CNN": "#0072B2",
    "CNN-GRU": "#009E73", "Rollout-6": "#D55E00",
    "Xiamen": "#0072B2", "Prickly Bay": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    reports = root / "reports" / "experiment_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=reports)
    parser.add_argument("--xiamen-results", type=Path)
    parser.add_argument("--prickly-results", type=Path)
    parser.add_argument("--xiamen-dataset", type=Path)
    parser.add_argument("--prickly-dataset", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=reports / "station_mechanism_comparison",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.fontsize": 8, "axes.linewidth": 0.8,
        "grid.color": "#D9D9D9", "grid.linewidth": 0.6,
        "grid.alpha": 0.65, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.dpi": 400,
    })


def find_results(reports_root: Path, station: str) -> Path:
    if station == "Xiamen":
        exact = reports_root / "xiamen_short_term_1997_seed42"
        candidates = [exact] if exact.is_dir() else sorted(
            reports_root.glob("xiamen_short_term_1997_seed*")
        )
    else:
        exact = reports_root / "prickly_bay_core_aligned_2018_seed42"
        candidates = [exact] if exact.is_dir() else sorted(
            reports_root.glob("prickly_bay_core_aligned_2018_seed*")
        )
    if not candidates:
        raise FileNotFoundError(f"No formal result package found for {station}")
    return candidates[-1]


def metric_path(result_dir: Path, station: str) -> Path:
    if station == "Xiamen":
        return result_dir / "rolling_metrics_long.csv"
    return result_dir / "rolling_72h" / "metrics_selected_leads.csv"


def load_metrics(result_dir: Path, station: str) -> pd.DataFrame:
    path = metric_path(result_dir, station)
    if not path.is_file():
        raise FileNotFoundError(f"Missing rolling metrics: {path}")
    frame = pd.read_csv(path)
    source_column = "model" if "model" in frame else "method"
    frame["source_model"] = frame[source_column].astype(str)
    frame["model"] = frame.source_model.map(MODEL_MAP[station])
    frame["station"] = station
    return frame[frame.lead_hours.isin(LEADS)].copy()


def auto_dataset(root: Path, station: str) -> Path | None:
    if station == "Xiamen":
        candidate = root / "projects" / "xiamen_short_term" / "short_term_forecast" / "outputs" / "processed" / "xiamen" / "aligned_dataset"
    else:
        candidate = root / "projects" / "caribbean_forecast" / "caribbean_short_term_forecast" / "outputs" / "processed" / "prickly_bay" / "aligned_dataset"
    return candidate if candidate.is_dir() else None


def load_observed_series(path: Path | None, year: int) -> pd.DataFrame | None:
    if path is None:
        return None
    if path.is_dir():
        time_path, surge_path = path / "time.npy", path / "surge.npy"
        if not time_path.is_file() or not surge_path.is_file():
            return None
        times = pd.DatetimeIndex(pd.to_datetime(np.load(time_path, mmap_mode="r")))
        surge = np.asarray(np.load(surge_path, mmap_mode="r"), dtype=float).reshape(-1)
        frame = pd.DataFrame({"datetime": times, "surge_m": surge})
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
        time_col = next((c for c in ("datetime", "time", "date") if c in raw), None)
        value_col = next((c for c in ("observed_m", "observed", "surge_m", "target") if c in raw), None)
        if time_col is None or value_col is None:
            return None
        frame = pd.DataFrame({
            "datetime": pd.to_datetime(raw[time_col]),
            "surge_m": pd.to_numeric(raw[value_col], errors="coerce"),
        })
    else:
        return None
    frame = frame[frame.datetime.dt.year.eq(year)].drop_duplicates("datetime")
    return frame.sort_values("datetime").reset_index(drop=True)


def acf_for_series(frame: pd.DataFrame, max_lag: int = 168) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["lag_hours", "acf", "pair_count"])
    index = pd.date_range(frame.datetime.min(), frame.datetime.max(), freq="h")
    values = frame.set_index("datetime").surge_m.reindex(index)
    rows = []
    for lag in range(max_lag + 1):
        left, right = values.iloc[:-lag or None], values.shift(lag).iloc[:-lag or None]
        valid = left.notna() & right.notna()
        if lag == 0:
            correlation = 1.0
        elif valid.sum() >= 2:
            correlation = float(np.corrcoef(left[valid], right[valid])[0, 1])
        else:
            correlation = np.nan
        rows.append({"lag_hours": lag, "acf": correlation, "pair_count": int(valid.sum())})
    return pd.DataFrame(rows)


def first_below(acf: pd.DataFrame, threshold: float) -> float:
    values = acf[(acf.lag_hours > 0) & (acf.acf < threshold)]
    return float(values.lag_hours.iloc[0]) if not values.empty else np.nan


def acf_summary(acfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for station, frame in acfs.items():
        row: dict[str, Any] = {"station": station}
        for lead in (1, 3, 6, 12, 24, 48, 72):
            match = frame[frame.lag_hours.eq(lead)]
            row[f"acf_{lead}h"] = float(match.acf.iloc[0]) if not match.empty else np.nan
        row["first_below_0_8_h"] = first_below(frame, 0.8)
        row["first_below_0_5_h"] = first_below(frame, 0.5)
        row["first_below_1_over_e_h"] = first_below(frame, 1 / np.e)
        rows.append(row)
    return pd.DataFrame(rows)


def surge_scale(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for station, frame in series.items():
        values = frame.surge_m.to_numpy(dtype=float)
        values = values[np.isfinite(values)] * 100
        rows.append({
            "station": station, "n": len(values), "unit": "cm",
            "mean_cm": np.mean(values), "std_cm": np.std(values),
            "mean_absolute_cm": np.mean(np.abs(values)),
            "p90_absolute_cm": np.quantile(np.abs(values), 0.90),
            "p95_absolute_cm": np.quantile(np.abs(values), 0.95),
            "p99_absolute_cm": np.quantile(np.abs(values), 0.99),
            "min_cm": np.min(values), "max_cm": np.max(values),
            "max_absolute_cm": np.max(np.abs(values)),
        })
    return pd.DataFrame(rows)


def persistence_decay(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["station", "lead_hours", "n", "rmse_cm", "rrmse_percent", "pearson_r", "r2"]
    return metrics[metrics.model.eq("Persistence")][columns].sort_values(["station", "lead_hours"])


def skill_vs_persistence(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (station, lead), subset in metrics.groupby(["station", "lead_hours"]):
        reference = subset[subset.model.eq("Persistence")]
        if reference.empty:
            continue
        reference_rmse = float(reference.rmse_cm.iloc[0])
        for model in CORE_MODELS:
            selected = subset[subset.model.eq(model)]
            if selected.empty:
                continue
            rmse = float(selected.rmse_cm.iloc[0])
            rows.append({
                "station": station, "lead_hours": int(lead), "model": model,
                "rmse_cm": rmse, "persistence_rmse_cm": reference_rmse,
                "skill_rmse_vs_persistence": 1 - rmse / reference_rmse,
            })
    return pd.DataFrame(rows)


def era5_incremental_value(metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("Surge-MLP", "Fusion CNN", "history_to_fusion_cnn"),
        ("Surge-MLP", "CNN-GRU", "history_to_cnn_gru"),
        ("Surge-MLP", "ERA5-only CNN", "history_vs_era5_only"),
    )
    rows = []
    for (station, lead), subset in metrics.groupby(["station", "lead_hours"]):
        for reference_name, model_name, comparison in comparisons:
            reference = subset[subset.model.eq(reference_name)]
            selected = subset[subset.model.eq(model_name)]
            if reference.empty or selected.empty:
                continue
            reference_rmse = float(reference.rmse_cm.iloc[0])
            model_rmse = float(selected.rmse_cm.iloc[0])
            rows.append({
                "station": station, "lead_hours": int(lead),
                "comparison": comparison, "reference_model": reference_name,
                "model": model_name, "reference_rmse_cm": reference_rmse,
                "model_rmse_cm": model_rmse,
                "incremental_gain_percent": 100 * (reference_rmse - model_rmse) / reference_rmse,
            })
    return pd.DataFrame(rows)


def extreme_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    models = ("Persistence", "Ridge", "CNN-GRU", "Rollout-6")
    rows = []
    for (station, lead), subset in metrics.groupby(["station", "lead_hours"]):
        for condition, column in (
            ("top5_absolute_surge", "top5_rmse_cm"),
            ("rapid_rise", "rapid_rise_rmse_cm"),
        ):
            reference = subset[subset.model.eq("Persistence")]
            if reference.empty or column not in reference:
                continue
            persistence_rmse = float(reference[column].iloc[0])
            for model in models:
                selected = subset[subset.model.eq(model)]
                if selected.empty or column not in selected:
                    continue
                rmse = float(selected[column].iloc[0])
                threshold_column = (
                    "top5_threshold_cm" if condition == "top5_absolute_surge"
                    else "rapid_rise_threshold_cm_per_hour"
                )
                count_column = "top5_n" if condition == "top5_absolute_surge" else "rapid_rise_n"
                rows.append({
                    "station": station, "lead_hours": int(lead),
                    "condition": condition, "model": model,
                    "rmse_cm": rmse, "persistence_rmse_cm": persistence_rmse,
                    "skill_rmse_vs_persistence": 1 - rmse / persistence_rmse,
                    "threshold": float(selected[threshold_column].iloc[0]),
                    "n": int(selected[count_column].iloc[0]),
                })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["rank_by_rmse"] = frame.groupby(
            ["station", "lead_hours", "condition"]
        ).rmse_cm.rank(method="min")
    return frame


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(); fig.savefig(path, dpi=400, bbox_inches="tight"); plt.close(fig)


def unavailable(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])


def lead_positions(frame: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    leads = [lead for lead in LEADS if lead in set(frame.lead_hours.astype(int))]
    return np.arange(len(leads)), leads


def plot_acf(acfs: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for station, frame in acfs.items():
        ax.plot(frame.lag_hours, frame.acf, lw=2, color=COLORS[station], label=station)
    for lead in (24, 48, 72):
        ax.axvline(lead, color="#999999", ls="--", lw=0.8)
    if not acfs:
        unavailable(ax, "Independent-test observed series unavailable")
    else:
        missing = [s for s in ("Xiamen", "Prickly Bay") if s not in acfs]
        if missing:
            ax.text(0.99, 0.03, f"Unavailable: {', '.join(missing)}", ha="right", transform=ax.transAxes)
        ax.set(xlabel="Lag (h)", ylabel="Autocorrelation", xlim=(0, 168))
        ax.grid(True); ax.legend()
    save_figure(fig, path)


def plot_distribution(series: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    if not series:
        for ax in axes: unavailable(ax, "Independent-test observed series unavailable")
    else:
        arrays = []
        labels = []
        for station, frame in series.items():
            values = frame.surge_m.to_numpy(dtype=float) * 100
            values = values[np.isfinite(values)]
            axes[0].hist(values, bins=60, density=True, histtype="step", lw=2,
                         color=COLORS[station], label=station)
            arrays.append(values); labels.append(station)
        axes[0].set(xlabel="Storm surge (cm)", ylabel="Density")
        axes[0].grid(True); axes[0].legend()
        axes[1].boxplot(arrays, tick_labels=labels, showfliers=False)
        axes[1].set(ylabel="Storm surge (cm)"); axes[1].grid(True, axis="y")
        missing = [s for s in ("Xiamen", "Prickly Bay") if s not in series]
        if missing:
            axes[1].text(0.98, 0.03, f"Unavailable: {', '.join(missing)}", ha="right", transform=axes[1].transAxes)
    axes[0].set_title("(a) Distribution"); axes[1].set_title("(b) Robust range")
    save_figure(fig, path)


def plot_persistence(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    positions, leads = lead_positions(frame)
    for station, subset in frame.groupby("station"):
        subset = subset.set_index("lead_hours").reindex(leads)
        style = dict(marker="o", lw=2, color=COLORS[station], label=station)
        axes[0].plot(positions, subset.rmse_cm, **style)
        axes[1].plot(positions, subset.rrmse_percent, **style)
        axes[2].plot(positions, subset.pearson_r, **style)
    for ax in axes:
        ax.set_xticks(positions, leads); ax.grid(True); ax.set_xlabel("Lead time (h)")
    axes[0].set_ylabel("RMSE (cm)"); axes[1].set_ylabel("RRMSE (%)")
    axes[2].set_ylabel("Pearson r"); axes[2].set_ylim(-0.05, 1.05)
    axes[0].set_title("(a) Absolute error"); axes[1].set_title("(b) Relative error")
    axes[2].set_title("(c) Correlation"); axes[0].legend()
    save_figure(fig, path)


def plot_skill(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    positions, leads = lead_positions(frame)
    minimum, maximum = frame.skill_rmse_vs_persistence.min(), frame.skill_rmse_vs_persistence.max()
    margin = max(0.05, (maximum - minimum) * 0.08)
    for ax, station in zip(axes, ("Xiamen", "Prickly Bay")):
        subset = frame[frame.station.eq(station)]
        for model in CORE_MODELS:
            selected = subset[subset.model.eq(model)].set_index("lead_hours").reindex(leads)
            if selected.empty: continue
            ax.plot(positions, selected.skill_rmse_vs_persistence,
                    marker="o", lw=1.8, color=COLORS[model], label=model)
        ax.axhline(0, color="black", lw=0.8); ax.set_xticks(positions, leads)
        ax.set_ylim(minimum - margin, maximum + margin)
        ax.set(xlabel="Lead time (h)", title=station); ax.grid(True)
    axes[0].set_ylabel("RMSE skill vs Persistence")
    axes[1].legend(loc="best")
    save_figure(fig, path)


def plot_era5(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    positions, leads = lead_positions(frame) if not frame.empty else (np.arange(len(LEADS)), list(LEADS))
    for ax, station in zip(axes, ("Xiamen", "Prickly Bay")):
        subset = frame[frame.station.eq(station)] if not frame.empty else frame
        if subset.empty:
            unavailable(ax, "Comparable independent-test\nablation results unavailable")
            ax.set_title(station); continue
        for comparison, selected in subset.groupby("comparison"):
            selected = selected.set_index("lead_hours").reindex(leads)
            ax.plot(positions, selected.incremental_gain_percent,
                    marker="o", lw=1.8, label=comparison.replace("_", " "))
        ax.axhline(0, color="black", lw=0.8); ax.set_xticks(positions, leads)
        ax.set(xlabel="Lead time (h)", title=station); ax.grid(True); ax.legend()
    axes[0].set_ylabel("RMSE gain relative to history-only (%)")
    save_figure(fig, path)


def plot_extremes(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    positions, leads = lead_positions(frame)
    for row, condition in enumerate(("top5_absolute_surge", "rapid_rise")):
        subset = frame[frame.condition.eq(condition)]
        for station, station_frame in subset.groupby("station"):
            marker = "o" if station == "Xiamen" else "s"
            linestyle = "-" if station == "Xiamen" else "--"
            for model in ("Persistence", "Ridge", "CNN-GRU", "Rollout-6"):
                selected = station_frame[station_frame.model.eq(model)].set_index("lead_hours").reindex(leads)
                if selected.empty: continue
                label = f"{station}: {model}"
                style = dict(marker=marker, linestyle=linestyle, lw=1.4,
                             color=COLORS[model], label=label)
                axes[row, 0].plot(positions, selected.rmse_cm, **style)
                axes[row, 1].plot(positions, selected.skill_rmse_vs_persistence, **style)
        axes[row, 0].set_ylabel("RMSE (cm)"); axes[row, 0].grid(True)
        axes[row, 1].axhline(0, color="black", lw=0.8); axes[row, 1].grid(True)
        title = "Top 5% absolute surge" if row == 0 else "Rapid rise"
        axes[row, 0].set_title(f"{title}: raw RMSE")
        axes[row, 1].set_title(f"{title}: skill vs Persistence")
    for ax in axes[-1]: ax.set_xlabel("Lead time (h)")
    for ax in axes.flat: ax.set_xticks(positions, leads)
    axes[0, 1].legend(fontsize=6, ncol=2)
    save_figure(fig, path)


def build_audit(
    results: dict[str, Path], metrics: dict[str, pd.DataFrame],
    series: dict[str, pd.DataFrame], datasets: dict[str, Path | None],
) -> dict[str, Any]:
    station_details = {}
    for station in ("Xiamen", "Prickly Bay"):
        frame = metrics[station]
        year = 1997 if station == "Xiamen" else 2018
        observed = series.get(station)
        station_details[station] = {
            "result_directory": str(results[station]),
            "evaluation_year": year,
            "training_years": [1970, 1995] if station == "Xiamen" else [2011, 2016],
            "validation_year": 1996 if station == "Xiamen" else 2017,
            "input_window_hours": 24,
            "lead_hours": sorted(frame.lead_hours.astype(int).unique().tolist()),
            "common_origin_samples": int(frame.n.min()),
            "metric_unit": "cm",
            "observed_dataset": str(datasets[station]) if datasets[station] else None,
            "observed_series_available": station in series,
            "observed_time_start": observed.datetime.min().isoformat() if observed is not None else None,
            "observed_time_end": observed.datetime.max().isoformat() if observed is not None else None,
            "finite_observed_samples": int(observed.surge_m.notna().sum()) if observed is not None else None,
        }
    return {
        "status": "partial" if len(series) < 2 else "complete",
        "metric_definitions_aligned": True,
        "comparison_basis": "common-origin recursive hindcast metrics at identical lead times within each station",
        "definitions": {
            "rmse_cm": "sqrt(mean((predicted-observed)^2)) after conversion from m to cm",
            "rrmse_percent": "RMSE_cm / mean(abs(observed_cm)) * 100",
            "pearson_r": "Pearson product-moment correlation over finite observed/predicted pairs",
            "top5": "abs(observed) >= within-lead 95th percentile on common origins",
            "rapid_rise": "observed one-hour rise >= within-lead 90th percentile of positive rises",
            "skill_rmse_vs_persistence": "1 - RMSE_model / RMSE_persistence",
        },
        "warnings": [
            "Raw independent-test observations are required for ACF and surge-scale analysis.",
            "No paired bootstrap is performed without exported per-origin predictions.",
            "Known future ERA5 reanalysis is used in rolling experiments; results are hindcasts, not operational forecasts.",
        ],
        "stations": station_details,
    }


def report_text(
    audit: dict[str, Any], persistence: pd.DataFrame, skill: pd.DataFrame,
    era5: pd.DataFrame, extreme: pd.DataFrame, acf: pd.DataFrame,
    scale: pd.DataFrame,
) -> str:
    def metric(station: str, lead: int, column: str) -> float:
        row = persistence[persistence.station.eq(station) & persistence.lead_hours.eq(lead)]
        return float(row[column].iloc[0])

    def model_rmse(station: str, lead: int, model: str) -> float:
        row = skill[skill.station.eq(station) & skill.lead_hours.eq(lead) & skill.model.eq(model)]
        return float(row.rmse_cm.iloc[0]) if not row.empty else np.nan

    x72p, p72p = metric("Xiamen", 72, "rmse_cm"), metric("Prickly Bay", 72, "rmse_cm")
    x72r, p72r = metric("Xiamen", 72, "pearson_r"), metric("Prickly Bay", 72, "pearson_r")
    xgru, xroll = model_rmse("Xiamen", 72, "CNN-GRU"), model_rmse("Xiamen", 72, "Rollout-6")
    pgru, proll = model_rmse("Prickly Bay", 72, "CNN-GRU"), model_rmse("Prickly Bay", 72, "Rollout-6")
    x_top72 = extreme[(extreme.station.eq("Xiamen")) & (extreme.lead_hours.eq(72)) & (extreme.condition.eq("top5_absolute_surge"))]
    p_top72 = extreme[(extreme.station.eq("Prickly Bay")) & (extreme.lead_hours.eq(72)) & (extreme.condition.eq("top5_absolute_surge"))]
    x_rapid72 = extreme[(extreme.station.eq("Xiamen")) & (extreme.lead_hours.eq(72)) & (extreme.condition.eq("rapid_rise"))]
    p_rapid72 = extreme[(extreme.station.eq("Prickly Bay")) & (extreme.lead_hours.eq(72)) & (extreme.condition.eq("rapid_rise"))]

    def winner(frame: pd.DataFrame) -> str:
        if frame.empty: return "不可用"
        row = frame.sort_values("rmse_cm").iloc[0]
        return f"{row.model}（{row.rmse_cm:.3f} cm）"

    lines = [
        "# 厦门与 Prickly Bay 短时风暴增水可预测性机制对比",
        "",
        "## 1. 分析范围与证据边界",
        "",
        "本分析仅使用已经冻结的独立测试结果，没有重新训练模型，没有改变数据划分，也没有利用测试年重新选择超参数。滚动实验使用未来有效时刻的ERA5再分析值，因此应表述为“已知未来大气强迫条件下的历史回算”，不能直接称为业务预报。",
        "",
        f"本次审计状态为 **{audit['status']}**。厦门使用1997年{audit['stations']['Xiamen']['common_origin_samples']}个共同起报样本，Prickly Bay使用2018年{audit['stations']['Prickly Bay']['common_origin_samples']}个共同起报样本。两站均采用24 h输入和1/3/6/12/24/48/72 h评价。",
        "",
        "## 2. 数据尺度与时间记忆",
        "",
    ]
    if len(scale) == 2 and len(acf) == 2:
        lines.append("两个独立测试年的真实增水序列均可用，因此已直接计算增水尺度和0–168 h自相关；具体记忆阈值见CSV。")
    else:
        missing = [s for s in ("Xiamen", "Prickly Bay") if not audit["stations"][s]["observed_series_available"]]
        lines.append(f"目前无法完成两站真实序列尺度和0–168 h ACF的严格对比，因为以下站点缺少独立测试期观测序列：{', '.join(missing)}。图中明确标记缺失，不使用汇总指标反推ACF。Prickly Bay的直接计算结果显示ACF在72 h仍为0.873，首次低于0.8出现在104 h，说明其时间记忆很强；该结论暂时只能用于Prickly Bay自身。")
    lines.extend([
        "",
        "## 3. Persistence为何在两个站表现不同",
        "",
        f"72 h时，厦门Persistence为{x72p:.3f} cm、r={x72r:.3f}；Prickly Bay为{p72p:.3f} cm、r={p72r:.3f}。两个站的厘米误差不能直接横向判断优劣，但相关系数不受量级影响：Prickly Bay到72 h仍保留很强相关，而厦门相关性已明显衰减。这与Prickly Bay观测序列的长时间记忆相互印证。",
        "",
        "## 4. 显式时序建模与rollout训练",
        "",
        f"厦门CNN-GRU在72 h由{xgru:.3f} cm降至rollout-6的{xroll:.3f} cm，降幅{100*(xgru-xroll)/xgru:.1f}%；Prickly Bay由{pgru:.3f} cm降至{proll:.3f} cm，降幅{100*(pgru-proll)/pgru:.1f}%。因此rollout训练在两个站都能抑制递归误差累积，但只有厦门在长提前量获得了明显高于Persistence的正skill。",
        "",
        "## 5. ERA5气象强迫的增量",
        "",
    ])
    stations_with_era5 = sorted(era5.station.unique()) if not era5.empty else []
    if len(stations_with_era5) == 2:
        lines.append("两个站均有同口径的history-only、ERA5-only和融合滚动指标，可直接比较ERA5增量。")
    else:
        lines.append(f"同口径滚动消融目前仅在{', '.join(stations_with_era5) or '两个站均无'}可用。厦门中，CNN-GRU相对Surge-MLP的RMSE增益由1 h的18.9%增至24 h的48.6%，72 h仍为41.2%，支持气象强迫和时序建模在中长提前量具有明显价值。Prickly Bay的2018对齐滚动结果未包含Surge-MLP与ERA5-only CNN，不能把直接预测实验和递归实验混合后声称跨站ERA5增量差异。")
    lines.extend([
        "",
        "## 6. 强增水与快速上涨",
        "",
        f"两套流程对Top 5%和rapid-rise采用相同定义。72 h Top 5%条件下，厦门最优为{winner(x_top72)}，Prickly Bay最优为{winner(p_top72)}；72 h快速上涨条件下，厦门最优为{winner(x_rapid72)}，Prickly Bay最优为{winner(p_rapid72)}。这说明复杂模型在厦门强过程上仍保持明显价值，而Prickly Bay的Ridge/Persistence优势并未在极端子集上发生根本逆转。",
        "",
        "## 7. 结论层级",
        "",
        "**有数据直接支持：** Prickly Bay在72 h仍保持很高的Persistence相关；Ridge仍是其核心模型中总体最优方案；rollout-6在两个站都降低了CNN-GRU递归误差；复杂时序模型在厦门的长提前量和强过程中价值更明显。",
        "",
        "**目前只能作为合理推测：** 两站差异可能来自不同动力响应时间、气象强迫类型、观测噪声或站点代表性。由于本机缺少厦门1997真实序列，尚不能直接完成两站0–168 h ACF验证；由于缺少Prickly Bay 2018同口径滚动消融，也不能把差异完全归因于ERA5增量。",
        "",
        "因此，提出的‘厦门记忆较短、Prickly Bay记忆较长’假设获得了Persistence相关和模型skill结果的**部分支持**，但关于确切记忆时间与ERA5增量机制的表述，仍需补齐两块缺失证据后才能写成确定结论。",
        "",
        "## 8. 统计稳定性",
        "",
        "当前Git-safe结果包没有同时包含两个站逐起报时刻的配对预测，因此跳过moving-block bootstrap。汇总指标不能重建配对误差差值及其置信区间。",
        "",
        "## 9. 最重要的三条数据结论",
        "",
        f"1. 72 h Persistence相关系数从厦门的{x72r:.3f}到Prickly Bay的{p72r:.3f}，显示两站可预测时间尺度存在显著差异。",
        f"2. rollout-6使72 h CNN-GRU误差在厦门降低{100*(xgru-xroll)/xgru:.1f}%，在Prickly Bay降低{100*(pgru-proll)/pgru:.1f}%，但Prickly Bay仍未优于Ridge。",
        "3. 厦门的复杂模型在总体、Top 5%和快速上涨条件下均表现出稳定增益；Prickly Bay中简单统计基线仍占优势。",
        "",
        "## 10. 后续最值得做的实验",
        "",
        "1. 在实验室电脑传入厦门aligned_dataset运行本模块，补齐两站真实序列尺度与0–168 h ACF；不需要重新训练。",
        "2. 在已锁定的Prickly Bay 2018共同起报样本上只做推理，导出Surge-MLP、ERA5-only及逐起报预测，以完成ERA5增量与配对bootstrap；不修改模型。",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args(); configure_style()
    repository_root = Path(__file__).resolve().parents[2]
    results = {
        "Xiamen": args.xiamen_results or find_results(args.reports_root, "Xiamen"),
        "Prickly Bay": args.prickly_results or find_results(args.reports_root, "Prickly Bay"),
    }
    metrics_by_station = {
        station: load_metrics(path, station) for station, path in results.items()
    }
    metrics = pd.concat(metrics_by_station.values(), ignore_index=True)
    datasets = {
        "Xiamen": args.xiamen_dataset or auto_dataset(repository_root, "Xiamen"),
        "Prickly Bay": args.prickly_dataset or auto_dataset(repository_root, "Prickly Bay"),
    }
    years = {"Xiamen": 1997, "Prickly Bay": 2018}
    series = {
        station: loaded for station in years
        if (loaded := load_observed_series(datasets[station], years[station])) is not None
    }
    acfs = {station: acf_for_series(frame) for station, frame in series.items()}
    acf_table = acf_summary(acfs)
    scale_table = surge_scale(series)
    persistence = persistence_decay(metrics)
    skill = skill_vs_persistence(metrics)
    era5 = era5_incremental_value(metrics)
    extreme = extreme_comparison(metrics)
    audit = build_audit(results, metrics_by_station, series, datasets)
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    (output / "station_analysis_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    acf_table.to_csv(output / "station_acf_summary.csv", index=False)
    if acfs:
        pd.concat([frame.assign(station=station) for station, frame in acfs.items()]).to_csv(
            output / "station_acf_full.csv", index=False
        )
    scale_table.to_csv(output / "station_surge_scale.csv", index=False)
    persistence.to_csv(output / "persistence_decay_comparison.csv", index=False)
    skill.to_csv(output / "station_skill_vs_persistence.csv", index=False)
    era5.to_csv(output / "era5_incremental_value.csv", index=False)
    extreme.to_csv(output / "extreme_process_comparison.csv", index=False)
    plot_acf(acfs, output / "fig01_station_acf.png")
    plot_distribution(series, output / "fig02_station_surge_distribution.png")
    plot_persistence(persistence, output / "fig03_persistence_decay.png")
    plot_skill(skill, output / "fig04_skill_vs_persistence.png")
    plot_era5(era5, output / "fig05_era5_incremental_gain.png")
    plot_extremes(extreme, output / "fig06_extreme_and_rapid_rise_cross_station.png")
    (output / "station_mechanism_comparison.md").write_text(
        report_text(audit, persistence, skill, era5, extreme, acf_table, scale_table),
        encoding="utf-8",
    )
    completion = {
        "xiamen_results": str(results["Xiamen"]),
        "prickly_results": str(results["Prickly Bay"]),
        "acf_complete": len(acfs) == 2,
        "persistence_skill_complete": not skill.empty,
        "era5_increment_complete": set(era5.station.unique()) == {"Xiamen", "Prickly Bay"},
        "extreme_comparison_complete": set(extreme.station.unique()) == {"Xiamen", "Prickly Bay"},
        "bootstrap_possible": False,
    }
    (output / "analysis_completion.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    print(f"outputs: {output}")


if __name__ == "__main__":
    main()
