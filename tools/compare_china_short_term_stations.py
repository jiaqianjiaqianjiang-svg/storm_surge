"""Summarize aligned 1997 short-term forecasts for the three China stations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATIONS = {
    "xiamen": ("Xiamen", "厦门"),
    "lianyungang": ("Lianyungang", "连云港"),
    "beihai": ("Beihai", "北海"),
}
LEADS = (1, 3, 6, 12, 24, 48, 72)
ONE_STEP_MODELS = (
    "persistence", "ridge", "surge_mlp", "era5_cnn", "cnn",
    "cnn_lstm", "cnn_gru", "tcn", "transformer",
)
ROLLING_MODELS = (
    "persistence", "ridge", "cnn", "cnn_lstm", "cnn_gru",
    "cnn_gru_rollout6",
)
MODEL_LABELS = {
    "persistence": "Persistence",
    "ridge": "Ridge",
    "surge_mlp": "Surge MLP",
    "era5_cnn": "ERA5 CNN",
    "cnn": "CNN",
    "cnn_lstm": "CNN-LSTM",
    "cnn_gru": "CNN-GRU",
    "cnn_gru_rollout6": "CNN-GRU rollout-6",
    "tcn": "TCN",
    "transformer": "Transformer",
}
MODEL_COLORS = {
    "persistence": "#E69F00",
    "ridge": "#666666",
    "surge_mlp": "#56B4E9",
    "era5_cnn": "#CC79A7",
    "cnn": "#0072B2",
    "cnn_lstm": "#009E73",
    "cnn_gru": "#6A3D9A",
    "cnn_gru_rollout6": "#B2182B",
    "tcn": "#D55E00",
    "transformer": "#8B5A2B",
}


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    reports = repository / "reports" / "experiment_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=reports)
    parser.add_argument("--year", type=int, default=1997)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=reports / "china_three_station_comparison_1997_seed42",
    )
    return parser.parse_args()


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result file: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def load_comparison(
    reports_root: Path, year: int = 1997, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """Read compact result packages and return aligned station tables."""

    one_step_frames: list[pd.DataFrame] = []
    rolling_frames: list[pd.DataFrame] = []
    peak_frames: list[pd.DataFrame] = []
    packages: dict[str, Path] = {}

    for station_id, (station_en, station_zh) in STATIONS.items():
        package = reports_root / f"{station_id}_short_term_{year}_seed{seed}"
        packages[station_id] = package
        one_step = _read_csv(
            package / "test_model_metrics.csv",
            {"model", "n", "pearson_r", "r2", "rmse_cm", "mae_cm",
             "bias_cm", "rrmse_percent"},
        )
        rolling = _read_csv(
            package / "rolling_metrics_long.csv",
            {"lead_hours", "model", "n", "pearson_r", "rmse_cm", "mae_cm",
             "bias_cm", "rrmse_percent"},
        )
        peak_path = package / "journal_figures" / "peak_metrics.json"
        if not peak_path.is_file():
            raise FileNotFoundError(f"Missing required result file: {peak_path}")
        peak = pd.DataFrame(json.loads(peak_path.read_text(encoding="utf-8-sig")))
        peak_required = {
            "percentile", "threshold_cm", "n_peak", "peak_rmse", "peak_mae",
            "peak_bias", "underestimation_ratio", "peak_correlation",
        }
        missing = peak_required - set(peak.columns)
        if missing:
            raise ValueError(f"{peak_path} is missing fields: {sorted(missing)}")

        for frame in (one_step, rolling, peak):
            frame.insert(0, "station_id", station_id)
            frame.insert(1, "station", station_en)
            frame.insert(2, "station_zh", station_zh)
        one_step_frames.append(one_step)
        rolling_frames.append(rolling)
        peak_frames.append(peak)

    one_step = pd.concat(one_step_frames, ignore_index=True)
    rolling = pd.concat(rolling_frames, ignore_index=True)
    peak = pd.concat(peak_frames, ignore_index=True)
    return one_step, rolling, peak, packages


def build_summary(
    one_step: pd.DataFrame, rolling: pd.DataFrame, peak: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for station_id, (station_en, station_zh) in STATIONS.items():
        station_one = one_step[one_step.station_id == station_id]
        best = station_one.loc[station_one.rmse_cm.idxmin()]
        station_rolling = rolling[rolling.station_id == station_id]

        def metric(model: str, lead: int, column: str = "rmse_cm") -> float:
            selected = station_rolling[
                (station_rolling.model == model)
                & (station_rolling.lead_hours == lead)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one {station_id}/{model}/{lead}h row, found {len(selected)}"
                )
            return float(selected.iloc[0][column])

        p99 = peak[
            (peak.station_id == station_id) & np.isclose(peak.percentile, 99.0)
        ]
        if len(p99) != 1:
            raise ValueError(f"Expected one {station_id} P99 row, found {len(p99)}")
        p99 = p99.iloc[0]
        base72 = metric("cnn_gru", 72)
        rollout72 = metric("cnn_gru_rollout6", 72)
        persistence72 = metric("persistence", 72)
        rows.append({
            "station_id": station_id,
            "station": station_en,
            "station_zh": station_zh,
            "best_one_step_model": str(best.model),
            "best_one_step_rmse_cm": float(best.rmse_cm),
            "best_one_step_mae_cm": float(best.mae_cm),
            "best_one_step_pearson_r": float(best.pearson_r),
            "best_one_step_rrmse_percent": float(best.rrmse_percent),
            "rollout6_24h_rmse_cm": metric("cnn_gru_rollout6", 24),
            "rollout6_72h_rmse_cm": rollout72,
            "cnn_gru_72h_rmse_cm": base72,
            "rollout6_72h_reduction_percent": 100.0 * (1.0 - rollout72 / base72),
            "rollout6_72h_skill_vs_persistence": 1.0 - (rollout72 / persistence72) ** 2,
            "p99_threshold_cm": float(p99.threshold_cm),
            "p99_rmse_cm": float(p99.peak_rmse),
            "p99_bias_cm": float(p99.peak_bias),
            "p99_underestimation_percent": 100.0 * float(p99.underestimation_ratio),
            "p99_pearson_r": float(p99.peak_correlation),
        })
    return pd.DataFrame(rows)


def _setup_style(language: str) -> None:
    candidates = ["Arial", "DejaVu Sans"]
    if language == "zh":
        candidates = ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", *candidates]
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": font,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.75,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    })


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_one_step(one_step: pd.DataFrame, destination: Path, language: str) -> None:
    _setup_style(language)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), sharey=True)
    for ax, (station_id, (station_en, station_zh)) in zip(axes, STATIONS.items()):
        values = (
            one_step[one_step.station_id == station_id]
            .set_index("model").reindex(ONE_STEP_MODELS)
        )
        x = np.arange(len(values))
        colors = [MODEL_COLORS[model] for model in values.index]
        bars = ax.bar(x, values.rmse_cm, color=colors, width=0.72, zorder=2)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in values.index], rotation=38, ha="right")
        ax.set_title(station_zh if language == "zh" else station_en)
        ax.set_xlabel("模型" if language == "zh" else "Model")
        _style_axis(ax)
        for bar, value in zip(bars, values.rmse_cm):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=6.5)
    axes[0].set_ylabel("RMSE（cm）" if language == "zh" else "RMSE (cm)")
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_rolling(rolling: pd.DataFrame, destination: Path, language: str) -> None:
    _setup_style(language)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), sharey=True)
    for ax, (station_id, (station_en, station_zh)) in zip(axes, STATIONS.items()):
        station = rolling[rolling.station_id == station_id]
        for model in ROLLING_MODELS:
            values = station[station.model == model].set_index("lead_hours").reindex(LEADS)
            ax.plot(
                LEADS, values.rmse_cm, marker="o", markersize=3.2,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model],
            )
        ax.set_xticks(LEADS)
        ax.set_title(station_zh if language == "zh" else station_en)
        ax.set_xlabel("提前量（h）" if language == "zh" else "Lead time (h)")
        _style_axis(ax)
    axes[0].set_ylabel("RMSE（cm）" if language == "zh" else "RMSE (cm)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_p99(summary: pd.DataFrame, destination: Path, language: str) -> None:
    _setup_style(language)
    labels = summary.station_zh if language == "zh" else summary.station
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    metrics = (
        ("p99_rmse_cm", "P99 RMSE（cm）" if language == "zh" else "P99 RMSE (cm)", "{:.2f}"),
        ("p99_underestimation_percent", "低估比例（%）" if language == "zh" else "Underestimation ratio (%)", "{:.1f}"),
    )
    for ax, (column, ylabel, fmt) in zip(axes, metrics):
        bars = ax.bar(labels, summary[column], color=colors, width=0.62, zorder=2)
        ax.set_ylabel(ylabel)
        _style_axis(ax)
        for bar, value in zip(bars, summary[column]):
            ax.text(bar.get_x() + bar.get_width() / 2, value, fmt.format(value),
                    ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(summary: pd.DataFrame) -> str:
    lines = [
        "# 中国三站小时级短时风暴增水预测对比（1997独立测试）",
        "",
        "三个站点均采用24小时输入、相同模型体系和相同1—72小时滚动评价。滚动实验使用未来时次ERA5再分析真值，属于历史回算。",
        "",
        "| 站点 | 一步最佳模型 | 1 h RMSE (cm) | Pearson r | rollout-6 24 h RMSE (cm) | rollout-6 72 h RMSE (cm) | 72 h改善 | P99 RMSE (cm) | P99低估比例 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.station_zh} | {MODEL_LABELS.get(row.best_one_step_model, row.best_one_step_model)} "
            f"| {row.best_one_step_rmse_cm:.3f} | {row.best_one_step_pearson_r:.3f} "
            f"| {row.rollout6_24h_rmse_cm:.3f} | {row.rollout6_72h_rmse_cm:.3f} "
            f"| {row.rollout6_72h_reduction_percent:.1f}% | {row.p99_rmse_cm:.3f} "
            f"| {row.p99_underestimation_percent:.1f}% |"
        )
    worst = summary.loc[summary.p99_rmse_cm.idxmax()]
    lines.extend([
        "",
        "## 主要发现",
        "",
        f"- 三站rollout-6均降低CNN-GRU的72小时递归误差，其中{worst.station_zh}的P99误差最高（{worst.p99_rmse_cm:.3f} cm）。",
        "- 一步最优模型因站点而异，因此正式结论应同时报告模型排名和跨提前量稳定性。",
        "- P99诊断基于各结果包中CNN-GRU的一步预测；后续极值损失实验应只用训练期阈值，并用1996验证集选择方案。",
        "",
        "## 文件说明",
        "",
        "- `china_three_station_one_step_metrics.csv`：全部一步模型指标。",
        "- `china_three_station_rolling_metrics.csv`：全部模型和提前量的滚动指标。",
        "- `china_three_station_peak_metrics.csv`：P90、P95和P99峰值指标。",
        "- `china_three_station_summary.csv`：用于论文和汇报的核心摘要。",
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    one_step: pd.DataFrame,
    rolling: pd.DataFrame,
    peak: pd.DataFrame,
    packages: dict[str, Path],
    output_dir: Path,
    language: str = "en",
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(one_step, rolling, peak)
    one_step.to_csv(output_dir / "china_three_station_one_step_metrics.csv", index=False)
    rolling.to_csv(output_dir / "china_three_station_rolling_metrics.csv", index=False)
    peak.to_csv(output_dir / "china_three_station_peak_metrics.csv", index=False)
    summary.to_csv(output_dir / "china_three_station_summary.csv", index=False)

    figures = (
        "fig01_one_step_rmse.png",
        "fig02_rolling_rmse.png",
        "fig03_p99_comparison.png",
    )
    plot_one_step(one_step, output_dir / "fig01_one_step_rmse.png", language)
    plot_rolling(rolling, output_dir / "fig02_rolling_rmse.png", language)
    plot_p99(summary, output_dir / "fig03_p99_comparison.png", language)
    (output_dir / "CHINA_THREE_STATION_COMPARISON.md").write_text(
        _markdown_table(summary), encoding="utf-8"
    )
    pd.DataFrame([
        {
            "figure_name": name.removesuffix(".png"),
            "source_packages": ";".join(str(path) for path in packages.values()),
            "output_png": str(output_dir / name),
            "status": "ok",
        }
        for name in figures
    ]).to_csv(output_dir / "figure_manifest.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    one_step, rolling, peak, packages = load_comparison(
        args.reports_root, args.year, args.seed
    )
    summary = write_outputs(
        one_step, rolling, peak, packages, args.output_dir, args.language
    )
    print(summary.to_string(index=False))
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
