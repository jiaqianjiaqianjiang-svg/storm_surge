"""Create the unified 1996 three-station extreme-loss validation report."""

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

from .extreme_loss_suite import (
    FORMAL_SEED,
    LOSS_SCHEMES,
    SUITE_STATIONS,
    one_step_path,
    rollout_path,
)
from .station_config import MODULE_ROOT, get_station_config
from ..short_term_forecast.journal_figures.style import (
    apply_axis_style,
    label_panels,
    setup_journal_style,
)


LEADS = (1, 3, 6, 12, 24, 48, 72)
LOSS_COLORS = {
    "mse/mse": "#555E66",
    "wmse/mse": "#899DA5",
    "wmse/wmse": "#20566B",
    "awmse/awmse": "#A74F49",
    "whuber/whuber": "#8C8077",
}


def repository_root() -> Path:
    return next(
        parent for parent in (MODULE_ROOT, *MODULE_ROOT.parents)
        if (parent / ".git").exists()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", nargs="+", choices=SUITE_STATIONS, default=list(SUITE_STATIONS))
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _loss_thresholds(
    path: Path, loss_type: str, prefix: str,
) -> dict[str, Any]:
    if loss_type == "mse" and not path.is_file():
        details: dict[str, Any] = {}
    else:
        details = _read_json(path)
    return {
        f"{prefix}_q90_m": details.get("q90_m"),
        f"{prefix}_q99_m": details.get("q99_m"),
        f"{prefix}_threshold_sample_count": details.get("threshold_sample_count"),
        f"{prefix}_threshold_source": "training targets only",
    }


def collect_one_step(station_id: str, seed: int) -> pd.DataFrame:
    station = get_station_config(station_id)
    root = station.experiment_root / f"extreme_loss_suite_one_step_{station.validation_year}_seed{seed}"
    summary_path = root / "validation_extreme_loss_summary.csv"
    details_path = root / "validation_extreme_loss_details.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing unified one-step comparison: {summary_path}")
    summary = pd.read_csv(summary_path)
    details = _read_json(details_path)
    required = {"mse", "wmse", "awmse", "whuber"}
    if set(summary.experiment) != required:
        raise ValueError(
            f"{station_id} one-step experiments are incomplete: {sorted(set(summary.experiment))}"
        )
    if summary.common_samples.nunique() != 1:
        raise ValueError(f"{station_id} one-step models do not share the same timestamps")
    rows: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        loss_type = str(row["experiment"])
        detail = details[loss_type]
        output = {
            "station": station_id,
            "station_name": station.name,
            "validation_year": station.validation_year,
            "one_step_loss": loss_type,
            **row,
        }
        for percentile in (90, 95, 99):
            metrics = detail[f"positive_extreme_p{percentile}"]
            prefix = f"positive_p{percentile}"
            output.update(
                {
                    f"{prefix}_n": metrics.get("n"),
                    f"{prefix}_threshold_cm": metrics.get("threshold_cm"),
                    f"{prefix}_rmse_cm": metrics.get("rmse_cm"),
                    f"{prefix}_bias_cm": metrics.get("bias_cm"),
                    f"{prefix}_underestimation_ratio": metrics.get("underestimation_ratio"),
                }
            )
        output.update(
            _loss_thresholds(
                one_step_path(station, loss_type, seed) / "extreme_loss_config.json",
                loss_type,
                "one_step_training_loss",
            )
        )
        rows.append(output)
    return pd.DataFrame(rows)


def collect_rollout(station_id: str, seed: int) -> pd.DataFrame:
    station = get_station_config(station_id)
    root = station.experiment_root / f"extreme_loss_suite_rolling_{station.validation_year}_seed{seed}"
    path = root / "rolling_metrics_long.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing unified rollout comparison: {path}")
    metrics = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for scheme in LOSS_SCHEMES:
        model = "cnn_gru_rollout6" if scheme.scheme_id == "mse/mse" else scheme.rolling_label
        subset = metrics.loc[metrics.model == model].copy()
        if set(subset.lead_hours.astype(int)) != set(LEADS):
            raise ValueError(
                f"{station_id} {scheme.scheme_id} lacks one or more required leads"
            )
        one_step_thresholds = _loss_thresholds(
            one_step_path(station, scheme.one_step_loss, seed)
            / "extreme_loss_config.json",
            scheme.one_step_loss,
            "one_step_training_loss",
        )
        rollout_thresholds = _loss_thresholds(
            rollout_path(station, scheme, seed) / "extreme_loss_config.json",
            scheme.rollout_loss,
            "rollout_training_loss",
        )
        for row in subset.to_dict("records"):
            rows.append(
                {
                    "station": station_id,
                    "station_name": station.name,
                    "validation_year": station.validation_year,
                    "scheme": scheme.scheme_id,
                    "display_name": scheme.display_name,
                    "one_step_loss": scheme.one_step_loss,
                    "rollout_loss": scheme.rollout_loss,
                    **row,
                    **one_step_thresholds,
                    **rollout_thresholds,
                }
            )
    result = pd.DataFrame(rows)
    counts = result.groupby("lead_hours")["n"].nunique()
    if (counts != 1).any():
        raise ValueError(f"{station_id} rollout schemes do not share identical sample counts")
    return result


def plot_one_step(frame: pd.DataFrame, destination: Path) -> None:
    setup_journal_style(font_size=8.0)
    stations = list(frame.station.drop_duplicates())
    fig, axes = plt.subplots(2, len(stations), figsize=(11.2, 6.1), squeeze=False)
    loss_order = ["mse", "wmse", "awmse", "whuber"]
    colors = ["#555E66", "#899DA5", "#20566B", "#A74F49"]
    for column, station_id in enumerate(stations):
        subset = frame.loc[frame.station == station_id].set_index("experiment").loc[loss_order]
        for row, (metric, ylabel) in enumerate(
            (("rmse_cm", "Overall RMSE (cm)"), ("positive_p99_rmse_cm", "Positive P99 RMSE (cm)"))
        ):
            ax = axes[row, column]
            values = subset[metric].to_numpy(float)
            ax.bar(np.arange(len(loss_order)), values, color=colors, width=0.68)
            ax.set_xticks(np.arange(len(loss_order)), [item.upper() for item in loss_order], rotation=20)
            ax.set_ylabel(ylabel if column == 0 else "")
            ax.set_title(str(subset.station_name.iloc[0]))
            apply_axis_style(ax)
    label_panels(axes.flat, x=-0.13, y=1.02)
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_rollout(frame: pd.DataFrame, destination: Path) -> None:
    setup_journal_style(font_size=8.0)
    stations = list(frame.station.drop_duplicates())
    fig, axes = plt.subplots(2, len(stations), figsize=(11.2, 6.2), squeeze=False, sharex=True)
    for column, station_id in enumerate(stations):
        station_frame = frame.loc[frame.station == station_id]
        for row, (metric, ylabel) in enumerate(
            (("rmse_cm", "Overall RMSE (cm)"), ("positive_p99_rmse_cm", "Positive P99 RMSE (cm)"))
        ):
            ax = axes[row, column]
            for scheme in LOSS_SCHEMES:
                subset = station_frame.loc[station_frame.scheme == scheme.scheme_id].sort_values("lead_hours")
                ax.plot(
                    subset.lead_hours,
                    subset[metric],
                    color=LOSS_COLORS[scheme.scheme_id],
                    marker="o",
                    markersize=3,
                    linewidth=1.4,
                    label=scheme.display_name,
                )
            ax.set_xscale("log")
            ax.set_xticks(LEADS)
            ax.set_xticklabels([str(value) for value in LEADS])
            ax.minorticks_off()
            ax.set_ylabel(ylabel if column == 0 else "")
            ax.set_xlabel("Lead time (h)" if row == 1 else "")
            ax.set_title(str(station_frame.station_name.iloc[0]))
            apply_axis_style(ax)
    axes[0, -1].legend(loc="upper left", fontsize=7)
    label_panels(axes.flat, x=-0.13, y=1.02)
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def _format_value(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def write_report(one: pd.DataFrame, rolling: pd.DataFrame, destination: Path) -> None:
    lines = [
        "# 三站极值损失统一验证报告",
        "",
        "## 实验边界",
        "",
        "- 三站均使用CNN-GRU、24小时历史输入、seed 42和1996验证集。",
        "- 一步模型按完全相同的有效时间戳比较；Rollout-6按完全相同的72小时完整起报集合比较。",
        "- 所有极值损失阈值仅由各站训练期目标拟合。表中的训练P90/P99与事后评价P99是两种口径，不能混用。",
        "- 未来大气输入是ERA5再分析真值，结果属于历史滚动回算，不是实时预报。",
        "- 1997测试结果没有参与本报告的方案选择。",
        "",
        "## 24、48、72小时验证结果",
        "",
        "| 站点 | 方案 | 24h RMSE | 48h RMSE | 72h RMSE | 72h P99 RMSE | 72h P99偏差 | 72h低估比例 | 72h峰高RMSE | 72h峰时MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for station_id in rolling.station.drop_duplicates():
        for scheme in LOSS_SCHEMES:
            subset = rolling.loc[
                (rolling.station == station_id) & (rolling.scheme == scheme.scheme_id)
            ].set_index("lead_hours")
            row72 = subset.loc[72]
            lines.append(
                "| {station} | {scheme} | {r24} | {r48} | {r72} | {p99} | {bias} | {under} | {peak} | {timing} |".format(
                    station=get_station_config(station_id).name,
                    scheme=scheme.display_name,
                    r24=_format_value(subset.loc[24, "rmse_cm"]),
                    r48=_format_value(subset.loc[48, "rmse_cm"]),
                    r72=_format_value(row72.rmse_cm),
                    p99=_format_value(row72.positive_p99_rmse_cm),
                    bias=_format_value(row72.positive_p99_bias_cm),
                    under=(f"{float(row72.positive_p99_underestimation_ratio):.1%}"),
                    peak=_format_value(row72.positive_event_peak_rmse_cm),
                    timing=_format_value(row72.positive_event_timing_mae_hours, 1),
                )
            )
    lines.extend(["", "## 站点差异", ""])
    for station_id in rolling.station.drop_duplicates():
        station_rows = rolling.loc[
            (rolling.station == station_id) & (rolling.lead_hours == 72)
        ].sort_values("rmse_cm")
        overall = station_rows.iloc[0]
        p99 = station_rows.sort_values("positive_p99_rmse_cm").iloc[0]
        lines.append(
            f"- **{get_station_config(station_id).name}**：72小时整体RMSE最低为"
            f"{overall.display_name}（{overall.rmse_cm:.2f} cm）；72小时P99 RMSE最低为"
            f"{p99.display_name}（{p99.positive_p99_rmse_cm:.2f} cm）。"
        )
    lines.extend(
        [
            "",
            "不同站点的整体最优与强增水最优可能不同。本报告不强行指定跨站唯一损失；固定方案时应同时检查72小时整体RMSE、P99误差、低估比例以及独立正向峰值误差。",
            "",
            "## 输出说明",
            "",
            "- `three_station_extreme_loss_validation.csv`：一步验证的整体、P90/P95/P99和独立正向峰值指标。",
            "- `three_station_extreme_rollout_validation.csv`：五种固定方案在1/3/6/12/24/48/72小时的统一滚动指标。",
            "- 一步训练和Rollout训练的损失阈值分列并以米记录；评价阈值及误差字段以厘米记录。WMSE/MSE的Rollout阶段无加权阈值，但保留其一步WMSE阈值。",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.seed != FORMAL_SEED:
        raise ValueError("The formal suite is fixed to seed 42")
    output = args.output_dir or (
        repository_root() / "reports" / "experiment_results"
        / f"three_station_extreme_loss_validation_1996_seed{args.seed}"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result package: {output}")
    one = pd.concat(
        [collect_one_step(station, args.seed) for station in args.stations],
        ignore_index=True,
    )
    rolling = pd.concat(
        [collect_rollout(station, args.seed) for station in args.stations],
        ignore_index=True,
    )
    output.mkdir(parents=True)
    one.to_csv(output / "three_station_extreme_loss_validation.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(output / "three_station_extreme_rollout_validation.csv", index=False, encoding="utf-8-sig")
    plot_one_step(one, output / "three_station_extreme_loss_comparison.png")
    plot_rollout(rolling, output / "three_station_extreme_rollout_comparison.png")
    write_report(one, rolling, output / "three_station_extreme_loss_report.md")
    metadata = {
        "stations": list(args.stations),
        "validation_year": 1996,
        "seed": args.seed,
        "test_year_loaded": False,
        "schemes": [scheme.scheme_id for scheme in LOSS_SCHEMES],
        "training_threshold_source": "station-specific training targets only",
        "evaluation_sample_rule": "common timestamps/origins within each station",
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Three-station extreme-loss report: {output}", flush=True)


if __name__ == "__main__":
    main()
