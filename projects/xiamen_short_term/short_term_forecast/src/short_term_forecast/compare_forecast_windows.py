"""对比不同历史窗口长度的短时预报结果。

示例：
python src/short_term_forecast/compare_forecast_windows.py --runs ERA5_1985_1985_hourly_t12_h1 ERA5_1985_1985_hourly_t24_h1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "outputs" / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比不同 input_steps 的短时预报结果。")
    parser.add_argument("--runs", nargs="+", required=True, help="outputs/forecast_xiamen 下的 run 目录名")
    parser.add_argument("--output-name", default="window_comparison", help="输出图名前缀")
    return parser.parse_args()


def load_run(run_name: str) -> tuple[dict[str, float], pd.DataFrame]:
    run_dir = config.FORECAST_OUTPUT_ROOT / run_name
    metrics_path = run_dir / "metrics.json"
    pred_path = run_dir / "val_predictions.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"缺少 metrics.json: {metrics_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"缺少 val_predictions.csv: {pred_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    df = pd.read_csv(pred_path)
    time_col = "datetime" if "datetime" in df.columns else "date"
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.rename(columns={time_col: "datetime"})
    return metrics, df


def plot_metric_bars(rows: list[dict[str, object]], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    metrics = ["pearson_r", "rmse", "mae", "rrmse"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes_flat = axes.ravel()
    for ax, metric in zip(axes_flat, metrics):
        ax.bar(df["run"], df[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_prediction_overlay(run_frames: list[tuple[str, pd.DataFrame]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    first_name, first_df = run_frames[0]
    ax.plot(first_df["datetime"], first_df["observed"], label="observed", color="black", linewidth=1.5)
    for run_name, df in run_frames:
        ax.plot(df["datetime"], df["predicted"], label=f"predicted {run_name}", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Time")
    ax.set_ylabel("Storm surge")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figure_dir = config.FORECAST_FIGURE_ROOT / args.output_name
    figure_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    run_frames: list[tuple[str, pd.DataFrame]] = []
    for run_name in args.runs:
        metrics, df = load_run(run_name)
        metric_rows.append({"run": run_name, **metrics})
        run_frames.append((run_name, df))

    metrics_csv = figure_dir / f"{args.output_name}_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    metric_fig = figure_dir / f"{args.output_name}_metrics.png"
    pred_fig = figure_dir / f"{args.output_name}_predictions.png"
    plot_metric_bars(metric_rows, metric_fig)
    plot_prediction_overlay(run_frames, pred_fig)

    print(f"[OUTPUT] 指标对比 CSV: {metrics_csv}")
    print(f"[OUTPUT] 指标柱状图: {metric_fig}")
    print(f"[OUTPUT] 验证集预测对比图: {pred_fig}")


if __name__ == "__main__":
    main()
