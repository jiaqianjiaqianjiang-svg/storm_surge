"""按 rolling forecast lead time 分析误差。"""

from __future__ import annotations

import argparse
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
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 rolling forecast 随 lead time 的误差。")
    parser.add_argument("--forecast-csv", nargs="+", required=True, help="一个或多个 rolling_forecast.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def compute_metrics(group: pd.DataFrame) -> dict[str, float]:
    observed = group["observed"].to_numpy(float)
    forecast = group["forecast"].to_numpy(float)
    mask = np.isfinite(observed) & np.isfinite(forecast)
    observed = observed[mask]
    forecast = forecast[mask]
    if len(observed) == 0:
        return {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "pearson_r": np.nan, "n": 0}
    error = forecast - observed
    if len(observed) >= 2 and np.std(observed) > 0 and np.std(forecast) > 0:
        pearson_r = float(np.corrcoef(observed, forecast)[0, 1])
    else:
        pearson_r = float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "pearson_r": pearson_r,
        "n": int(len(observed)),
    }


def load_forecast_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "forecast" not in df.columns or "observed" not in df.columns:
        raise KeyError(f"{path} 缺少 forecast 或 observed 列")
    df = df.copy()
    if "lead_step" not in df.columns:
        df["lead_step"] = np.arange(1, len(df) + 1)
    df["source"] = path.parent.name
    return df


def plot_metric(df: pd.DataFrame, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["lead_step"], df[metric], marker="o")
    ax.set_xlabel("Lead time (hour)")
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frames = [load_forecast_csv(Path(item)) for item in args.forecast_csv]
    all_df = pd.concat(frames, ignore_index=True)
    rows = []
    for lead_step, group in all_df.groupby("lead_step"):
        rows.append({"lead_step": int(lead_step), **compute_metrics(group)})
    metrics_df = pd.DataFrame(rows).sort_values("lead_step")

    output_dir = args.output_dir or Path(args.forecast_csv[0]).resolve().parent / "lead_time_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "lead_time_metrics.csv"
    metrics_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_metric(metrics_df, "rmse", output_dir / "lead_time_rmse.png")
    plot_metric(metrics_df, "mae", output_dir / "lead_time_mae.png")
    plot_metric(metrics_df, "bias", output_dir / "lead_time_bias.png")

    print(f"[OUTPUT] lead_time_metrics.csv: {csv_path}")
    print(f"[OUTPUT] lead_time_rmse.png: {output_dir / 'lead_time_rmse.png'}")
    print(f"[OUTPUT] lead_time_mae.png: {output_dir / 'lead_time_mae.png'}")
    print(f"[OUTPUT] lead_time_bias.png: {output_dir / 'lead_time_bias.png'}")


if __name__ == "__main__":
    main()
