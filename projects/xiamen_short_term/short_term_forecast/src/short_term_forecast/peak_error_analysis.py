"""分析 rolling forecast 在高 storm surge 样本上的峰值低估情况。"""

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
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析高增水样本的峰值低估。")
    parser.add_argument("--forecast-csv", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def compute_peak_metrics(df: pd.DataFrame) -> dict[str, float]:
    observed = df["observed"].to_numpy(float)
    forecast = df["forecast"].to_numpy(float)
    error = forecast - observed
    return {
        "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else float("nan"),
        "mae": float(np.mean(np.abs(error))) if len(error) else float("nan"),
        "bias": float(np.mean(error)) if len(error) else float("nan"),
        "mean_predicted_minus_observed": float(np.mean(error)) if len(error) else float("nan"),
        "underestimation_ratio": float(np.mean(error < 0)) if len(error) else float("nan"),
        "n_peak": int(len(error)),
    }


def plot_peak_scatter(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if df.empty:
        ax.text(0.5, 0.5, "No peak samples", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.scatter(df["observed"], df["forecast"], alpha=0.75)
        min_value = float(np.nanmin([df["observed"].min(), df["forecast"].min()]))
        max_value = float(np.nanmax([df["observed"].max(), df["forecast"].max()]))
        ax.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Observed storm surge")
    ax.set_ylabel("Forecast storm surge")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.forecast_csv)
    if "forecast" not in df.columns or "observed" not in df.columns:
        raise KeyError("forecast CSV 需要包含 forecast 和 observed 列")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["forecast", "observed"]).copy()
    threshold = float(np.nanpercentile(df["observed"], args.percentile))
    peak_df = df.loc[df["observed"] >= threshold].copy()
    metrics = compute_peak_metrics(peak_df)
    metrics.update({"percentile": args.percentile, "threshold": threshold, "n_total": int(len(df))})

    output_dir = args.output_dir or args.forecast_csv.resolve().parent / f"peak_error_p{int(args.percentile)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "peak_error_metrics.json"
    scatter_path = output_dir / "peak_scatter.png"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_peak_scatter(peak_df, scatter_path)

    print(f"[PEAK] percentile={args.percentile}, threshold={threshold:.6f}, n_peak={len(peak_df)}")
    print(f"[OUTPUT] peak_error_metrics.json: {metrics_path}")
    print(f"[OUTPUT] peak_scatter.png: {scatter_path}")


if __name__ == "__main__":
    main()
