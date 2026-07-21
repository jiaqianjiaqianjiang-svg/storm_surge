"""汇总 persistence 和多种深度学习模型的指标对比。"""

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


KNOWN_MODELS = ("persistence", "cnn", "cnn_lstm", "cnn_gru", "tcn", "transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比不同短时预报模型。")
    parser.add_argument("--root-dir", type=Path, required=True, help="例如 outputs/short_term_forecast/xiamen")
    return parser.parse_args()


def infer_model_name(run_name: str, metrics: dict[str, object]) -> str:
    if "model" in metrics:
        return str(metrics["model"])
    if "model_name" in metrics:
        return str(metrics["model_name"])
    for name in KNOWN_MODELS:
        if run_name.endswith(f"_{name}") or f"_{name}_" in run_name or run_name.startswith(name):
            return name
    return "unknown"


def collect_metrics(root_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(root_dir.rglob("*.json")):
        if path.name not in {"metrics.json", "baseline_metrics.json"}:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        run_name = path.parent.name
        rows.append({"run": run_name, "model": infer_model_name(run_name, metrics), **metrics})
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    if df.empty or metric not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["model"].astype(str) + "\n" + df["run"].astype(str), df[metric])
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir
    root_dir.mkdir(parents=True, exist_ok=True)
    df = collect_metrics(root_dir)
    csv_path = root_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_bar(df, "rmse", root_dir / "model_rmse_comparison.png")
    plot_bar(df, "mae", root_dir / "model_mae_comparison.png")
    corr_col = "pearson_r" if "pearson_r" in df.columns else "correlation"
    plot_bar(df, corr_col, root_dir / "model_correlation_comparison.png")
    print(f"[OUTPUT] model_comparison.csv: {csv_path}")
    print(f"[OUTPUT] model_rmse_comparison.png: {root_dir / 'model_rmse_comparison.png'}")
    print(f"[OUTPUT] model_mae_comparison.png: {root_dir / 'model_mae_comparison.png'}")
    print(f"[OUTPUT] model_correlation_comparison.png: {root_dir / 'model_correlation_comparison.png'}")


if __name__ == "__main__":
    main()
