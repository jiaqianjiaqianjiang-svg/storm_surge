"""Persistence baseline：用最近一个观测 storm surge 作为未来预测。

该基线用于判断深度学习模型是否优于简单持续性预报。
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
import numpy as np
import pandas as pd

import config
from forecast_dataset import frequency_to_timedelta, load_surge_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="厦门站小时级 persistence baseline。")
    parser.add_argument("--start-date", default="1985-11-01 00:00", help="第一个预报目标时间")
    parser.add_argument("--input-steps", type=int, default=config.FORECAST_INPUT_STEPS)
    parser.add_argument("--forecast-steps", "--steps", dest="forecast_steps", type=int, default=12)
    parser.add_argument("--frequency", choices=["hourly"], default="hourly")
    return parser.parse_args()


def compute_metrics(observed: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(observed) & np.isfinite(forecast)
    observed = observed[mask].astype(float)
    forecast = forecast[mask].astype(float)
    if len(observed) == 0:
        return {"pearson_r": float("nan"), "rmse": float("nan"), "mae": float("nan"), "bias": float("nan"), "n": 0}
    error = forecast - observed
    if len(observed) >= 2 and np.std(observed) > 0 and np.std(forecast) > 0:
        pearson_r = float(np.corrcoef(observed, forecast)[0, 1])
    else:
        pearson_r = float("nan")
    return {
        "pearson_r": pearson_r,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "n": int(len(observed)),
    }


def plot_forecast(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    dates = pd.to_datetime(df["datetime"])
    ax.plot(dates, df["observed"], marker="s", label="observed")
    ax.plot(dates, df["forecast"], marker="o", label="persistence")
    ax.set_xlabel("Time")
    ax.set_ylabel("Storm surge")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    first_target = pd.Timestamp(args.start_date)
    step_delta = frequency_to_timedelta(args.frequency)
    start_year = (first_target - args.input_steps * step_delta).year
    end_year = (first_target + args.forecast_steps * step_delta).year

    surge = load_surge_series(start_year, end_year, frequency=args.frequency)
    last_observed_time = first_target - step_delta
    if last_observed_time not in surge.index:
        raise KeyError(f"缺少 persistence 初始观测值: {last_observed_time}")
    persistence_value = float(surge.loc[last_observed_time, "storm_surge"])

    rows: list[dict[str, object]] = []
    for lead in range(1, args.forecast_steps + 1):
        target_time = first_target + (lead - 1) * step_delta
        observed = float(surge.loc[target_time, "storm_surge"]) if target_time in surge.index else np.nan
        rows.append(
            {
                "datetime": target_time.strftime("%Y-%m-%d %H:%M:%S"),
                "lead_step": lead,
                "forecast": persistence_value,
                "observed": observed,
            }
        )

    run_name = f"persistence_hourly_t{args.input_steps}_{first_target.strftime('%Y%m%d%H')}_n{args.forecast_steps}"
    output_dir = config.FORECAST_OUTPUT_ROOT / run_name
    figure_dir = config.FORECAST_FIGURE_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    pred_path = output_dir / "baseline_predictions.csv"
    metrics_path = output_dir / "baseline_metrics.json"
    fig_path = figure_dir / "observed_vs_persistence.png"
    df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    metrics = compute_metrics(df["observed"].to_numpy(float), df["forecast"].to_numpy(float))
    metrics.update({"model": "persistence", "input_steps": args.input_steps, "forecast_steps": args.forecast_steps})
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_forecast(df, fig_path)

    print(f"[BASELINE] persistence 初值: {last_observed_time} -> {persistence_value:.6f}")
    print(f"[OUTPUT] baseline_predictions.csv: {pred_path}")
    print(f"[OUTPUT] baseline_metrics.json: {metrics_path}")
    print(f"[OUTPUT] observed vs persistence 图: {fig_path}")


if __name__ == "__main__":
    main()
