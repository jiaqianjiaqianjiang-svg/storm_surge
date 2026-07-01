"""厦门站短时滚动预报。

逻辑：
1. 使用最近 t 个 storm surge 历史值和对应 t 个大气场预测下一步。
2. 把预测值追加到 storm surge 历史序列。
3. 窗口右移，继续预测下一步。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from forecast_cnn_model import ForecastCNN
from forecast_dataset import EraDailyFieldStore, frequency_to_timedelta, load_surge_series


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except OSError:
                pass


configure_console_encoding()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="厦门站 storm surge 滚动预报。")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-source", choices=["ERA5", "ERA20C"], default=None)
    parser.add_argument("--frequency", choices=["hourly", "3hourly", "daily"], default=None)
    parser.add_argument("--start-date", type=str, required=True, help="第一个预报目标时间，例如 1985-12-01 00:00")
    parser.add_argument("--steps", type=int, default=7, help="向后滚动预报步数")
    parser.add_argument("--input-steps", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_checkpoint_arg(args: argparse.Namespace, checkpoint_args: dict[str, object], name: str, default: object) -> object:
    """命令行参数优先；未提供时使用训练 checkpoint 中保存的参数。"""

    cli_value = getattr(args, name)
    ckpt_value = checkpoint_args.get(name)
    if cli_value is not None:
        if ckpt_value is not None and cli_value != ckpt_value:
            print(f"[WARN] 命令行 --{name.replace('_', '-')}={cli_value} 与 checkpoint args 中的 {ckpt_value} 不一致，使用命令行值")
        return cli_value
    if ckpt_value is not None:
        print(f"[ROLLING] 从 checkpoint args 读取 {name}: {ckpt_value}")
        return ckpt_value
    return default


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境没有可用 CUDA")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def standardize_atmosphere_for_inference(atm: np.ndarray, scalers: dict[str, object]) -> np.ndarray:
    """使用训练时保存的每个气象通道均值/标准差标准化。"""

    mean = np.asarray(scalers["atmosphere_channel_mean"], dtype="float32").reshape(-1, 1, 1)
    std = np.asarray(scalers["atmosphere_channel_std"], dtype="float32").reshape(-1, 1, 1)
    std = np.where(std < 1e-6, 1.0, std)
    return ((atm - mean) / std).astype("float32")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.model_path, map_location=device)
    scalers = checkpoint["scalers"]
    variables = checkpoint.get("variables", config.VARIABLES)
    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}
    args.input_steps = int(resolve_checkpoint_arg(args, checkpoint_args, "input_steps", config.FORECAST_INPUT_STEPS))
    args.frequency = str(resolve_checkpoint_arg(args, checkpoint_args, "frequency", config.FORECAST_FREQUENCY))
    args.data_source = str(resolve_checkpoint_arg(args, checkpoint_args, "data_source", config.FORECAST_DATA_SOURCE))
    model = ForecastCNN(input_steps=args.input_steps, n_variables=len(variables), grid_size=config.GRID_SIZE).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    first_target = pd.Timestamp(args.start_date)
    if args.frequency == "daily":
        first_target = first_target.normalize()
    step_delta = frequency_to_timedelta(args.frequency)
    start_year = (first_target - args.input_steps * step_delta).year
    end_year = (first_target + args.steps * step_delta).year
    surge = load_surge_series(start_year, end_year, frequency=args.frequency)
    era = EraDailyFieldStore(args.data_source, start_year, end_year, list(variables), frequency=args.frequency)
    era.load()

    history_dates = list(pd.date_range(end=first_target - step_delta, periods=args.input_steps, freq=step_delta))
    known_history = surge.loc[history_dates, "storm_surge"].to_numpy(dtype="float32").tolist()
    rows: list[dict[str, object]] = []

    for step in range(args.steps):
        target_date = first_target + step * step_delta
        atm = era.build_atmosphere_window(history_dates)
        atm_std = standardize_atmosphere_for_inference(atm, scalers)
        surge_hist = np.asarray(known_history[-args.input_steps :], dtype="float32")
        surge_std = (surge_hist - scalers["surge_history_mean"]) / scalers["surge_history_std"]
        with torch.no_grad():
            pred_std = model(
                torch.from_numpy(atm_std[None, ...]).to(device),
                torch.from_numpy(surge_std[None, ...].astype("float32")).to(device),
            ).cpu().numpy()[0]
        pred = float(pred_std * scalers["target_std"] + scalers["target_mean"])
        observed = float(surge.loc[target_date, "storm_surge"]) if target_date in surge.index else np.nan
        rows.append({"datetime": target_date.strftime("%Y-%m-%d %H:%M:%S"), "forecast": pred, "observed": observed})
        print(f"[ROLLING] {target_date} forecast={pred:.6f} observed={observed}")

        known_history.append(pred)
        history_dates = history_dates[1:] + [target_date]

    run_name = f"{args.data_source}_{args.frequency}_rolling_{first_target.strftime('%Y%m%d%H')}_n{args.steps}"
    output_dir = config.FORECAST_OUTPUT_ROOT / run_name
    figure_dir = config.FORECAST_FIGURE_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = output_dir / "rolling_forecast.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    dates = pd.to_datetime(df["datetime"])
    ax.plot(dates, df["forecast"], marker="o", label="forecast")
    if df["observed"].notna().any():
        ax.plot(dates, df["observed"], marker="s", label="observed")
    ax.set_xlabel("Date")
    ax.set_ylabel("Storm surge")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig_path = figure_dir / "observed_vs_forecast.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    metadata = {"model_path": str(args.model_path), "scalers": scalers, "variables": variables}
    (output_dir / "rolling_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OUTPUT] rolling_forecast.csv: {csv_path}")
    print(f"[OUTPUT] observed vs forecast 曲线: {fig_path}")


if __name__ == "__main__":
    main()
