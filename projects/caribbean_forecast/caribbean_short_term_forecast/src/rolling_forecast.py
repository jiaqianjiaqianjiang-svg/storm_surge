"""Recursive 12/24/48/72-hour forecast using future ERA5 forcing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .evaluate import calculate_metrics
    from .forecast_model import model_from_checkpoint
    from .plotting import cumulative_error, observed_vs_predicted, rolling_error
except ImportError:
    from evaluate import calculate_metrics
    from forecast_model import model_from_checkpoint
    from plotting import cumulative_error, observed_vs_predicted, rolling_error


MODULE_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STEPS = {12, 24, 48, 72}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--forecast-steps", type=int, choices=sorted(ALLOWED_STEPS), default=12)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def recursive_forecast(
    model: torch.nn.Module,
    atmosphere: np.ndarray,
    surge: np.ndarray,
    times: object,
    start_time: str | pd.Timestamp,
    forecast_steps: int,
    scalers: dict[str, dict[str, list[float]]],
    device: torch.device,
) -> pd.DataFrame:
    if forecast_steps not in ALLOWED_STEPS:
        raise ValueError(f"forecast_steps must be one of {sorted(ALLOWED_STEPS)}")
    index = pd.DatetimeIndex(pd.to_datetime(times))
    requested = pd.Timestamp(start_time)
    if requested.tzinfo is not None:
        requested = requested.tz_convert("UTC").tz_localize(None)
    positions = np.flatnonzero(index == requested)
    if not len(positions):
        raise ValueError(f"Start time {requested} not found in the prepared hourly dataset")
    origin = int(positions[0]); input_steps = model.input_steps
    if origin < input_steps or origin + forecast_steps > len(index):
        raise ValueError("Insufficient history or future ERA5 forcing for the requested forecast")
    expected = pd.date_range(index[origin - input_steps], periods=input_steps + forecast_steps, freq="1h")
    if not index[origin - input_steps : origin + forecast_steps].equals(expected):
        raise ValueError("History/forcing window is not strictly hourly; forecast stopped")
    atmosphere_mean = np.asarray(scalers["atmosphere"]["mean"], dtype=np.float32).reshape(1, -1, 1, 1)
    atmosphere_scale = np.asarray(scalers["atmosphere"]["scale"], dtype=np.float32).reshape(1, -1, 1, 1)
    surge_mean = float(scalers["surge"]["mean"][0]); surge_scale = float(scalers["surge"]["scale"][0])
    history = np.asarray(surge[origin - input_steps : origin], dtype=np.float32).tolist()
    if not np.isfinite(history).all():
        raise ValueError("Known storm-surge history contains missing values")
    rows = []; model.eval()
    with torch.no_grad():
        for lead in range(1, forecast_steps + 1):
            target = origin + lead - 1
            weather_window = np.asarray(atmosphere[target - input_steps : target], dtype=np.float32)
            if not np.isfinite(weather_window).all():
                raise ValueError(f"ERA5 forcing contains missing values at lead {lead}")
            weather_window = (weather_window - atmosphere_mean) / atmosphere_scale
            channels = weather_window.reshape(1, -1, weather_window.shape[-2], weather_window.shape[-1])
            scaled_history = (np.asarray(history[-input_steps:], dtype=np.float32) - surge_mean) / surge_scale
            prediction_scaled = model(
                torch.from_numpy(channels).to(device), torch.from_numpy(scaled_history[None]).to(device)
            ).item()
            prediction = prediction_scaled * surge_scale + surge_mean
            history.append(prediction)
            observed = float(surge[target]) if np.isfinite(surge[target]) else np.nan
            error = prediction - observed if np.isfinite(observed) else np.nan
            rows.append({"datetime": index[target], "lead_time": lead, "observed": observed, "predicted": prediction, "error": error, "absolute_error": abs(error) if np.isfinite(error) else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args(); dataset_path = args.dataset_path or MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if checkpoint.get("station_id") != args.station:
        raise ValueError(f"Checkpoint station {checkpoint.get('station_id')} does not match {args.station}")
    model = model_from_checkpoint(checkpoint).to(device)
    if dataset_path.is_dir():
        atmosphere = np.load(dataset_path / "atmosphere.npy", mmap_mode="r", allow_pickle=False)
        surge = np.load(dataset_path / "surge.npy", mmap_mode="r", allow_pickle=False)
        times = np.load(dataset_path / "time.npy", mmap_mode="r", allow_pickle=False)
        frame = recursive_forecast(model, atmosphere, surge, times, args.start_time, args.forecast_steps, checkpoint["scalers"], device)
    else:
        with np.load(dataset_path, mmap_mode="r", allow_pickle=False) as data:
            frame = recursive_forecast(model, data["atmosphere"], data["surge"], data["time"], args.start_time, args.forecast_steps, checkpoint["scalers"], device)
    output = args.output_dir or MODULE_ROOT / "outputs" / "rolling" / args.station; output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "rolling_forecast.csv", index=False)
    metrics = calculate_metrics(frame.observed, frame.predicted)
    (output / "rolling_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    observed_vs_predicted(frame.datetime, frame.observed, frame.predicted, output / "rolling_forecast.png")
    rolling_error(frame, output / "rolling_error.png"); cumulative_error(frame, output / "cumulative_error.png")
    print(json.dumps(metrics, indent=2)); print(f"Forecast saved to {output.resolve()}")


if __name__ == "__main__":
    main()
