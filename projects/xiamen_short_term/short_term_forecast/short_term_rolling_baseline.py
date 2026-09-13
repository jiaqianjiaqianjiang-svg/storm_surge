"""Hourly rolling storm-surge forecast baseline for the Xiamen GESLA record.

This experiment is intentionally lightweight. It demonstrates the short-term
recursive workflow with one available year of data; it is not a replacement
for the paper's ERA20C-driven CNN reconstruction model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from utide import reconstruct, solve


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CONSTITUENTS = (
    "M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "M4", "MS4",
    "MN4", "2N2", "MU2", "NU2", "L2", "T2", "J1", "OO1", "M6", "M8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gesla-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "xiamen_GESLA" / "xiamen-376a-chn-uhslc",
    )
    parser.add_argument("--year", type=int, default=1985)
    parser.add_argument("--latitude", type=float, default=24.45)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--windows", type=int, nargs="+", default=[6, 12, 24, 48])
    parser.add_argument("--max-horizon", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def read_gesla_year(path: Path, year: int) -> pd.Series:
    prefix = f"{year}/"
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(prefix):
                continue
            tokens = line.split()
            if len(tokens) < 5:
                continue
            value = float(tokens[2])
            use_flag = int(float(tokens[4]))
            if use_flag == 0 or value <= -90:
                continue
            dates.append(pd.Timestamp(f"{tokens[0]} {tokens[1]}"))
            values.append(value)

    series = pd.Series(values, index=pd.DatetimeIndex(dates), name="sea_level").sort_index()
    if series.empty:
        raise ValueError(f"No usable GESLA observations found for {year}")
    expected = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00:00", freq="h")
    series = series[~series.index.duplicated(keep="first")].reindex(expected)
    missing = int(series.isna().sum())
    if missing:
        series = series.interpolate(method="time", limit=3).dropna()
    if len(series) < 24 * 180:
        raise ValueError(f"Only {len(series)} hourly observations remain after cleaning")
    print(f"[DATA] {year}: {len(series)} hourly values, original missing={missing}")
    return series


def separate_tide(sea_level: pd.Series, latitude: float) -> pd.DataFrame:
    time_days = np.asarray(
        (sea_level.index - sea_level.index[0]).total_seconds() / 86400.0,
        dtype=float,
    )
    epoch = sea_level.index[0].to_pydatetime()
    observed = sea_level.to_numpy(dtype=float)
    coef = solve(
        time_days,
        observed,
        lat=latitude,
        epoch=epoch,
        constit=CONSTITUENTS,
        method="ols",
        trend=False,
        conf_int="none",
        verbose=False,
    )
    tide = reconstruct(
        time_days,
        coef,
        epoch=epoch,
        constit=coef.name,
        verbose=False,
    ).h
    frame = pd.DataFrame(
        {
            "sea_level_m": observed,
            "predicted_tide_m": tide,
            "storm_surge_m": observed - tide,
        },
        index=sea_level.index,
    )
    print(
        "[TIDE] storm surge (m): "
        f"mean={frame.storm_surge_m.mean():.4f}, "
        f"std={frame.storm_surge_m.std():.4f}, "
        f"min={frame.storm_surge_m.min():.4f}, "
        f"max={frame.storm_surge_m.max():.4f}"
    )
    return frame


def make_training_windows(values: np.ndarray, window: int, train_end: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([values[i - window : i] for i in range(window, train_end)])
    y = values[window:train_end]
    return x, y


def fit_ridge(values: np.ndarray, window: int, train_end: int, alpha: float):
    x_train, y_train = make_training_windows(values, window, train_end)
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    model = Ridge(alpha=alpha)
    model.fit(x_scaler.transform(x_train), y_scaler.transform(y_train.reshape(-1, 1)).ravel())
    return model, x_scaler, y_scaler


def predict_next(model, x_scaler: StandardScaler, y_scaler: StandardScaler, history: list[float]) -> float:
    x = np.asarray(history, dtype=float).reshape(1, -1)
    pred_scaled = model.predict(x_scaler.transform(x)).reshape(-1, 1)
    return float(y_scaler.inverse_transform(pred_scaled)[0, 0])


def recursive_predictions(
    values: np.ndarray,
    train_end: int,
    window: int,
    max_horizon: int,
    model,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    last_origin = len(values) - max_horizon
    for origin in range(train_end, last_origin + 1):
        history = values[origin - window : origin].astype(float).tolist()
        persistence = history[-1]
        for lead in range(1, max_horizon + 1):
            pred = predict_next(model, x_scaler, y_scaler, history[-window:])
            target_index = origin + lead - 1
            rows.append(
                {
                    "origin_index": origin,
                    "target_index": target_index,
                    "lead_hour": lead,
                    "observed_m": float(values[target_index]),
                    "ridge_m": pred,
                    "persistence_m": persistence,
                }
            )
            history.append(pred)
    return pd.DataFrame(rows)


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    corr = float(np.corrcoef(observed, predicted)[0, 1]) if len(observed) > 1 else float("nan")
    return {
        "rmse_m": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae_m": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "corr": corr,
    }


def evaluate_by_lead(predictions: pd.DataFrame, window: int) -> pd.DataFrame:
    rows = []
    for lead, group in predictions.groupby("lead_hour"):
        observed = group.observed_m.to_numpy()
        for model_name, column in (("ridge", "ridge_m"), ("persistence", "persistence_m")):
            row = {"window_hours": window, "lead_hour": int(lead), "model": model_name, "n": len(group)}
            row.update(metrics(observed, group[column].to_numpy()))
            rows.append(row)
    return pd.DataFrame(rows)


def save_figures(metrics_frame: pd.DataFrame, predictions: pd.DataFrame, dates: pd.DatetimeIndex, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ridge_metrics = metrics_frame[metrics_frame.model == "ridge"]
    for window, group in ridge_metrics.groupby("window_hours"):
        ax.plot(group.lead_hour, group.rmse_m * 100, linewidth=2, label=f"Ridge t={window}h")

    # Persistence does not use the history window, so all t values produce the
    # same curve. Plot it once instead of drawing four identical lines.
    persistence_metrics = metrics_frame[metrics_frame.model == "persistence"]
    first_window = persistence_metrics.window_hours.min()
    persistence = persistence_metrics[persistence_metrics.window_hours == first_window]
    ax.plot(
        persistence.lead_hour,
        persistence.rmse_m * 100,
        "--",
        color="#263238",
        linewidth=2,
        label="Persistence baseline",
    )
    ax.set(xlabel="Forecast lead (hour)", ylabel="RMSE (cm)", title="Recursive forecast error by lead")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_by_lead.png", dpi=180)
    plt.close(fig)

    first_origin = int(predictions.origin_index.min())
    sample = predictions[predictions.origin_index == first_origin].copy()
    sample_dates = dates[sample.target_index.to_numpy(dtype=int)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sample_dates, sample.observed_m * 100, marker="o", label="Observed surge")
    ax.plot(sample_dates, sample.ridge_m * 100, marker="o", label="Recursive Ridge")
    ax.plot(sample_dates, sample.persistence_m * 100, linestyle="--", label="Persistence")
    ax.set(xlabel="Time", ylabel="Storm surge (cm)", title="Example 24-hour recursive forecast")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "example_24h_forecast.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0.5 <= args.train_ratio < 1:
        raise ValueError("--train-ratio must be in [0.5, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sea_level = read_gesla_year(args.gesla_file, args.year)
    tide_frame = separate_tide(sea_level, args.latitude)
    values = tide_frame.storm_surge_m.to_numpy(dtype=float)
    train_end = int(len(values) * args.train_ratio)
    dates = tide_frame.index

    all_metrics = []
    prediction_frames: dict[int, pd.DataFrame] = {}
    for window in args.windows:
        model, x_scaler, y_scaler = fit_ridge(values, window, train_end, args.alpha)
        predictions = recursive_predictions(
            values, train_end, window, args.max_horizon, model, x_scaler, y_scaler
        )
        prediction_frames[window] = predictions
        all_metrics.append(evaluate_by_lead(predictions, window))

    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    lead_24_ridge = metrics_frame[
        (metrics_frame.lead_hour == args.max_horizon) & (metrics_frame.model == "ridge")
    ]
    best_window = int(lead_24_ridge.sort_values("rmse_m").iloc[0].window_hours)
    selected = prediction_frames[best_window].copy()
    selected["origin_time"] = dates[selected.origin_index.to_numpy(dtype=int)].astype(str)
    selected["target_time"] = dates[selected.target_index.to_numpy(dtype=int)].astype(str)

    tide_frame.to_csv(args.output_dir / "xiamen_1985_hourly_surge.csv", encoding="utf-8-sig")
    metrics_frame.to_csv(args.output_dir / "metrics_by_lead.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(args.output_dir / "selected_window_predictions.csv", index=False, encoding="utf-8-sig")
    save_figures(metrics_frame, selected, dates, args.output_dir)

    summary = {
        "year": args.year,
        "n_hourly": len(values),
        "train_end_time": str(dates[train_end - 1]),
        "validation_start_time": str(dates[train_end]),
        "train_ratio": args.train_ratio,
        "windows_hours": args.windows,
        "max_horizon_hours": args.max_horizon,
        "best_window_by_24h_rmse": best_window,
        "surge_mean_m": float(np.mean(values)),
        "surge_std_m": float(np.std(values)),
        "surge_min_m": float(np.min(values)),
        "surge_max_m": float(np.max(values)),
    }
    for lead in (1, 3, 6, 12, 24):
        for model_name in ("ridge", "persistence"):
            row = metrics_frame[
                (metrics_frame.window_hours == best_window)
                & (metrics_frame.lead_hour == lead)
                & (metrics_frame.model == model_name)
            ].iloc[0]
            summary[f"{model_name}_{lead}h_rmse_cm"] = float(row.rmse_m * 100)
            summary[f"{model_name}_{lead}h_mae_cm"] = float(row.mae_m * 100)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[SAVE] {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
