"""Read-only stage-1 analysis of the compact 1997 typhoon hindcast package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.short_term_forecast.journal_figures.style import DPI, apply_axis_style, setup_journal_style

from .typhoon_events import score_forecast_window


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SOURCE = REPO_ROOT / "reports/experiment_results/typhoon_events_1997_seed42_compact"
DEFAULT_OUTPUT = REPO_ROOT / "reports/experiment_results/typhoon_events_1997_stage1_analysis"
EVENTS = {
    "beihai_1997232N17115": "Beihai / ZITA",
    "lianyungang_1997217N06168": "Lianyungang / WINNIE",
}
MAIN_MODELS = {
    "beihai_1997232N17115": ("cnn_gru", "cnn_gru_rollout6", "ridge"),
    "lianyungang_1997217N06168": (
        "cnn_gru", "cnn_gru_rollout6", "wmse_wmse", "awmse_awmse"
    ),
}
LOSS_MODELS = ("cnn_gru_rollout6", "wmse_mse", "wmse_wmse", "awmse_awmse", "whuber_whuber")
COLORS = {
    "observed": "#202428",
    "ridge": "#757C82",
    "persistence": "#A4A7A6",
    "cnn_gru": "#20566B",
    "cnn_gru_rollout6": "#A74F49",
    "wmse_mse": "#777F4D",
    "wmse_wmse": "#2C7C74",
    "awmse_awmse": "#765E87",
    "whuber_whuber": "#8E806F",
}
LABELS = {
    "observed": "Observed",
    "ridge": "Ridge",
    "persistence": "Persistence",
    "cnn_gru": "CNN-GRU",
    "cnn_gru_rollout6": "Rollout-6 (MSE/MSE)",
    "wmse_mse": "WMSE/MSE",
    "wmse_wmse": "WMSE/WMSE",
    "awmse_awmse": "AWMSE/AWMSE",
    "whuber_whuber": "WHuber/WHuber",
}
LINESTYLES = ("-", "--", "-.", ":", (0, (4, 2)))


def load_case(source: Path, event_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    case = source / event_id
    forecasts = pd.read_csv(case / "forecasts_72h.csv", encoding="utf-8-sig")
    original_windows = pd.read_csv(case / "event_metrics.csv", encoding="utf-8-sig")
    original_exact = pd.read_csv(case / "exact_lead_metrics.csv", encoding="utf-8-sig")
    config = json.loads((case / "experiment_config.json").read_text(encoding="utf-8"))
    required = {"event_id", "model", "origin_offset_hours", "origin_time_utc", "lead_hours", "valid_time_utc", "surge_cm"}
    if not required.issubset(forecasts.columns):
        raise ValueError(f"Missing forecast columns: {required - set(forecasts.columns)}")
    if forecasts.duplicated(["origin_offset_hours", "model", "lead_hours"]).any():
        raise ValueError(f"Duplicate model/origin/lead rows in {event_id}")
    if set(forecasts.event_id) != {event_id} or set(forecasts.origin_offset_hours) != {-72, -48, -24}:
        raise ValueError(f"Unexpected event or origin in {event_id}")
    if not np.isfinite(forecasts.surge_cm.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite surge in {event_id}")
    models = set(forecasts.model)
    if "observed" not in models or not set(MAIN_MODELS[event_id]).issubset(models):
        raise ValueError(f"Required forecast series missing from {event_id}")
    observations = forecasts.loc[forecasts.model == "observed"]
    if observations.groupby("valid_time_utc").surge_cm.agg(lambda values: values.max() - values.min()).max() > 1e-7:
        raise ValueError(f"Overlapping origins disagree about observed surge: {event_id}")
    for _, group in forecasts.groupby(["origin_offset_hours", "model"]):
        if group.lead_hours.tolist() != list(range(1, 73)):
            raise ValueError(f"Incomplete or unordered 72-hour forecast in {event_id}")
    return forecasts, original_windows, original_exact, config


def recompute_case(
    forecasts: pd.DataFrame, original_windows: pd.DataFrame, original_exact: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """重算指标并逐项核对原包；精确时效仅有三次相关起报。"""
    event_id = str(forecasts.event_id.iloc[0])
    rows, error_rows, coverage_rows = [], [], []
    all_observed = forecasts.loc[forecasts.model == "observed"].drop_duplicates("valid_time_utc")
    event_peak_time = all_observed.loc[all_observed.surge_cm.idxmax(), "valid_time_utc"]
    if not np.isclose(all_observed.surge_cm.max(), config["observed_peak_cm_in_120h_window"], atol=1e-5):
        raise ValueError(f"Event peak differs from archived configuration: {event_id}")
    for offset, section in forecasts.groupby("origin_offset_hours", sort=True):
        observed = section.loc[section.model == "observed"].sort_values("lead_hours")
        origin_time = pd.Timestamp(observed.origin_time_utc.iloc[0])
        coverage_rows.append({
            "event_id": event_id,
            "origin_offset_hours": offset,
            "origin_time_utc": origin_time.isoformat(),
            "window_end_utc": observed.valid_time_utc.iloc[-1],
            "event_peak_time_utc": event_peak_time,
            "contains_120h_event_peak": event_peak_time in set(observed.valid_time_utc),
            "observed_peak_in_72h_cm": float(observed.surge_cm.max()),
            "observed_peak_in_120h_cm": float(all_observed.surge_cm.max()),
        })
        for model, model_rows in section.groupby("model", sort=True):
            if model == "observed":
                continue
            prediction = model_rows.sort_values("lead_hours")
            if not np.array_equal(prediction.valid_time_utc.to_numpy(), observed.valid_time_utc.to_numpy()):
                raise ValueError(f"Model/observation times differ: {event_id} {offset} {model}")
            error = prediction.surge_cm.to_numpy() - observed.surge_cm.to_numpy()
            for lead, (value, truth, predicted) in enumerate(zip(error, observed.surge_cm, prediction.surge_cm), 1):
                error_rows.append({
                    "event_id": event_id, "origin_offset_hours": offset, "model": model,
                    "lead_hours": lead, "observed_cm": truth, "predicted_cm": predicted,
                    "error_cm": value, "absolute_error_cm": abs(value),
                })
            for horizon in (24, 48, 72):
                score = score_forecast_window(
                    observed.surge_cm.to_numpy() / 100,
                    prediction.surge_cm.to_numpy() / 100,
                    origin_time, horizon,
                )
                rows.append({"event_id": event_id, "origin_offset_hours": offset, "model": model, **score})
    windows = pd.DataFrame(rows)
    errors = pd.DataFrame(error_rows)
    exact = errors.loc[errors.lead_hours.isin([24, 48, 72])].groupby(
        ["event_id", "model", "lead_hours"], sort=True
    ).agg(
        n_origins=("error_cm", "size"),
        lead_rmse_cm=("error_cm", lambda x: np.sqrt(np.mean(np.square(x)))),
        lead_mae_cm=("absolute_error_cm", "mean"),
        lead_bias_cm=("error_cm", "mean"),
    ).reset_index()
    for keys, original, current, fields in (
        (["origin_offset_hours", "model", "horizon_hours"], original_windows, windows,
         ["rmse_cm", "mae_cm", "bias_cm", "observed_peak_cm", "predicted_peak_cm", "peak_height_error_cm", "peak_time_error_hours"]),
        (["model", "lead_hours"], original_exact, exact,
         ["lead_rmse_cm", "lead_mae_cm", "lead_bias_cm", "n_origins"]),
    ):
        merged = original.merge(current, on=keys, suffixes=("_original", "_new"), validate="one_to_one")
        if len(merged) != len(original) or len(merged) != len(current):
            raise ValueError(f"Archived metrics have missing or extra keys: {event_id}")
        for field in fields:
            if not np.allclose(merged[f"{field}_original"], merged[f"{field}_new"], atol=1e-5, rtol=0):
                raise ValueError(f"Recomputed {field} differs from archived result: {event_id}")
    return windows, errors, exact, pd.DataFrame(coverage_rows)


def lead_summary(errors: pd.DataFrame) -> pd.DataFrame:
    return errors.groupby(["event_id", "model", "lead_hours"], sort=True).agg(
        n_origins=("error_cm", "size"),
        rmse_cm=("error_cm", lambda x: np.sqrt(np.mean(np.square(x)))),
        mae_cm=("absolute_error_cm", "mean"),
    ).reset_index()


def lead_band_summary(errors: pd.DataFrame) -> pd.DataFrame:
    """固定时段汇总同一次起报的逐小时误差，不作独立事件推断。"""
    data = errors.copy()
    data["lead_band_hours"] = pd.cut(
        data.lead_hours, bins=[0, 12, 24, 48, 72],
        labels=["1-12", "13-24", "25-48", "49-72"],
    )
    return data.groupby(
        ["event_id", "origin_offset_hours", "model", "lead_band_hours"],
        observed=True, sort=True,
    ).agg(
        n_hours=("error_cm", "size"),
        rmse_cm=("error_cm", lambda x: np.sqrt(np.mean(np.square(x)))),
        mae_cm=("absolute_error_cm", "mean"),
    ).reset_index()


def high_frequency_diagnostics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """只描述逐小时变化，不将高频波动解释为残余潮汐。"""
    rows = []
    for model, by_model in forecasts.groupby("model", sort=True):
        groups = [("union", by_model.drop_duplicates("valid_time_utc").sort_values("valid_time_utc"))] if model == "observed" else [
            (str(offset), section.sort_values("lead_hours"))
            for offset, section in by_model.groupby("origin_offset_hours", sort=True)
        ]
        for origin, section in groups:
            values = section.surge_cm.to_numpy(dtype=float)
            changes = np.abs(np.diff(values))
            lag1 = (float(np.corrcoef(values[:-1], values[1:])[0, 1])
                    if np.std(values[:-1]) > 0 and np.std(values[1:]) > 0 else np.nan)
            rows.append({
                "event_id": str(section.event_id.iloc[0]), "model": model,
                "origin_offset_hours": origin, "n_hours": len(values),
                "median_abs_hourly_change_cm": float(np.median(changes)),
                "p95_abs_hourly_change_cm": float(np.percentile(changes, 95)),
                "max_abs_hourly_change_cm": float(changes.max()),
                "lag1_autocorrelation": lag1,
            })
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_process(forecasts: pd.DataFrame, coverage: pd.DataFrame, event_id: str, models: tuple[str, ...], stem: Path) -> None:
    setup_journal_style(font_size=8.5)
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.0), sharey=True, layout="constrained")
    selected = ("observed", *models)
    for panel, (ax, (offset, section)) in enumerate(zip(axes, forecasts.groupby("origin_offset_hours", sort=True))):
        for index, model in enumerate(selected):
            data = section.loc[section.model == model].sort_values("lead_hours")
            time = pd.to_datetime(data.valid_time_utc)
            ax.plot(time, data.surge_cm, color=COLORS[model],
                    linestyle="-" if model == "observed" else LINESTYLES[index - 1],
                    linewidth=1.8 if model == "observed" else 1.25, label=LABELS[model])
            if model == "observed":
                peak = data.iloc[int(data.surge_cm.to_numpy().argmax())]
                ax.scatter(pd.Timestamp(peak.valid_time_utc), peak.surge_cm, color=COLORS[model], s=20, zorder=5)
            else:
                peak = data.iloc[int(data.surge_cm.to_numpy().argmax())]
                ax.scatter(pd.Timestamp(peak.valid_time_utc), peak.surge_cm, color=COLORS[model],
                           marker="x", s=16, linewidths=0.8, zorder=4)
        row = coverage.loc[coverage.origin_offset_hours == offset].iloc[0]
        peak_note = "event peak in window" if row.contains_120h_event_peak else "event peak outside window"
        ax.text(0.01, 0.96, f"({chr(97 + panel)})  Origin {row.origin_time_utc[:16]} UTC (t0{offset:+d} h) | {peak_note}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        ax.set_ylabel("Surge (cm)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        apply_axis_style(ax)
    axes[0].legend(loc="lower right", ncol=2, fontsize=7)
    axes[-1].set_xlabel("UTC valid time")
    save_figure(fig, stem)


def plot_lead_errors(errors: pd.DataFrame, event_id: str, models: tuple[str, ...], stem: Path) -> None:
    setup_journal_style(font_size=8.5)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8), sharex=True, layout="constrained")
    flat = axes.ravel()
    for index, offset in enumerate((-72, -48, -24)):
        ax = flat[index]
        for model_index, model in enumerate(models):
            data = errors.loc[(errors.origin_offset_hours == offset) & (errors.model == model)]
            ax.plot(data.lead_hours, data.absolute_error_cm, color=COLORS[model],
                    linestyle=LINESTYLES[model_index], linewidth=1.15, label=LABELS[model])
        ax.set_title(f"({chr(97 + index)})  Origin t0{offset:+d} h; one absolute error per lead")
        ax.set_ylabel("Absolute error (cm)")
        apply_axis_style(ax)
    pooled = lead_summary(errors)
    for model_index, model in enumerate(models):
        data = pooled.loc[pooled.model == model]
        flat[3].plot(data.lead_hours, data.rmse_cm, color=COLORS[model],
                     linestyle=LINESTYLES[model_index], linewidth=1.15, label=LABELS[model])
    flat[3].set_title("(d)  Descriptive RMSE, n=3 overlapping origins")
    flat[3].set_ylabel("RMSE across origins (cm)")
    apply_axis_style(flat[3])
    for ax in flat[2:]:
        ax.set_xlabel("Lead time (h)")
    for ax in flat:
        ax.set_xlim(1, 72)
    flat[0].legend(loc="upper left", ncol=2, fontsize=7)
    save_figure(fig, stem)


def plot_window_comparison(windows: pd.DataFrame, models: tuple[str, ...], stem: Path) -> None:
    setup_journal_style(font_size=8.5)
    chosen = windows.loc[(windows.horizon_hours == 72) & windows.model.isin(models)]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), layout="constrained")
    for ax, field, label in zip(axes, ("rmse_cm", "absolute_peak_height_error_cm"),
                                ("72 h window RMSE (cm)", "72 h window |peak-height error| (cm)")):
        for index, model in enumerate(models):
            data = chosen.loc[chosen.model == model].sort_values("origin_offset_hours")
            ax.plot(data.origin_offset_hours, data[field], color=COLORS[model],
                    marker=("o", "s", "^", "D", "v")[index], linestyle=LINESTYLES[index],
                    label=LABELS[model])
        ax.set(xticks=[-72, -48, -24], xlabel="Origin relative to closest approach (h)", ylabel=label)
        apply_axis_style(ax)
    axes[0].legend(loc="best", fontsize=7)
    save_figure(fig, stem)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output_dir}")
    cases = {event_id: load_case(args.source, event_id) for event_id in EVENTS}
    results = {event_id: recompute_case(*case) for event_id, case in cases.items()}
    args.output_dir.mkdir(parents=True)
    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir()
    tables.mkdir()
    for event_id, (forecasts, _, _, _) in cases.items():
        windows, errors, _, coverage = results[event_id]
        plot_process(forecasts, coverage, event_id, MAIN_MODELS[event_id], figures / f"{event_id}_process")
        plot_lead_errors(errors, event_id, MAIN_MODELS[event_id], figures / f"{event_id}_lead_error")
        plot_window_comparison(windows, MAIN_MODELS[event_id], figures / f"{event_id}_window_comparison")
    ly = "lianyungang_1997217N06168"
    plot_process(cases[ly][0], results[ly][3], ly, LOSS_MODELS, figures / f"{ly}_loss_ablation")
    table_map = {
        "window_metrics_recomputed.csv": [value[0] for value in results.values()],
        "lead_error_by_origin.csv": [value[1] for value in results.values()],
        "exact_lead_metrics_recomputed.csv": [value[2] for value in results.values()],
        "peak_coverage.csv": [value[3] for value in results.values()],
        "lead_rmse_three_origins.csv": [lead_summary(value[1]) for value in results.values()],
        "lead_band_metrics.csv": [lead_band_summary(value[1]) for value in results.values()],
        "hourly_variability_diagnostics.csv": [high_frequency_diagnostics(case[0]) for case in cases.values()],
    }
    for name, frames in table_map.items():
        pd.concat(frames, ignore_index=True).to_csv(tables / name, index=False, encoding="utf-8-sig")
    print(f"Validated original metrics and generated stage-1 figures/tables: {args.output_dir}")


if __name__ == "__main__":
    main()
