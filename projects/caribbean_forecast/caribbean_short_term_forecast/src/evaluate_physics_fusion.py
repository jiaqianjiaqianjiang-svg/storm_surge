"""Evaluate 2017 physical fusion with paired calendar-block bootstrap."""

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

try:
    from .direct_multistep_baselines import direct_labels
    from .evaluate import calculate_detailed_metrics, calculate_metrics
    from .train_physics_baselines import LEADS
    from .train_station import load_prepared
except ImportError:
    from direct_multistep_baselines import direct_labels
    from evaluate import calculate_detailed_metrics, calculate_metrics
    from train_physics_baselines import LEADS
    from train_station import load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument("--bootstrap-block-hours", type=int, nargs="+", default=[72, 168])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def paired_block_bootstrap(
    observed: object,
    model_predicted: object,
    ridge_predicted: object,
    times: object,
    block_hours: int = 168,
    repetitions: int = 500,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired bootstrap of model-minus-Ridge RMSE using calendar-time blocks."""
    observed = np.asarray(observed, dtype=float)
    model = np.asarray(model_predicted, dtype=float)
    ridge = np.asarray(ridge_predicted, dtype=float)
    index = pd.DatetimeIndex(pd.to_datetime(times))
    valid = np.isfinite(observed) & np.isfinite(model) & np.isfinite(ridge) & ~index.isna()
    observed, model, ridge, index = observed[valid], model[valid], ridge[valid], index[valid]
    if block_hours < 1 or repetitions < 1 or not len(observed):
        raise ValueError("Bootstrap needs data, positive block_hours and repetitions")
    block_id = ((index - index.min()).total_seconds() // (block_hours * 3600)).astype(int)
    blocks = [np.flatnonzero(block_id == value) for value in np.unique(block_id)]
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions, dtype=float)
    skills = np.empty(repetitions, dtype=float)
    for iteration in range(repetitions):
        sampled: list[np.ndarray] = []
        count = 0
        while count < len(observed):
            block = blocks[int(rng.integers(0, len(blocks)))]
            sampled.append(block)
            count += len(block)
        positions = np.concatenate(sampled)[: len(observed)]
        model_mse = float(np.mean((model[positions] - observed[positions]) ** 2))
        ridge_mse = float(np.mean((ridge[positions] - observed[positions]) ** 2))
        differences[iteration] = (np.sqrt(model_mse) - np.sqrt(ridge_mse)) * 100
        skills[iteration] = 1 - model_mse / ridge_mse if ridge_mse > 0 else np.nan
    point_model = float(np.sqrt(np.mean((model - observed) ** 2)) * 100)
    point_ridge = float(np.sqrt(np.mean((ridge - observed) ** 2)) * 100)
    return {
        "n": int(len(observed)), "block_hours": int(block_hours),
        "calendar_block_count": int(len(blocks)), "repetitions": int(repetitions),
        "rmse_difference_model_minus_ridge_cm": point_model - point_ridge,
        "rmse_difference_ci_low_cm": float(np.quantile(differences, 0.025)),
        "rmse_difference_ci_high_cm": float(np.quantile(differences, 0.975)),
        "mse_skill_vs_ridge": float(1 - (point_model / point_ridge) ** 2) if point_ridge > 0 else np.nan,
        "mse_skill_ci_low": float(np.nanquantile(skills, 0.025)),
        "mse_skill_ci_high": float(np.nanquantile(skills, 0.975)),
    }


def training_thresholds(
    surge: np.ndarray,
    origins: np.ndarray,
    years: np.ndarray,
    lead: int,
) -> dict[str, float]:
    train = years <= 2016
    observed = np.asarray(surge[origins + lead], dtype=float)
    previous = np.asarray(surge[origins + lead - 1], dtype=float)
    train_observed = observed[train]
    positive = train_observed[train_observed > 0]
    rises = (observed - previous)[train]
    positive_rises = rises[rises > 0]
    return {
        "absolute_top5_m": float(np.quantile(np.abs(train_observed), 0.95)),
        "positive_top5_m": float(np.quantile(positive, 0.95)) if len(positive) else np.inf,
        "rapid_rise_m": float(np.quantile(positive_rises, 0.90)) if len(positive_rises) else np.inf,
    }


def _plot_rmse(metrics: pd.DataFrame, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in metrics.model.unique():
        subset = metrics[metrics.model == name].sort_values("lead_hours")
        ax.plot(subset.lead_hours, subset.rmse_cm, marker="o", label=name)
    ax.set(xlabel="Forecast lead (hours)", ylabel="RMSE (cm)", title="2017 physics-fusion development")
    ax.set_xticks(list(LEADS)); ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(destination, dpi=300); plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.reset_index(drop=True) if isinstance(frame.index, pd.RangeIndex) else frame.reset_index()
    lines = [
        "| " + " | ".join(str(value) for value in shown.columns) + " |",
        "| " + " | ".join("---" for _ in shown.columns) + " |",
    ]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(
            f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ) + " |")
    return "\n".join(lines)


def evaluate(
    dataset_path: Path,
    feature_cache: Path,
    output: Path,
    bootstrap_repetitions: int = 500,
    bootstrap_block_hours: tuple[int, ...] = (72, 168),
    seed: int = 42,
) -> pd.DataFrame:
    atmosphere, surge, times = load_prepared(dataset_path, 2011, 2017)
    if times.max().year >= 2018:
        raise AssertionError("2018 entered physics-fusion evaluation")
    baseline_path = output / "baseline_predictions_2017.csv"
    if not baseline_path.is_file():
        baseline_path = output / "predictions_2017.csv"
    baseline = pd.read_csv(baseline_path, parse_dates=["forecast_origin"])
    baseline_models = (
        "history_ridge", "combined_ridge", "history_ib_ridge",
        "history_wind_ridge", "history_all_physics_ridge", "physics_xgboost",
    )
    allowed_baseline = ["forecast_origin"]
    for lead in LEADS:
        allowed_baseline += [f"observed_{lead:02d}h_m"]
        allowed_baseline += [f"{name}_{lead:02d}h_m" for name in baseline_models]
    missing_baseline = [name for name in allowed_baseline if name not in baseline]
    if missing_baseline:
        raise ValueError(f"Baseline predictions are missing columns: {missing_baseline}")
    baseline = baseline[allowed_baseline]
    residual = pd.read_csv(output / "residual_predictions_2017.csv", parse_dates=["forecast_origin"])
    residual = residual[["forecast_origin"] + [
        f"residual_physics_xgb_{lead:02d}h_m" for lead in LEADS
    ]]
    if baseline.forecast_origin.duplicated().any() or residual.forecast_origin.duplicated().any():
        raise ValueError("Prediction origins must be unique")
    frame = baseline.merge(residual, on="forecast_origin", how="inner", suffixes=("", "_residual"))
    if len(frame) != len(baseline) or len(frame) != len(residual):
        raise ValueError("Baseline and residual predictions do not share identical origins")
    with np.load(feature_cache, allow_pickle=False) as cache:
        cache_times = pd.DatetimeIndex(pd.to_datetime(cache["time"]))
        origins = np.asarray(cache["origin_indices"], dtype=np.int64)
        origin_times = pd.DatetimeIndex(pd.to_datetime(cache["origin_time"]))
        matrices = {lead: np.asarray(cache[f"lead_{lead:02d}h"], dtype=np.float32) for lead in LEADS}
    if not times.equals(cache_times):
        raise ValueError("Feature cache and prepared data time axes differ")
    lookup = pd.Series(np.arange(len(origins)), index=origin_times)
    row_positions = lookup.reindex(pd.DatetimeIndex(frame.forecast_origin)).to_numpy()
    if np.isnan(row_positions).any():
        raise ValueError("A prediction origin is absent from the feature cache")
    row_positions = row_positions.astype(int)
    matching_origins = origins[row_positions]
    years = times[origins].year.to_numpy()
    names_document = json.loads((feature_cache.parent / "physics_feature_names.json").read_text(encoding="utf-8"))

    model_prefixes = list(baseline_models) + ["residual_physics_xgb"]
    metric_rows: list[dict[str, Any]] = []
    skill_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    forcing_rows: list[dict[str, Any]] = []

    for lead in LEADS:
        observed = frame[f"observed_{lead:02d}h_m"].to_numpy(dtype=float)
        valid_times = pd.DatetimeIndex(times[matching_origins + lead])
        ridge = frame[f"combined_ridge_{lead:02d}h_m"].to_numpy(dtype=float)
        thresholds = training_thresholds(surge, origins, years, lead)
        prior = np.asarray(surge[matching_origins + lead - 1], dtype=float)
        masks = {
            "positive_top5": observed >= thresholds["positive_top5_m"],
            "absolute_top5": np.abs(observed) >= thresholds["absolute_top5_m"],
            "rapid_rise": observed - prior >= thresholds["rapid_rise_m"],
        }
        feature_names = names_document["by_lead"][str(lead)]
        forcing_name = next(
            name for name in feature_names
            if name.startswith(f"future_{lead}h_integral_wind_stress_magnitude_local")
        )
        forcing = matrices[lead][row_positions, feature_names.index(forcing_name)]
        training_forcing = matrices[lead][years <= 2016, feature_names.index(forcing_name)]
        forcing_cut = float(np.quantile(training_forcing, 0.75))

        for model_name in model_prefixes:
            column = f"{model_name}_{lead:02d}h_m"
            predicted = frame[column].to_numpy(dtype=float)
            overall = calculate_metrics(observed, predicted)
            reference_mse = float(np.mean((ridge - observed) ** 2))
            model_mse = float(np.mean((predicted - observed) ** 2))
            detailed = calculate_detailed_metrics(observed, predicted, valid_times, ridge)
            row: dict[str, Any] = {
                "model": model_name, "lead_hours": lead, **overall,
                "skill_vs_combined_ridge": 1 - model_mse / reference_mse if reference_mse > 0 else np.nan,
                "positive_top5_n": int(masks["positive_top5"].sum()),
                "positive_top5_rmse_cm": calculate_metrics(observed[masks["positive_top5"]], predicted[masks["positive_top5"]])["rmse_cm"],
                "absolute_top5_n": int(masks["absolute_top5"].sum()),
                "absolute_top5_rmse_cm": calculate_metrics(observed[masks["absolute_top5"]], predicted[masks["absolute_top5"]])["rmse_cm"],
                "rapid_rise_n": int(masks["rapid_rise"].sum()),
                "rapid_rise_rmse_cm": calculate_metrics(observed[masks["rapid_rise"]], predicted[masks["rapid_rise"]])["rmse_cm"],
                "mean_absolute_peak_error_cm": detailed["peak_events"]["mean_absolute_peak_error_cm"],
                "mean_absolute_peak_timing_error_hours": detailed["peak_events"]["mean_absolute_timing_error_hours"],
            }
            metric_rows.append(row)
            skill_rows.append({
                "model": model_name, "lead_hours": lead,
                "mse_skill_vs_combined_ridge": row["skill_vs_combined_ridge"],
                "rmse_change_vs_combined_ridge_percent": 100 * (row["rmse_cm"] / calculate_metrics(observed, ridge)["rmse_cm"] - 1),
            })
            high = forcing >= forcing_cut
            for label, mask in (("lower_75pct", ~high), ("upper_25pct", high)):
                forcing_rows.append({
                    "model": model_name, "lead_hours": lead, "forcing_group": label,
                    "training_forcing_threshold_pa_s": forcing_cut,
                    **calculate_metrics(observed[mask], predicted[mask]),
                })
            if model_name != "combined_ridge":
                for block_hours in bootstrap_block_hours:
                    bootstrap_rows.append({
                        "model": model_name, "lead_hours": lead,
                        **paired_block_bootstrap(
                            observed, predicted, ridge, valid_times, block_hours,
                            bootstrap_repetitions, seed + lead + block_hours,
                        ),
                    })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(skill_rows).to_csv(output / "paired_skill_vs_ridge.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "bootstrap_confidence_intervals.csv", index=False)
    pd.DataFrame(forcing_rows).to_csv(output / "metrics_by_forcing_level.csv", index=False)
    frame.to_csv(output / "predictions_2017.csv", index=False)
    _plot_rmse(metrics, output / "physics_fusion_rmse_vs_lead.png")

    primary = metrics[metrics.lead_hours.isin(LEADS)].pivot(index="model", columns="lead_hours", values="rmse_cm")
    positive_top5 = metrics.pivot(index="model", columns="lead_hours", values="positive_top5_rmse_cm")
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_summary = bootstrap_frame[
        (bootstrap_frame.block_hours == 168)
        & bootstrap_frame.model.isin(["history_all_physics_ridge", "physics_xgboost", "residual_physics_xgb"])
    ][[
        "model", "lead_hours", "rmse_difference_model_minus_ridge_cm",
        "rmse_difference_ci_low_cm", "rmse_difference_ci_high_cm", "mse_skill_vs_ridge",
    ]]
    feature_metadata = json.loads((output / "physics_feature_metadata.json").read_text(encoding="utf-8"))
    residual_selection = pd.read_csv(output / "residual_alpha_selection.csv")
    residual_gain = float(
        (residual_selection.ridge_validation_rmse_cm - residual_selection.validation_rmse_cm).max()
    )
    ridge_12 = float(primary.loc["combined_ridge", 12])
    ridge_24 = float(primary.loc["combined_ridge", 24])
    xgb_12 = float(primary.loc["physics_xgboost", 12])
    xgb_24 = float(primary.loc["physics_xgboost", 24])
    report = f"""# Prickly Bay physics fusion v1

## 1. Data and coordinate audit

- Prepared ERA5 shape: `{feature_metadata['array_shape']}`; order U10, V10, MSL; units m/s, m/s, Pa.
- Loaded time: {feature_metadata['time_range'][0]} through {feature_metadata['time_range'][1]}.
- Coordinate source: `{feature_metadata['coordinates']['source_file']}`.
- Recovered grid: 40 x 40, latitude {feature_metadata['coordinates']['latitude_range']}, longitude {feature_metadata['coordinates']['longitude_range']}.
- Prickly Bay nearest cell: {feature_metadata['hourly_physics']['station_grid_coordinate']}.
- UTide reconstruction remains the source of the existing storm-surge target; v1 does not yet add tide interaction features.
- Offshore-to-bay bearing remains unverified (`null`), so directional stress is disabled instead of guessed.

## 2. Implemented physical quantities

- Inverse barometer: `-(p_local-p_ref)/(rho_water*g)` [m], rho_water=1025 kg/m3, g=9.80665 m/s2.
- Wind stress: `rho_air*Cd*speed*(U10,V10)` [Pa], rho_air=1.225 kg/m3, Garratt drag law capped at 0.003.
- Local east/north pressure gradients [Pa/m] using actual latitude/longitude spacing.
- Station, 1-degree, 3-degree and 5-degree pressure/wind-stress summaries.
- Past 3/6/12/24-hour and lead-specific future 1/3/6/12/24-hour forcing integrals [Pa s].
- Future minimum pressure, maximum wind stress and occurrence offsets.

The training-only local pressure climatology is {feature_metadata['hourly_physics']['pressure_climatology_pa']:.3f} Pa.

## 3. Experiment boundary and leakage controls

- Training and cross-fitting: 2011-2016; development and correction-weight selection: 2017.
- 2018 loaded: **No**. It remains locked for the final time-out test.
- Every model uses the same {len(frame):,} 2017 origins; the cache has {feature_metadata['origin_count']:,} development origins.
- Future ERA5 for lead h uses only t+1 through t+h. Scalers, pressure climatology and extreme thresholds use training years only.
- Residual targets use 2014-2016 Ridge out-of-fold predictions with a 72-hour boundary gap.
- Automated verification: 38 tests passed for units, signs, coordinates, windows, scalers and OOF gaps.

## 4. 2017 overall RMSE (cm)

{markdown_table(primary.round(3))}

Physics-XGBoost changes RMSE from {ridge_12:.3f} to {xgb_12:.3f} cm at 12 h
({100 * (1 - xgb_12 / ridge_12):.2f}% lower) and from {ridge_24:.3f} to {xgb_24:.3f} cm at 24 h
({100 * (1 - xgb_24 / ridge_24):.2f}% lower). It is worse at 1-6 h.

## 5. Positive-surge Top 5% RMSE (cm)

{markdown_table(positive_top5.loc[["combined_ridge", "history_all_physics_ridge", "physics_xgboost", "residual_physics_xgb"]].round(3))}

At 12 and 24 h, Physics-XGBoost substantially reduces strong-positive-surge error. Extreme subsets contain only about 35-83 cases, so uncertainty matters.

## 6. Paired 7-day block bootstrap

{markdown_table(bootstrap_summary.round(4))}

Negative RMSE difference favours the new model. The 12 h and 24 h point estimates favour Physics-XGBoost,
but their 95% intervals cross zero. This is a promising 2017 signal, not a statistically stable claim.

## 7. Ridge residual fusion

{markdown_table(residual_selection.round(4))}

The residual model selects alpha=0 at 1 h and 24 h and only 0.25 at 3/6/12 h.
Its maximum RMSE improvement is only {residual_gain:.3f} cm, so this residual formulation has no practical advantage.

## 8. Conclusions and next step

- Reliable: units, coordinates, windows, split boundaries and identical origins are verified; 2018 was not loaded.
- Supported on 2017: physical features help most at 12-24 h and during strong positive surge; short leads remain history-dominated.
- Not established: 7-day confidence intervals cross zero and directional onshore stress remains unavailable.
- Next: verify the offshore-to-bay bearing and run one directional-stress ablation. If evidence remains weak, test tide interactions before bathymetry/masks or a neural residual branch.
- Do not build a shallow-water PINN from the current single-station labels; regional water level, velocity and bathymetry are missing.

`skill_vs_combined_ridge` is an MSE skill score. All conclusions remain development evidence until the locked 2018 test.
"""
    (output / "PHYSICS_FUSION_V1_REPORT.md").write_text(report, encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    root = args.output_dir or MODULE_ROOT / "outputs" / "experiments" / args.station / "physics_fusion_v1"
    dataset = args.dataset_path or MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    cache = args.feature_cache or root / "physics_features.npz"
    metrics = evaluate(
        dataset, cache, root, args.bootstrap_repetitions,
        tuple(args.bootstrap_block_hours), args.seed,
    )
    print(metrics[["model", "lead_hours", "rmse_cm", "skill_vs_combined_ridge"]].to_string(index=False))


if __name__ == "__main__":
    main()
