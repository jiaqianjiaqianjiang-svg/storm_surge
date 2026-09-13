"""Train 2011-2016 physical-feature baselines and evaluate only on 2017."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from .direct_multistep_baselines import ALPHAS, build_feature_matrix, direct_labels
    from .evaluate import calculate_metrics
    from .evaluate_baselines import summarise_atmosphere
    from .train_station import load_prepared
except ImportError:
    from direct_multistep_baselines import ALPHAS, build_feature_matrix, direct_labels
    from evaluate import calculate_metrics
    from evaluate_baselines import summarise_atmosphere
    from train_station import load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[1]
LEADS = (1, 3, 6, 12, 24)
RIDGE_MODELS = (
    "history_ridge", "combined_ridge", "history_ib_ridge",
    "history_wind_ridge", "history_all_physics_ridge",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def fit_tuned_ridge(
    features: np.ndarray,
    target: np.ndarray,
    years: np.ndarray,
    alphas: tuple[float, ...] = ALPHAS,
) -> tuple[StandardScaler, Ridge, float, float]:
    """Select alpha on 2016, refit scaler and Ridge on all 2011-2016."""
    early = years <= 2015
    tuning = years == 2016
    training = years <= 2016
    if not early.any() or not tuning.any():
        raise ValueError("Ridge tuning needs both pre-2016 and 2016 samples")
    selection_scaler = StandardScaler().fit(features[early])
    x_early = selection_scaler.transform(features[early])
    x_tuning = selection_scaler.transform(features[tuning])
    scores = []
    for alpha in alphas:
        candidate = Ridge(alpha=alpha, solver="lsqr").fit(x_early, target[early])
        scores.append(float(np.sqrt(np.mean((candidate.predict(x_tuning) - target[tuning]) ** 2))))
    best_index = int(np.argmin(scores))
    best_alpha = float(alphas[best_index])
    scaler = StandardScaler().fit(features[training])
    model = Ridge(alpha=best_alpha, solver="lsqr").fit(scaler.transform(features[training]), target[training])
    return scaler, model, best_alpha, scores[best_index]


def physics_column_groups(names: list[str]) -> dict[str, np.ndarray]:
    lower = np.asarray([name.lower() for name in names])
    pressure = np.asarray([
        ("pressure" in name or "barometer" in name) and "wind_stress" not in name
        for name in lower
    ])
    wind = np.asarray(["wind" in name or "drag_coefficient" in name for name in lower])
    return {"ib": pressure, "wind": wind, "all": np.ones(len(names), dtype=bool)}


def _xgb(seed: int, device: str, smoke_test: bool):
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=30 if smoke_test else 300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=10.0,
        objective="reg:squarederror",
        tree_method="hist",
        device=device,
        n_jobs=1,
        random_state=seed,
    )


def train_baselines(
    dataset_path: Path,
    feature_cache: Path,
    output: Path,
    seed: int = 42,
    xgb_device: str = "cuda",
    smoke_test: bool = False,
) -> pd.DataFrame:
    atmosphere, surge, times = load_prepared(dataset_path, 2011, 2017)
    if times.max().year >= 2018:
        raise AssertionError("2018 entered physics baseline development")
    with np.load(feature_cache, allow_pickle=False) as cache:
        origins = np.asarray(cache["origin_indices"], dtype=np.int64)
        cache_times = pd.DatetimeIndex(pd.to_datetime(cache["time"]))
        matrices = {lead: np.asarray(cache[f"lead_{lead:02d}h"], dtype=np.float32) for lead in LEADS}
    if not times.equals(cache_times):
        raise ValueError("Physics cache time axis does not match prepared dataset")
    years = times[origins].year.to_numpy()
    development = years <= 2017
    origins = origins[development]
    years = years[development]
    matrices = {lead: matrix[development] for lead, matrix in matrices.items()}
    if smoke_test:
        keep = np.concatenate([
            np.flatnonzero(years <= 2015)[-300:], np.flatnonzero(years == 2016)[-150:],
            np.flatnonzero(years == 2017)[:150],
        ])
        origins, years = origins[keep], years[keep]
        matrices = {lead: matrix[keep] for lead, matrix in matrices.items()}
    labels_24 = direct_labels(surge, origins, 24)
    summaries = summarise_atmosphere(atmosphere)
    history = build_feature_matrix(summaries, surge, origins, True, "none")
    combined = build_feature_matrix(summaries, surge, origins, True, "past_future")
    names_document = json.loads((feature_cache.parent / "physics_feature_names.json").read_text(encoding="utf-8"))

    validation = years == 2017
    training = years <= 2016
    if not validation.any() or not training.any():
        raise ValueError("Expected both 2011-2016 training and 2017 validation origins")
    prediction_columns: dict[str, Any] = {"forecast_origin": times[origins[validation]]}
    metrics: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    saved_models: dict[str, Any] = {}

    for lead in LEADS:
        target = labels_24[:, lead - 1]
        physics = matrices[lead]
        names = names_document["by_lead"][str(lead)]
        groups = physics_column_groups(names)
        feature_sets = {
            "history_ridge": history,
            "combined_ridge": combined,
            "history_ib_ridge": np.column_stack([history, physics[:, groups["ib"]]]),
            "history_wind_ridge": np.column_stack([history, physics[:, groups["wind"]]]),
            "history_all_physics_ridge": np.column_stack([history, physics]),
        }
        lead_models: dict[str, Any] = {}
        lead_predictions: dict[str, np.ndarray] = {}
        for name, features in feature_sets.items():
            scaler, model, alpha, tuning_rmse = fit_tuned_ridge(features, target, years)
            predicted = model.predict(scaler.transform(features[validation])).astype(np.float32)
            lead_predictions[name] = predicted
            lead_models[name] = {"scaler": scaler, "model": model, "alpha": alpha}
            selected.append({"model": name, "lead_hours": lead, "alpha": alpha, "2016_rmse_cm": tuning_rmse * 100})
        xgb_features = np.column_stack([history, physics])
        xgb = _xgb(seed, xgb_device, smoke_test)
        xgb.fit(xgb_features[training], target[training])
        lead_predictions["physics_xgboost"] = xgb.predict(xgb_features[validation]).astype(np.float32)
        lead_models["physics_xgboost"] = xgb
        reference = lead_predictions["combined_ridge"]
        observed = target[validation]
        reference_mse = float(np.mean((reference - observed) ** 2))
        threshold = float(np.quantile(np.abs(target[training]), 0.95))
        top = np.abs(observed) >= threshold
        prior = np.asarray(surge[origins[validation] + lead - 1], dtype=float)
        changes = observed - prior
        training_prior = np.asarray(surge[origins[training] + lead - 1], dtype=float)
        training_changes = target[training] - training_prior
        positive = training_changes[training_changes > 0]
        rapid_threshold = float(np.quantile(positive, 0.90)) if len(positive) else np.inf
        rapid = changes >= rapid_threshold
        for name, predicted in lead_predictions.items():
            overall = calculate_metrics(observed, predicted)
            top_metrics = calculate_metrics(observed[top], predicted[top])
            rapid_metrics = calculate_metrics(observed[rapid], predicted[rapid])
            mse = float(np.mean((predicted - observed) ** 2))
            metrics.append({
                "model": name, "lead_hours": lead, **overall,
                "skill_vs_combined_ridge": 1 - mse / reference_mse if reference_mse > 0 else np.nan,
                "top5_n": int(top.sum()), "top5_threshold_cm": threshold * 100,
                "top5_rmse_cm": top_metrics["rmse_cm"],
                "rapid_rise_n": int(rapid.sum()), "rapid_rise_threshold_cm": rapid_threshold * 100,
                "rapid_rise_rmse_cm": rapid_metrics["rmse_cm"],
            })
            prediction_columns[f"{name}_{lead:02d}h_m"] = predicted
        prediction_columns[f"observed_{lead:02d}h_m"] = observed
        prediction_columns[f"valid_time_{lead:02d}h"] = times[origins[validation] + lead]
        saved_models[str(lead)] = lead_models
        print(f"physics baselines lead={lead}h complete", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output / "metrics.csv", index=False)
    metrics_frame[metrics_frame.lead_hours.isin(LEADS)].to_csv(output / "feature_ablation.csv", index=False)
    pd.DataFrame(selected).to_csv(output / "selected_hyperparameters.csv", index=False)
    baseline_predictions = pd.DataFrame(prediction_columns)
    baseline_predictions.to_csv(output / "baseline_predictions_2017.csv", index=False)
    baseline_predictions.to_csv(output / "predictions_2017.csv", index=False)
    joblib.dump(saved_models, output / "physics_baseline_models.joblib", compress=3)
    metadata = {
        "experiment": "Prickly Bay physical feature baselines v1",
        "train_years": [2011, 2016], "alpha_selection_year": 2016,
        "validation_year": 2017, "2018_loaded": False,
        "forecast_leads": list(LEADS), "validation_samples": int(validation.sum()),
        "seed": seed, "xgboost_device": xgb_device,
        "physics_xgboost_inputs": "24-hour surge history plus all lead-safe physics features",
        "smoke_test": smoke_test,
    }
    (output / "baseline_experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics_frame


def main() -> None:
    args = parse_args()
    root = MODULE_ROOT / "outputs" / "experiments" / args.station / "physics_fusion_v1"
    dataset = args.dataset_path or MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    cache = args.feature_cache or root / "physics_features.npz"
    output = args.output_dir or root
    metrics = train_baselines(dataset, cache, output, args.seed, args.xgb_device, args.smoke_test)
    print(metrics[metrics.lead_hours.isin(LEADS)][["model", "lead_hours", "rmse_cm", "top5_rmse_cm"]].to_string(index=False))


if __name__ == "__main__":
    main()
