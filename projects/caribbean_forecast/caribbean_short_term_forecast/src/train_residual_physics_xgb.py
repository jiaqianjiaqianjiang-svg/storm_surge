"""Cross-fitted Ridge plus physical-feature XGBoost residual correction."""

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
    from .evaluate_baselines import summarise_atmosphere
    from .train_physics_baselines import LEADS
    from .train_station import load_prepared
except ImportError:
    from direct_multistep_baselines import ALPHAS, build_feature_matrix, direct_labels
    from evaluate_baselines import summarise_atmosphere
    from train_physics_baselines import LEADS
    from train_station import load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[1]
OOF_YEARS = (2014, 2015, 2016)
CORRECTION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gap-hours", type=int, default=72)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _select_ridge_alpha(
    features: np.ndarray,
    target: np.ndarray,
    years: np.ndarray,
    tuning_year: int,
) -> float:
    early = years < tuning_year
    tuning = years == tuning_year
    if not early.any() or not tuning.any():
        return 10.0
    scaler = StandardScaler().fit(features[early])
    scores = []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver="lsqr").fit(scaler.transform(features[early]), target[early])
        scores.append(np.mean((model.predict(scaler.transform(features[tuning])) - target[tuning]) ** 2))
    return float(ALPHAS[int(np.argmin(scores))])


def expanding_oof_ridge(
    features: np.ndarray,
    target: np.ndarray,
    origin_times: pd.DatetimeIndex,
    gap_hours: int = 72,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Create time-out-of-fold predictions for 2014-2016 with a boundary gap."""
    years = origin_times.year.to_numpy()
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for prediction_year in OOF_YEARS:
        fold_validation = years == prediction_year
        boundary = pd.Timestamp(f"{prediction_year}-01-01") - pd.Timedelta(hours=gap_hours)
        fold_training = (years < prediction_year) & (origin_times <= boundary)
        if not fold_training.any() or not fold_validation.any():
            continue
        alpha = _select_ridge_alpha(
            features[fold_training], target[fold_training], years[fold_training], prediction_year - 1
        )
        scaler = StandardScaler().fit(features[fold_training])
        model = Ridge(alpha=alpha, solver="lsqr").fit(scaler.transform(features[fold_training]), target[fold_training])
        prediction[fold_validation] = model.predict(scaler.transform(features[fold_validation]))
        rows.append({
            "prediction_year": prediction_year, "train_max_time": origin_times[fold_training].max().isoformat(),
            "gap_hours": gap_hours, "train_n": int(fold_training.sum()),
            "prediction_n": int(fold_validation.sum()), "ridge_alpha": alpha,
        })
    mask = np.isfinite(prediction)
    return prediction, mask, rows


def _fit_final_ridge(features: np.ndarray, target: np.ndarray, years: np.ndarray) -> tuple[StandardScaler, Ridge, float]:
    alpha = _select_ridge_alpha(features, target, years, 2016)
    train = years <= 2016
    scaler = StandardScaler().fit(features[train])
    model = Ridge(alpha=alpha, solver="lsqr").fit(scaler.transform(features[train]), target[train])
    return scaler, model, alpha


def _xgb(seed: int, device: str, smoke_test: bool):
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=30 if smoke_test else 300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=10.0,
        objective="reg:squarederror", tree_method="hist", device=device,
        n_jobs=1, random_state=seed,
    )


def train_residual_fusion(
    dataset_path: Path,
    feature_cache: Path,
    output: Path,
    seed: int = 42,
    xgb_device: str = "cuda",
    gap_hours: int = 72,
    smoke_test: bool = False,
) -> pd.DataFrame:
    if gap_hours < 48:
        raise ValueError("gap_hours must cover the 24-hour input and 24-hour target windows")
    atmosphere, surge, times = load_prepared(dataset_path, 2011, 2017)
    if times.max().year >= 2018:
        raise AssertionError("2018 entered residual model development")
    with np.load(feature_cache, allow_pickle=False) as cache:
        cache_times = pd.DatetimeIndex(pd.to_datetime(cache["time"]))
        origins = np.asarray(cache["origin_indices"], dtype=np.int64)
        physics = {lead: np.asarray(cache[f"lead_{lead:02d}h"], dtype=np.float32) for lead in LEADS}
    if not times.equals(cache_times):
        raise ValueError("Physics cache does not match prepared dataset")
    years = times[origins].year.to_numpy()
    keep = years <= 2017
    origins, years = origins[keep], years[keep]
    physics = {lead: values[keep] for lead, values in physics.items()}
    if smoke_test:
        indices = np.concatenate([
            np.flatnonzero(years == year)[-120:] for year in range(2011, 2017)
        ] + [np.flatnonzero(years == 2017)[:120]])
        origins, years = origins[indices], years[indices]
        physics = {lead: values[indices] for lead, values in physics.items()}
    labels = direct_labels(surge, origins, 24)
    summaries = summarise_atmosphere(atmosphere)
    ridge_features = build_feature_matrix(summaries, surge, origins, True, "past_future")
    validation = years == 2017
    prediction_data: dict[str, Any] = {"forecast_origin": times[origins[validation]]}
    selection_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}

    for lead in LEADS:
        target = labels[:, lead - 1]
        oof, oof_mask, folds = expanding_oof_ridge(
            ridge_features, target, pd.DatetimeIndex(times[origins]), gap_hours
        )
        if not oof_mask.any():
            raise ValueError("No OOF Ridge predictions were generated")
        residual_target = target[oof_mask] - oof[oof_mask]
        residual_features = np.column_stack([physics[lead], oof])
        residual_model = _xgb(seed, xgb_device, smoke_test)
        residual_model.fit(residual_features[oof_mask], residual_target)

        scaler, ridge, ridge_alpha = _fit_final_ridge(ridge_features, target, years)
        ridge_validation = ridge.predict(scaler.transform(ridge_features[validation]))
        residual_validation_features = np.column_stack([physics[lead][validation], ridge_validation])
        raw_correction = residual_model.predict(residual_validation_features)
        observed = target[validation]
        scores = [
            float(np.sqrt(np.mean((ridge_validation + alpha * raw_correction - observed) ** 2)))
            for alpha in CORRECTION_ALPHAS
        ]
        best_index = int(np.argmin(scores))
        correction_alpha = float(CORRECTION_ALPHAS[best_index])
        fused = ridge_validation + correction_alpha * raw_correction
        prediction_data[f"observed_{lead:02d}h_m"] = observed
        prediction_data[f"ridge_{lead:02d}h_m"] = ridge_validation
        prediction_data[f"raw_residual_{lead:02d}h_m"] = raw_correction
        prediction_data[f"residual_physics_xgb_{lead:02d}h_m"] = fused
        prediction_data[f"valid_time_{lead:02d}h"] = times[origins[validation] + lead]
        selection_rows.append({
            "lead_hours": lead, "ridge_alpha": ridge_alpha,
            "correction_alpha": correction_alpha, "validation_rmse_cm": scores[best_index] * 100,
            "ridge_validation_rmse_cm": float(np.sqrt(np.mean((ridge_validation - observed) ** 2)) * 100),
            "oof_samples": int(oof_mask.sum()),
        })
        for row in folds:
            fold_rows.append({"lead_hours": lead, **row})
        models[str(lead)] = {
            "ridge_scaler": scaler, "ridge": ridge, "ridge_alpha": ridge_alpha,
            "residual_xgboost": residual_model, "correction_alpha": correction_alpha,
        }
        print(f"residual physics lead={lead}h complete", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(prediction_data)
    predictions.to_csv(output / "residual_predictions_2017.csv", index=False)
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(output / "residual_alpha_selection.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "ridge_oof_folds.csv", index=False)
    joblib.dump(models, output / "residual_physics_models.joblib", compress=3)
    metadata = {
        "experiment": "Ridge plus cross-fitted physical residual XGBoost",
        "train_years": [2011, 2016], "oof_prediction_years": list(OOF_YEARS),
        "validation_year": 2017, "2018_loaded": False, "gap_hours": gap_hours,
        "correction_alpha_candidates": list(CORRECTION_ALPHAS),
        "correction_alpha_selected_on": 2017, "seed": seed,
        "xgboost_device": xgb_device, "smoke_test": smoke_test,
    }
    (output / "residual_experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selection


def main() -> None:
    args = parse_args()
    root = MODULE_ROOT / "outputs" / "experiments" / args.station / "physics_fusion_v1"
    dataset = args.dataset_path or MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    cache = args.feature_cache or root / "physics_features.npz"
    selection = train_residual_fusion(
        dataset, cache, args.output_dir or root, args.seed, args.xgb_device,
        args.gap_hours, args.smoke_test,
    )
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
