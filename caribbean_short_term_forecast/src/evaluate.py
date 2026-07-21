"""Forecast metrics reported in centimetres."""

from __future__ import annotations

import numpy as np


def calculate_metrics(observed_m: object, predicted_m: object) -> dict[str, float | int]:
    observed = np.asarray(observed_m, dtype=float)
    predicted = np.asarray(predicted_m, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    if not len(observed):
        return {"n": 0, "pearson_r": float("nan"), "rmse_cm": float("nan"), "mae_cm": float("nan"), "bias_cm": float("nan"), "rrmse_percent": float("nan")}
    error_cm = (predicted - observed) * 100
    rmse_cm = float(np.sqrt(np.mean(error_cm**2)))
    denominator = float(np.mean(np.abs(observed * 100)))
    correlation = float(np.corrcoef(observed, predicted)[0, 1]) if len(observed) > 1 and np.std(observed) > 0 and np.std(predicted) > 0 else float("nan")
    return {
        "n": len(observed),
        "pearson_r": correlation,
        "rmse_cm": rmse_cm,
        "mae_cm": float(np.mean(np.abs(error_cm))),
        "bias_cm": float(np.mean(error_cm)),
        "rrmse_percent": float(rmse_cm / denominator * 100) if denominator > 0 else float("nan"),
    }
