"""Overall, extreme-event and reference-skill metrics reported in centimetres."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_pearson_r(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return float("nan")
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(np.clip(np.sum(x * y) / denominator, -1.0, 1.0))


def calculate_metrics(observed_m: object, predicted_m: object) -> dict[str, float | int]:
    observed = np.asarray(observed_m, dtype=float)
    predicted = np.asarray(predicted_m, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    if not len(observed):
        return {
            "n": 0, "pearson_r": float("nan"), "r2": float("nan"),
            "rmse_cm": float("nan"), "mae_cm": float("nan"),
            "bias_cm": float("nan"), "rrmse_percent": float("nan"),
        }
    error_cm = (predicted - observed) * 100
    rmse_cm = float(np.sqrt(np.mean(error_cm**2)))
    denominator = float(np.mean(np.abs(observed * 100)))
    correlation = _safe_pearson_r(observed, predicted)
    total_variance = float(np.sum((observed - observed.mean()) ** 2))
    residual_variance = float(np.sum((predicted - observed) ** 2))
    return {
        "n": len(observed),
        "pearson_r": correlation,
        "r2": float(1 - residual_variance / total_variance) if total_variance > 0 else float("nan"),
        "rmse_cm": rmse_cm,
        "mae_cm": float(np.mean(np.abs(error_cm))),
        "bias_cm": float(np.mean(error_cm)),
        "rrmse_percent": float(rmse_cm / denominator * 100) if denominator > 0 else float("nan"),
    }


def _subset_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    threshold_cm: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = calculate_metrics(observed[mask], predicted[mask])
    if threshold_cm is not None:
        result["threshold_cm"] = threshold_cm
    return result


def _peak_event_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    dates: pd.DatetimeIndex,
    peak_count: int = 10,
    separation_hours: int = 24,
    search_window_hours: int = 12,
) -> dict[str, Any]:
    candidates = np.argsort(np.abs(observed))[::-1]
    selected: list[int] = []
    for index in candidates:
        if all(
            abs((dates[index] - dates[other]).total_seconds()) / 3600 >= separation_hours
            for other in selected
        ):
            selected.append(int(index))
        if len(selected) == peak_count:
            break
    events = []
    for index in selected:
        time_distance = np.abs((dates - dates[index]).total_seconds() / 3600)
        window = np.flatnonzero(time_distance <= search_window_hours)
        predicted_peak = int(window[np.argmax(np.abs(predicted[window]))])
        events.append(
            {
                "observed_peak_time": dates[index].isoformat(),
                "observed_peak_m": float(observed[index]),
                "prediction_at_observed_peak_m": float(predicted[index]),
                "peak_error_cm": float((predicted[index] - observed[index]) * 100),
                "predicted_local_peak_time": dates[predicted_peak].isoformat(),
                "timing_error_hours": float(
                    (dates[predicted_peak] - dates[index]).total_seconds() / 3600
                ),
                "underestimated_absolute_peak": bool(
                    abs(predicted[index]) < abs(observed[index])
                ),
            }
        )
    absolute_errors = np.asarray([abs(event["peak_error_cm"]) for event in events])
    timing_errors = np.asarray([abs(event["timing_error_hours"]) for event in events])
    maximum = selected[0]
    return {
        "event_count": len(events),
        "minimum_separation_hours": separation_hours,
        "local_peak_search_window_hours": search_window_hours,
        "mean_absolute_peak_error_cm": float(absolute_errors.mean()),
        "mean_absolute_timing_error_hours": float(timing_errors.mean()),
        "maximum_observed_m": float(observed[maximum]),
        "prediction_at_maximum_observed_m": float(predicted[maximum]),
        "maximum_peak_error_cm": float((predicted[maximum] - observed[maximum]) * 100),
        "underestimated_maximum_absolute_peak": bool(
            abs(predicted[maximum]) < abs(observed[maximum])
        ),
        "events": events,
    }


def calculate_detailed_metrics(
    observed_m: object,
    predicted_m: object,
    dates: object,
    reference_predicted_m: object | None = None,
) -> dict[str, Any]:
    """Calculate overall, high-surge, rapid-rise, peak and reference-skill metrics."""
    observed = np.asarray(observed_m, dtype=float)
    predicted = np.asarray(predicted_m, dtype=float)
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    valid = np.isfinite(observed) & np.isfinite(predicted) & ~index.isna()
    observed, predicted, index = observed[valid], predicted[valid], index[valid]
    result: dict[str, Any] = {"overall": calculate_metrics(observed, predicted)}
    if reference_predicted_m is not None:
        reference = np.asarray(reference_predicted_m, dtype=float)[valid]
        reference_valid = np.isfinite(reference)
        model_mse = float(np.mean((predicted[reference_valid] - observed[reference_valid]) ** 2))
        reference_mse = float(np.mean((reference[reference_valid] - observed[reference_valid]) ** 2))
        result["skill_score_vs_ridge"] = (
            float(1 - model_mse / reference_mse) if reference_mse > 0 else float("nan")
        )
        result["ridge_reference_n"] = int(reference_valid.sum())

    absolute = np.abs(observed)
    for percentile, label in ((0.90, "top_absolute_10_percent"), (0.95, "top_absolute_5_percent")):
        threshold = float(np.quantile(absolute, percentile))
        result[label] = _subset_metrics(
            observed, predicted, absolute >= threshold, threshold * 100
        )

    contiguous = np.zeros(len(index), dtype=bool)
    contiguous[1:] = np.diff(index.values) == np.timedelta64(1, "h")
    rises = np.full(len(observed), np.nan, dtype=float)
    rises[1:] = np.diff(observed)
    positive_rises = rises[contiguous & (rises > 0)]
    if len(positive_rises):
        rise_threshold = float(np.quantile(positive_rises, 0.90))
        rapid_mask = contiguous & (rises >= rise_threshold)
        result["rapid_rise_top_10_percent"] = _subset_metrics(
            observed, predicted, rapid_mask, rise_threshold * 100
        )
        result["rapid_rise_top_10_percent"]["threshold_definition"] = (
            "90th percentile of positive one-hour observed rises"
        )
    else:
        result["rapid_rise_top_10_percent"] = {"n": 0}
    result["peak_events"] = _peak_event_metrics(observed, predicted, index)
    return result
