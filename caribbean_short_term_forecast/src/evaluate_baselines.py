"""Evaluate zero, persistence and compact-feature ridge baselines by year."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .dataset_builder import build_year_datasets
    from .evaluate import calculate_metrics
    from .train_station import load_prepared
except ImportError:
    from dataset_builder import build_year_datasets
    from evaluate import calculate_metrics
    from train_station import load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[1]


def summarise_atmosphere(atmosphere: np.ndarray, chunk_hours: int = 168) -> np.ndarray:
    """Return spatial mean/std/min/max for each time and ERA5 variable."""
    summaries = np.empty((len(atmosphere), atmosphere.shape[1], 4), dtype=np.float32)
    for start in range(0, len(atmosphere), chunk_hours):
        chunk = np.asarray(atmosphere[start : start + chunk_hours], dtype=np.float32)
        stop = start + len(chunk)
        summaries[start:stop, :, 0] = chunk.mean(axis=(2, 3))
        summaries[start:stop, :, 1] = chunk.std(axis=(2, 3))
        summaries[start:stop, :, 2] = chunk.min(axis=(2, 3))
        summaries[start:stop, :, 3] = chunk.max(axis=(2, 3))
    return summaries


def build_ridge_features(
    summaries: np.ndarray,
    surge: np.ndarray,
    targets: list[int],
    input_steps: int,
) -> np.ndarray:
    weather_width = input_steps * summaries.shape[1] * summaries.shape[2]
    features = np.empty((len(targets), weather_width + input_steps), dtype=np.float32)
    for row, target in enumerate(targets):
        features[row, :weather_width] = summaries[
            target - input_steps : target
        ].reshape(-1)
        features[row, weather_width:] = surge[target - input_steps : target]
    return features


def evaluate_baselines(
    dataset_path: str | Path,
    output_dir: str | Path,
    input_steps: int = 24,
    train_start_year: int = 2011,
    train_end_year: int = 2016,
    validation_year: int = 2017,
    test_year: int = 2018,
    ridge_alpha: float = 10.0,
) -> dict[str, Any]:
    atmosphere, surge, times = load_prepared(
        Path(dataset_path), train_start_year, test_year
    )
    train, validation, test, split_report = build_year_datasets(
        atmosphere, surge, times, input_steps,
        train_start_year, train_end_year, validation_year, test_year,
    )
    summaries = summarise_atmosphere(atmosphere)
    features = {
        "train": build_ridge_features(summaries, surge, train.targets, input_steps),
        "validation": build_ridge_features(
            summaries, surge, validation.targets, input_steps
        ),
        "test": build_ridge_features(summaries, surge, test.targets, input_steps),
    }
    observed = {
        "train": np.asarray(surge[train.targets], dtype=np.float32),
        "validation": np.asarray(surge[validation.targets], dtype=np.float32),
        "test": np.asarray(surge[test.targets], dtype=np.float32),
    }
    ridge = Pipeline(
        [
            ("standardise", StandardScaler()),
            ("ridge", Ridge(alpha=ridge_alpha)),
        ]
    )
    ridge.fit(features["train"], observed["train"])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge, destination / "ridge_pipeline.joblib")

    datasets = {"validation": validation, "test": test}
    report: dict[str, Any] = {
        "split": split_report,
        "ridge": {
            "alpha": ridge_alpha,
            "feature_definition": (
                "24 hourly spatial mean/std/min/max for U10/V10/MSL "
                "plus 24 hourly observed surge history"
            ),
            "feature_count": int(features["train"].shape[1]),
        },
        "metrics": {},
    }
    for split_name, dataset in datasets.items():
        targets = dataset.targets
        predictions = {
            "zero": np.zeros(len(targets), dtype=np.float32),
            "persistence": np.asarray(surge[np.asarray(targets) - 1], dtype=np.float32),
            "ridge": ridge.predict(features[split_name]).astype(np.float32),
        }
        report["metrics"][split_name] = {
            name: calculate_metrics(observed[split_name], values)
            for name, values in predictions.items()
        }
        frame = pd.DataFrame(
            {
                "datetime": dataset.times[targets],
                "observed_m": observed[split_name],
                "zero_m": predictions["zero"],
                "persistence_m": predictions["persistence"],
                "ridge_m": predictions["ridge"],
            }
        )
        frame.to_csv(destination / f"{split_name}_baseline_predictions.csv", index=False)
    (destination / "baseline_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-steps", type=int, default=24)
    parser.add_argument("--train-start-year", type=int, default=2011)
    parser.add_argument("--train-end-year", type=int, default=2016)
    parser.add_argument("--validation-year", type=int, default=2017)
    parser.add_argument("--test-year", type=int, default=2018)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    dataset = arguments.dataset_path or (
        MODULE_ROOT / "outputs" / "processed" / arguments.station / "aligned_dataset"
    )
    output = arguments.output_dir or MODULE_ROOT / "models" / arguments.station / "baselines"
    result = evaluate_baselines(
        dataset, output, arguments.input_steps,
        arguments.train_start_year, arguments.train_end_year,
        arguments.validation_year, arguments.test_year,
        arguments.ridge_alpha,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
