import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.xiamen_forecast.dataset_builder import build_year_datasets
from src.xiamen_forecast.compare_models import collect_model_metrics
from src.xiamen_forecast.era5_loader import load_era5_files
from src.xiamen_forecast.evaluate import calculate_detailed_metrics
from src.xiamen_forecast.forecast_model import create_model
from src.xiamen_forecast.prepare_xiamen import resolve_era5_files
from src.xiamen_forecast.rolling_diagnostics import display_name
from src.xiamen_forecast.tide_quality_control import quality_control
from src.xiamen_forecast.train_rollout_cnn import rollout_forward, rollout_origins
from src.xiamen_forecast.train_rollout_temporal import temporal_rollout_forward
from src.xiamen_forecast.train_xiamen import load_prepared


def test_resolve_xiamen_split_variable_files(tmp_path: Path) -> None:
    for name in (
        "xiamen_10u_1970_1997.nc",
        "xiamen_v10_1970_1997.nc",
        "xiamen_slp_1970_1997.nc",
    ):
        (tmp_path / name).touch()
    paths = resolve_era5_files(tmp_path)
    assert [path.name for path in paths] == [
        "xiamen_10u_1970_1997.nc",
        "xiamen_v10_1970_1997.nc",
        "xiamen_slp_1970_1997.nc",
    ]


def test_gesla_quality_flags_keep_correct_and_interpolated_records() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("1985-01-01", periods=5, freq="1h"),
            "water_level": [1.0, 1.1, 1.2, 1.3, 1.4],
            "qc_flag": ["0", "1", "2", "3", "1"],
            "use_flag": ["1", "1", "1", "1", "0"],
            "sensor": ["default"] * 5,
        }
    )

    clean, report = quality_control(frame)

    assert clean["water_level"].tolist() == [1.0, 1.1, 1.2]
    assert report["removed_quality_flag_count"] == 2


def test_year_split_is_temporal_and_scalers_use_training_data() -> None:
    times = pd.date_range("1994-01-01", "1997-01-03", freq="1h")
    atmosphere = np.zeros((len(times), 3, 2, 2), dtype=np.float32)
    surge = np.sin(np.arange(len(times), dtype=np.float32) / 24)
    train, validation, test, report = build_year_datasets(
        atmosphere,
        surge,
        times,
        input_steps=24,
        train_start_year=1994,
        train_end_year=1995,
        validation_year=1996,
        test_year=1997,
    )
    assert len(train) and len(validation) and len(test)
    assert report["train_years"] == [1994, 1995]
    assert report["validation_year"] == 1996
    assert report["test_year"] == 1997
    assert max(train.times[train.targets]).year == 1995
    assert min(test.times[test.targets]).year == 1997


@pytest.mark.parametrize(
    "model_type",
    [
        "cnn",
        "cnn_gru",
        "cnn_lstm",
        "dual",
        "era5_cnn",
        "surge_mlp",
        "tcn",
        "transformer",
    ],
)
def test_models_accept_24_hour_windows(model_type: str) -> None:
    model = create_model(model_type, 24, ("U10", "V10", "MSL"), 40)
    weather = torch.zeros(1, 72, 40, 40)
    history = torch.zeros(1, 24)
    assert model(weather, history).shape == (1,)


def test_rollout_training_reuses_predictions_without_future_truth() -> None:
    model = create_model("cnn", 4, ("U10", "V10", "MSL"), 8)
    weather = torch.zeros(2, 3, 128)
    history = torch.zeros(2, 4)
    targets = torch.full((2, 3), 99.0)
    predicted = rollout_forward(model, weather, history, targets, teacher_ratio=0.0)
    assert predicted.shape == (2, 3)
    assert torch.isfinite(predicted).all()


def test_temporal_encoded_forward_matches_raw_weather_forward() -> None:
    model = create_model("cnn_gru", 4, ("U10", "V10", "MSL"), 8).eval()
    weather = torch.randn(2, 12, 8, 8)
    history = torch.randn(2, 4)
    with torch.inference_mode():
        embeddings = model.weather_features(weather)
        raw_prediction = model(weather, history)
        encoded_prediction = model.forward_encoded(embeddings, history)
    assert torch.allclose(raw_prediction, encoded_prediction)


def test_temporal_rollout_does_not_use_targets_without_teacher_forcing() -> None:
    model = create_model("cnn_gru", 4, ("U10", "V10", "MSL"), 8).eval()
    weather = torch.randn(2, 3, 4, 64)
    history = torch.randn(2, 4)
    first_targets = torch.zeros(2, 3)
    second_targets = torch.full((2, 3), 100.0)
    with torch.inference_mode():
        first = temporal_rollout_forward(
            model, weather, history, first_targets, teacher_ratio=0.0
        )
        second = temporal_rollout_forward(
            model, weather, history, second_targets, teacher_ratio=0.0
        )
    assert torch.allclose(first, second)
    assert display_name("cnn_gru_rollout6") == "CNN-GRU rollout-6"


def test_rollout_origins_respect_year_and_complete_windows() -> None:
    times = pd.date_range("1995-12-31 20:00", periods=36, freq="1h")
    valid = np.ones(len(times), dtype=bool)
    surge = np.zeros(len(times), dtype=np.float32)
    origins = rollout_origins(times, valid, surge, {1996}, 4, 3)
    assert len(origins)
    assert all(times[index].year == 1996 for index in origins)
    assert all(times[index + 2].year == 1996 for index in origins)


def test_model_comparison_collects_formal_and_baseline_metrics(tmp_path: Path) -> None:
    model_root = tmp_path / "formal_seed42"
    baseline_dir = tmp_path / "baselines"
    (model_root / "cnn").mkdir(parents=True)
    baseline_dir.mkdir()
    overall = {
        "n": 10,
        "pearson_r": 0.9,
        "rmse_cm": 4.0,
        "mae_cm": 3.0,
        "bias_cm": 0.2,
    }
    (model_root / "cnn" / "metrics.json").write_text(
        json.dumps({"validation": {"overall": overall}}), encoding="utf-8"
    )
    (baseline_dir / "baseline_metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "validation": {
                        "persistence": {"overall": overall},
                        "ridge": {"overall": overall},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.warns(UserWarning):
        frame = collect_model_metrics(model_root, baseline_dir)
    assert frame.model.tolist() == ["persistence", "ridge", "cnn"]


def test_prepared_directory_loads_as_memory_maps(tmp_path: Path) -> None:
    dataset = tmp_path / "aligned_dataset"
    dataset.mkdir()
    times = pd.date_range("1996-01-01", periods=72, freq="1h")
    np.save(dataset / "atmosphere.npy", np.zeros((72, 3, 2, 2), dtype=np.float32))
    np.save(dataset / "surge.npy", np.zeros(72, dtype=np.float32))
    np.save(dataset / "time.npy", times.to_numpy(dtype="datetime64[ns]"))
    atmosphere, surge, loaded_times = load_prepared(dataset, 1996, 1996)
    assert atmosphere.shape == (72, 3, 2, 2)
    assert surge.shape == (72,)
    assert loaded_times.equals(times)


def test_detailed_metrics_include_extreme_and_peak_sections() -> None:
    observed = np.sin(np.arange(240) / 12) * 0.3
    predicted = observed + 0.01
    times = pd.date_range("1996-01-01", periods=len(observed), freq="1h")
    metrics = calculate_detailed_metrics(observed, predicted, times)
    assert metrics["overall"]["rmse_cm"] == pytest.approx(1.0)
    assert "top_absolute_5_percent" in metrics
    assert "rapid_rise_top_10_percent" in metrics
    assert metrics["peak_events"]["event_count"] > 0


def test_metrics_do_not_call_numpy_corrcoef(monkeypatch) -> None:
    from src.short_term_forecast.journal_figures.io_utils import (
        compute_metrics_from_predictions,
    )

    def aborting_corrcoef(*args, **kwargs):
        raise AssertionError("np.corrcoef must not be used on the Windows MKL path")

    monkeypatch.setattr(np, "corrcoef", aborting_corrcoef)
    frame = pd.DataFrame(
        {"observed": [1.0, 2.0, 3.0], "predicted": [1.1, 2.1, 3.1]}
    )
    metrics = compute_metrics_from_predictions(frame)
    assert metrics["pearson_r"] == pytest.approx(1.0)


def test_era5_loader_is_python39_compatible(monkeypatch, tmp_path: Path) -> None:
    import xarray as xr
    from src.xiamen_forecast import era5_loader

    times = pd.date_range("1996-01-01", periods=2, freq="1h")
    latitude = np.array([-1.0, 1.0])
    longitude = np.array([-1.0, 1.0])
    datasets = {}
    paths = []
    for variable, value in (("u10", 1.0), ("v10", 2.0), ("msl", 3.0)):
        path = tmp_path / f"{variable}.mock"
        path.touch()
        paths.append(path)
        datasets[path.name] = xr.Dataset(
            {
                variable: (
                    ("time", "latitude", "longitude"),
                    np.full((2, 2, 2), value, dtype=np.float32),
                )
            },
            coords={"time": times, "latitude": latitude, "longitude": longitude},
        )
    monkeypatch.setattr(era5_loader, "_open", lambda path: datasets[path.name])
    result = load_era5_files(paths, 0.0, 0.0, grid_size=2, region_size_degrees=4)
    assert list(result.coords["variable"].values) == ["U10", "V10", "MSL"]
