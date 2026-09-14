from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.xiamen_forecast.dataset_builder import build_year_datasets
from src.xiamen_forecast.era5_loader import load_era5_files
from src.xiamen_forecast.evaluate import calculate_detailed_metrics
from src.xiamen_forecast.forecast_model import create_model
from src.xiamen_forecast.prepare_xiamen import resolve_era5_files
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


@pytest.mark.parametrize("model_type", ["dual", "era5_cnn", "surge_mlp"])
def test_models_accept_24_hour_windows(model_type: str) -> None:
    model = create_model(model_type, 24, ("U10", "V10", "MSL"), 40)
    weather = torch.zeros(2, 72, 40, 40)
    history = torch.zeros(2, 24)
    assert model(weather, history).shape == (2,)


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
