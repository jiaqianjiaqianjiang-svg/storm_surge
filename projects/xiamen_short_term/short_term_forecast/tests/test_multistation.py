from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.xiamen_forecast.audit_prepared_dataset import audit_dataset
from src.xiamen_forecast.dataset_builder import (
    build_train_validation_year_datasets,
)
from src.xiamen_forecast.era5_loader import inspect_era5_files
from src.xiamen_forecast import prepare_xiamen
from src.xiamen_forecast.run_final_test import build_commands
from src.xiamen_forecast.station_config import (
    STATIONS,
    apply_station_defaults,
    get_station_config,
    validate_dataset_identity,
    validate_output_location,
)
from src.xiamen_forecast.tide_quality_control import quality_control


def test_station_configuration_has_exact_years_paths_and_isolated_outputs() -> None:
    xiamen = get_station_config("xiamen")
    lianyungang = get_station_config("Lian-yun-gang")
    beihai = get_station_config("bei hai")

    assert (xiamen.train_start_year, xiamen.test_year) == (1970, 1997)
    assert (lianyungang.train_start_year, lianyungang.test_year) == (1975, 1997)
    assert (beihai.train_start_year, beihai.test_year) == (1975, 1997)
    assert lianyungang.latitude == pytest.approx(34.750)
    assert beihai.longitude == pytest.approx(109.083)
    assert lianyungang.era5_filenames["U10"] == (
        "lianyungang_10u_1975_1997.nc"
    )
    assert str(beihai.gesla_file).endswith("beihai-636a-chn-uhslc")
    assert len({station.dataset_dir for station in STATIONS.values()}) == 3
    assert len({station.model_root for station in STATIONS.values()}) == 3


def test_station_defaults_preserve_explicit_overrides() -> None:
    arguments = argparse.Namespace(
        station="lianyungang", train_start_year=None, validation_year=2001
    )
    station = apply_station_defaults(
        arguments,
        {
            "train_start_year": "train_start_year",
            "validation_year": "validation_year",
        },
    )
    assert station.station_id == "lianyungang"
    assert arguments.train_start_year == 1975
    assert arguments.validation_year == 2001


def test_quality_control_reports_and_rejects_gesla_5_0_pair() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("1995-01-01", periods=4, freq="1h"),
            "water_level": [1.0, 1.1, 8.0, 1.2],
            "qc_flag": ["1", "1", "5", "2"],
            "use_flag": ["1", "1", "0", "1"],
            "sensor": ["default"] * 4,
        }
    )
    clean, report = quality_control(frame)
    assert clean.water_level.tolist() == [1.0, 1.1, 1.2]
    assert report["quality_use_flag_pairs"]["5|0"] == 1
    assert report["removed_bad_qc_and_use_count"] == 1


def test_dataset_identity_blocks_cross_station_and_missing_metadata(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "aligned_dataset"
    dataset.mkdir()
    metadata = {"station_id": "lianyungang"}
    (dataset / "dataset_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    assert validate_dataset_identity(dataset, "lianyungang") == metadata
    with pytest.raises(ValueError, match="not 'beihai'"):
        validate_dataset_identity(dataset, "beihai")
    (dataset / "dataset_metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata is missing"):
        validate_dataset_identity(dataset, "lianyungang")


def test_output_location_rejects_cross_station_directory(tmp_path: Path) -> None:
    lianyungang = tmp_path / "models" / "lianyungang"
    beihai = tmp_path / "models" / "beihai"
    accepted = validate_output_location(
        lianyungang / "formal_seed42", lianyungang, "Model output"
    )
    assert accepted == (lianyungang / "formal_seed42").resolve()
    with pytest.raises(ValueError, match="must remain inside"):
        validate_output_location(beihai / "formal_seed42", lianyungang, "Model output")


def test_validation_only_dataset_rejects_future_year_and_uses_training_scalers() -> None:
    times = pd.date_range("1994-12-29", "1996-01-04", freq="1h")
    atmosphere = np.zeros((len(times), 3, 2, 2), dtype=np.float32)
    surge = np.linspace(-0.2, 0.3, len(times), dtype=np.float32)
    train, validation, report = build_train_validation_year_datasets(
        atmosphere,
        surge,
        times,
        input_steps=24,
        train_start_year=1994,
        train_end_year=1995,
        validation_year=1996,
    )
    assert len(train) and len(validation)
    assert report["test_year_loaded"] is False
    assert max(train.times[train.targets]).year == 1995
    with pytest.raises(ValueError, match="later than the validation year"):
        build_train_validation_year_datasets(
            atmosphere,
            surge,
            times.append(pd.DatetimeIndex([pd.Timestamp("1997-01-01")])),
            input_steps=24,
            train_start_year=1994,
            train_end_year=1995,
            validation_year=1996,
        )


def _write_era5_file(
    path: Path,
    variable: str,
    unit: str,
    times: pd.DatetimeIndex,
    longitudes: np.ndarray | None = None,
) -> None:
    latitudes = np.array([33.0, 34.0], dtype=np.float32)
    longitudes = (
        np.array([118.0, 119.0], dtype=np.float32)
        if longitudes is None
        else longitudes
    )
    values = np.zeros(
        (len(times), len(latitudes), len(longitudes)), dtype=np.float32
    )
    data = xr.Dataset(
        {
            variable: (
                ("time", "latitude", "longitude"),
                values,
                {"units": unit},
            )
        },
        coords={
            "time": times,
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )
    data.to_netcdf(path)


def test_era5_source_audit_checks_units_and_coordinate_identity(
    tmp_path: Path,
) -> None:
    times = pd.date_range("1995-01-01", periods=4, freq="1h")
    paths = [tmp_path / name for name in ("u.nc", "v.nc", "p.nc")]
    _write_era5_file(paths[0], "u10", "m s**-1", times)
    _write_era5_file(paths[1], "v10", "m s**-1", times)
    _write_era5_file(paths[2], "msl", "Pa", times)
    report = inspect_era5_files(paths)
    assert report["status"] == "passed"
    assert report["coordinate_axes_identical"] is True
    assert list(report["variables"]) == ["U10", "V10", "MSL"]

    _write_era5_file(
        paths[2],
        "msl",
        "hPa",
        times,
        longitudes=np.array([118.0, 120.0], dtype=np.float32),
    )
    failed = inspect_era5_files(paths)
    assert failed["status"] == "failed"
    assert failed["coordinate_axes_identical"] is False
    assert any("unexpected MSL unit" in item for item in failed["errors"])


def test_synthetic_preparation_writes_only_selected_station_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    station = get_station_config("lianyungang")
    era5_dir = tmp_path / "raw_era5"
    era5_dir.mkdir()
    for filename in station.era5_filenames.values():
        (era5_dir / filename).touch()
    gesla = tmp_path / "raw_gesla" / station.gesla_file.name
    gesla.parent.mkdir()
    gesla.write_text("synthetic", encoding="utf-8")
    output = tmp_path / "project_outputs" / "lianyungang"
    pieces = [
        pd.date_range(f"{year}-01-01", periods=48, freq="1h")
        for year in (1995, 1996, 1997)
    ]
    all_times = pieces[0].append(pieces[1:])
    raw = pd.DataFrame(
        {
            "datetime": all_times,
            "water_level": np.sin(np.arange(len(all_times)) / 12),
            "qc_flag": "1",
            "use_flag": "1",
            "sensor": "default",
        }
    )

    monkeypatch.setattr(prepare_xiamen, "load_tide_gauge", lambda *a, **k: raw)
    monkeypatch.setattr(
        prepare_xiamen,
        "inspect_era5_files",
        lambda paths: {"status": "passed", "errors": [], "variables": {}},
    )

    def fake_separate(frame, latitude, output_dir, calibration_end):
        return (
            pd.DataFrame(
                {
                    "datetime": frame.datetime,
                    "storm_surge_m": frame.water_level * 0.01,
                }
            ),
            {"calibration_end": calibration_end.isoformat()},
        )

    def fake_era5(paths, latitude, longitude, **kwargs):
        start = pd.Timestamp(kwargs["start_time"])
        selected = all_times[all_times.year == start.year]
        values = np.zeros((len(selected), 3, 2, 2), dtype=np.float32)
        values[:, 2] = 101325.0
        return xr.DataArray(
            values,
            dims=("time", "variable", "latitude", "longitude"),
            coords={
                "time": selected,
                "variable": ["U10", "V10", "MSL"],
                "latitude": [34.0, 35.0],
                "longitude": [119.0, 120.0],
            },
        )

    monkeypatch.setattr(prepare_xiamen, "separate_tide", fake_separate)
    monkeypatch.setattr(prepare_xiamen, "load_era5_files", fake_era5)
    dataset = prepare_xiamen.prepare_station(
        "lianyungang",
        era5_dir=era5_dir,
        gesla_file=gesla,
        output_dir=output,
        start_year=1995,
        end_year=1997,
        calibration_end_year=1995,
        grid_size=2,
    )
    metadata = json.loads(
        (dataset / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["station_id"] == "lianyungang"
    assert metadata["utide_calibration_end_year"] == 1995
    assert np.load(dataset / "atmosphere.npy", mmap_mode="r").shape == (
        len(all_times),
        3,
        2,
        2,
    )
    assert not (tmp_path / "project_outputs" / "beihai").exists()


def test_synthetic_audit_reports_split_samples_and_72h_origins(
    tmp_path: Path,
) -> None:
    output = tmp_path / "outputs" / "processed" / "lianyungang"
    dataset = output / "aligned_dataset"
    dataset.mkdir(parents=True)
    times = pd.date_range("1995-12-28", "1997-01-06", freq="1h")
    atmosphere = np.empty((len(times), 3, 2, 2), dtype=np.float32)
    atmosphere[:, 0] = 5.0
    atmosphere[:, 1] = -2.0
    atmosphere[:, 2] = 101325.0
    surge = (0.15 * np.sin(np.arange(len(times)) / 12)).astype(np.float32)
    np.save(dataset / "atmosphere.npy", atmosphere)
    np.save(dataset / "surge.npy", surge)
    np.save(dataset / "time.npy", times.to_numpy(dtype="datetime64[ns]"))
    (dataset / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "station_id": "lianyungang",
                "prepared_years": [1995, 1997],
                "utide_calibration_end_year": 1995,
                "array_schema": {
                    "atmosphere_variable_order": ["U10", "V10", "MSL"]
                },
            }
        ),
        encoding="utf-8",
    )
    (output / "preparation_report.json").write_text(
        json.dumps(
            {
                "gesla_raw_time_range": [times[0].isoformat(), times[-1].isoformat()],
                "selected_period_observation_count": len(times),
                "years": [],
                "era5_files": ["u.nc", "v.nc", "p.nc"],
                "era5_source_audit": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )
    (output / "tide_qc_report.json").write_text(
        json.dumps(
            {
                "raw_record_count": len(times),
                "removed_quality_flag_count": 0,
                "quality_use_flag_pairs": {"1|1": len(times)},
            }
        ),
        encoding="utf-8",
    )
    report = audit_dataset(
        dataset,
        station_id="lianyungang",
        train_start_year=1995,
        train_end_year=1995,
        validation_year=1996,
        test_year=1997,
        expected_grid_size=2,
    )
    assert report["status"] == "passed"
    assert all(report["split_valid_samples"][name] > 0 for name in ("training", "validation", "test"))
    assert report["common_72h_origins"]["validation"] > 0
    assert report["common_72h_origins"]["test"] > 0
    assert report["variables"][2]["expected_unit"] == "Pa"


def test_final_test_commands_propagate_station_to_every_entrypoint(
    tmp_path: Path,
) -> None:
    commands = build_commands(
        1997,
        42,
        "cuda",
        "en",
        tmp_path / "model.pth",
        station="lianyungang",
    )
    assert all("--station" in command for command in commands)
    assert all("lianyungang" in command for command in commands)
    rolling = commands[3]
    assert "--models" in rolling
    assert "cnn_gru" in rolling
