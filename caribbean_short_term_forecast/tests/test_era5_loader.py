from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from caribbean_short_term_forecast.src import era5_loader
from caribbean_short_term_forecast.src.prepare_station import group_era5_files_by_period


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _single_variable_dataset(name: str, value: float) -> xr.Dataset:
    return xr.Dataset(
        {
            name: (
                ("time", "latitude", "longitude"),
                np.full((2, 2, 2), value, dtype=np.float32),
            )
        },
        coords={
            "time": pd.date_range("2011-01-01", periods=2, freq="1h"),
            "latitude": [-1.0, 1.0],
            "longitude": [-1.0, 1.0],
        },
    )


def test_split_variable_files_are_merged_in_canonical_order(monkeypatch) -> None:
    datasets = {
        "u10_2011.mock": _single_variable_dataset("u10", 1.0),
        "v10_2011.mock": _single_variable_dataset("v10", 2.0),
        "msl_2011.mock": _single_variable_dataset("msl", 3.0),
    }
    monkeypatch.setattr(era5_loader, "_open", lambda path: datasets[path.name])
    paths = [FIXTURES / name for name in datasets]
    result = era5_loader.load_era5_files(
        paths, latitude=0.0, longitude=0.0, grid_size=2, region_size_degrees=4.0
    )
    assert result.dims == ("time", "variable", "latitude", "longitude")
    assert result.coords["variable"].values.tolist() == ["U10", "V10", "MSL"]
    assert np.allclose(result[:, 0], 1.0)
    assert np.allclose(result[:, 1], 2.0)
    assert np.allclose(result[:, 2], 3.0)


def test_era5_files_are_grouped_by_year() -> None:
    paths = [
        Path("caribbean_10u_2012.nc"),
        Path("caribbean_slp_2011.nc"),
        Path("caribbean_v10_2011.nc"),
        Path("caribbean_10u_2011.nc"),
    ]
    groups = group_era5_files_by_period(paths)
    assert [period for period, _ in groups] == ["2011", "2012"]
    assert [path.name for path in groups[0][1]] == [
        "caribbean_10u_2011.nc",
        "caribbean_slp_2011.nc",
        "caribbean_v10_2011.nc",
    ]
