import numpy as np
import pandas as pd
import xarray as xr

from caribbean_short_term_forecast.src.time_alignment import align_hourly


def test_exact_utc_alignment_and_report() -> None:
    era_times = pd.date_range("2011-01-01", periods=5, freq="h")
    era5 = xr.DataArray(
        np.zeros((5, 3, 2, 2), dtype=np.float32),
        dims=("time", "variable", "latitude", "longitude"),
        coords={"time": era_times, "variable": ["U10", "V10", "MSL"]},
    )
    surge = pd.DataFrame(
        {
            "datetime": pd.date_range("2011-01-01 01:00", periods=3, freq="h"),
            "storm_surge_m": [0.1, 0.2, 0.3],
        }
    )
    aligned_era, aligned_surge, report = align_hourly(era5, surge, "UTC")
    assert aligned_era.shape[0] == len(aligned_surge) == 3
    assert report["matched_times"] == 3
    assert report["missing_surge_times"] == 2
    assert report["continuous_segments"] == 1
