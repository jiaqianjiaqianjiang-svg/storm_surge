import numpy as np
import pandas as pd

from src.rolling_diagnostics import build_metrics, find_common_origins


def test_common_origins_require_every_requested_target_and_stay_in_year():
    times = pd.date_range("2017-01-01", periods=130, freq="h")
    atmosphere_valid = np.ones(len(times), dtype=bool)
    surge = np.arange(len(times), dtype=np.float32)
    surge[30] = np.nan
    origins = find_common_origins(
        times, atmosphere_valid, surge, 2017, input_steps=24, max_lead=72
    )
    assert 30 not in origins  # lead 1 is missing
    assert 28 not in origins  # lead 3 is missing
    assert 24 in origins      # unverified intermediate hours need not be observed
    assert 7 not in origins   # cannot provide the 24-hour initial history
    assert origins[-1] == 58


def test_metrics_use_same_lead_ridge_as_skill_reference():
    origins = np.asarray([24, 25, 26])
    surge = np.linspace(0.0, 0.2, 100, dtype=np.float32)
    predictions = {
        name: np.zeros((3, 72), dtype=np.float32)
        for name in ("persistence", "ridge", "surge_mlp", "era5_cnn", "dual")
    }
    for lead in range(72):
        observed = surge[origins + lead]
        predictions["ridge"][:, lead] = observed + 0.02
        predictions["dual"][:, lead] = observed + 0.01
    metrics = build_metrics(surge, origins, predictions)
    dual_one = metrics[(metrics.lead_hours == 1) & (metrics.model == "dual")].iloc[0]
    ridge_one = metrics[(metrics.lead_hours == 1) & (metrics.model == "ridge")].iloc[0]
    assert np.isclose(dual_one.skill_vs_same_lead_ridge, 0.75)
    assert np.isclose(ridge_one.skill_vs_same_lead_ridge, 0.0)
