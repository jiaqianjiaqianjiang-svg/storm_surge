import numpy as np
import pandas as pd

from analysis.station_mechanism_comparison.run_analysis import (
    acf_for_series,
    extreme_comparison,
    merge_observed_evidence,
    skill_vs_persistence,
)


def test_acf_detects_persistent_hourly_series() -> None:
    rng = np.random.default_rng(4)
    values = np.cumsum(rng.normal(size=300))
    frame = pd.DataFrame({
        "datetime": pd.date_range("2018-01-01", periods=len(values), freq="h"),
        "surge_m": values,
    })
    acf = acf_for_series(frame, max_lag=24)
    assert len(acf) == 25
    assert acf.loc[0, "acf"] == 1.0
    assert acf.loc[1, "acf"] > 0.8


def test_skill_and_extreme_tables_use_same_lead_persistence() -> None:
    rows = []
    for station in ("Xiamen", "Prickly Bay"):
        for lead in (1, 24, 72):
            for model, rmse in (("Persistence", 4.0), ("Ridge", 2.0), ("CNN-GRU", 3.0), ("Rollout-6", 2.5)):
                rows.append({
                    "station": station, "lead_hours": lead, "model": model,
                    "rmse_cm": rmse, "top5_rmse_cm": rmse * 2,
                    "rapid_rise_rmse_cm": rmse * 1.5,
                    "top5_threshold_cm": 10, "rapid_rise_threshold_cm_per_hour": 2,
                    "top5_n": 20, "rapid_rise_n": 15,
                })
    metrics = pd.DataFrame(rows)
    skill = skill_vs_persistence(metrics)
    ridge = skill[skill.model.eq("Ridge")]
    assert np.allclose(ridge.skill_rmse_vs_persistence, 0.5)
    extreme = extreme_comparison(metrics)
    ridge_extreme = extreme[extreme.model.eq("Ridge")]
    assert np.allclose(ridge_extreme.skill_rmse_vs_persistence, 0.5)


def test_merge_observed_evidence_uses_git_safe_derived_tables(tmp_path) -> None:
    derived = tmp_path / "xiamen"
    derived.mkdir()
    pd.DataFrame({
        "station": ["Xiamen", "Xiamen"],
        "lag_hours": [0, 1], "acf": [1.0, 0.8], "pair_count": [10, 9],
    }).to_csv(derived / "station_acf_full.csv", index=False)
    pd.DataFrame({
        "station": ["Xiamen"], "n": [10], "unit": ["cm"],
        "mean_cm": [0.0], "std_cm": [2.0], "mean_absolute_cm": [1.5],
        "p90_absolute_cm": [3.0], "p95_absolute_cm": [3.5],
        "p99_absolute_cm": [4.0], "min_cm": [-4.0], "max_cm": [4.0],
        "max_absolute_cm": [4.0],
    }).to_csv(derived / "station_surge_scale.csv", index=False)

    acfs, scales, _ = merge_observed_evidence(
        series={}, derived_dirs={"Xiamen": derived, "Prickly Bay": None}
    )

    assert set(acfs) == {"Xiamen"}
    assert acfs["Xiamen"].acf.tolist() == [1.0, 0.8]
    assert scales.station.tolist() == ["Xiamen"]
