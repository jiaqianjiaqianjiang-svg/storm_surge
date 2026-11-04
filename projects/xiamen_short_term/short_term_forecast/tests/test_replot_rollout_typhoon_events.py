"""归档预测后处理：阈值来源、成对样本和输出隔离。"""

import numpy as np
import pandas as pd
import pytest

from src.xiamen_forecast.replot_rollout_typhoon_events import (
    ROOT, SOURCES, aligned_forecasts, event_table, extreme_rows, load_thresholds, run,
)


def test_fixed_threshold_sources():
    values, source = load_thresholds("xiamen")
    assert source == "training_target_all_valid"
    assert values["p95"] == pytest.approx(40.2332805097103)
    values, source = load_thresholds("kanmen")
    assert source == "validation_1996_lead1_positive"
    assert values["p99"] > values["p95"]


def test_archived_event_paired_and_metrics():
    event = SOURCES[1] / "kanmen_1997217N06168"
    _, wide = aligned_forecasts(event / "forecasts_72h.csv")
    assert len(wide) == 216
    thresholds, source = load_thresholds("kanmen")
    extremes = extreme_rows(wide, "WINNIE", thresholds, source)
    assert set(extremes.sample_group) == {"all", "p95", "p99"}
    assert len(extremes) == 18
    all_rows = extremes.loc[extremes.sample_group == "all"]
    assert set(all_rows.n_samples) == {72}
    assert (extremes.mean_underestimation_cm.dropna() >= 0).all()
    metrics = pd.read_csv(event / "event_metrics.csv")
    table = event_table(metrics, extremes)
    assert len(table) == 6
    row = table.loc[(table.origin_offset_hours == -72) & (table.model == "cnn_gru_rollout6")].iloc[0]
    assert row.rmse72_cm == pytest.approx(21.28530316209959)


def test_refuses_existing_output(tmp_path):
    target = tmp_path / "review"
    target.mkdir()
    with pytest.raises(FileExistsError):
        run(destination=target)


def test_missing_extreme_samples_are_not_fabricated():
    event = SOURCES[1] / "kanmen_1997217N06168"
    _, wide = aligned_forecasts(event / "forecasts_72h.csv")
    table = extreme_rows(wide, "WINNIE", {"p95": 1000., "p99": 2000.}, "synthetic_test")
    empty = table.loc[table.sample_group != "all"]
    assert (empty.n_samples == 0).all()
    assert empty.rmse_cm.isna().all()
