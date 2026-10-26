"""仅用临时合成数据验证统计规则，不代表真实站点结果。"""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from src.xiamen_forecast import build_station_event_statistics as statistics
from src.xiamen_forecast import station_config
from src.xiamen_forecast import summarize_typhoon_four_cases as four_cases
from src.xiamen_forecast.station_config import get_station_config
from src.xiamen_forecast.typhoon_events import read_ibtracs_wp


def test_four_cases_keep_exact_and_cumulative_metrics_distinct():
    cases = [(label, four_cases.load_verified_case(four_cases.RESULT_ROOT / package, event_id))
             for package, event_id, label in four_cases.CASES]
    summary, comparison = four_cases.compile_tables(cases)
    assert len(summary) == 4
    assert set(comparison.metric_scope) == {"window_cumulative", "exact_lead_aggregate"}
    assert set(comparison.loc[comparison.metric_scope.eq("exact_lead_aggregate")].n_origins) == {3}
    assert summary.set_index("storm_name").loc["ZITA", "observed_event_peak_cm"] == pytest.approx(78.108549)


def test_valid_targets_and_training_only_thresholds():
    station = replace(get_station_config("xiamen"), train_start_year=1995, train_end_year=1995,
                      validation_year=1996, test_year=1997, input_steps=2)
    times = pd.DatetimeIndex(["1995-01-01 00:00", "1995-01-01 01:00", "1995-01-01 02:00",
                              "1995-01-01 04:00", "1995-01-01 05:00", "1995-01-01 06:00",
                              "1996-01-01 00:00", "1997-01-01 00:00"])
    surge = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 10.0, 20.0])
    valid = statistics.valid_target_mask(times, surge, np.ones(len(times), dtype=bool), 2)
    assert np.flatnonzero(valid).tolist() == [2, 5]
    thresholds = statistics.training_thresholds(times, surge, valid, station)
    assert thresholds["p99_m"] == pytest.approx(np.quantile([0.03, 0.06], 0.99))
    bad_atmosphere = np.ones(len(times), dtype=bool)
    bad_atmosphere[1] = False
    assert not statistics.valid_target_mask(times, surge, bad_atmosphere, 2)[2]


def test_peak_clustering_and_boundary():
    times = pd.date_range("1995-12-31 20:00", periods=60, freq="h")
    surge = np.zeros(len(times))
    surge[[3, 5, 31]] = [0.2, 0.3, 0.4]
    short = statistics.independent_peak_events(times, surge, 0.1, 12)
    long = statistics.independent_peak_events(times, surge, 0.1, 48)
    assert len(short) == 2
    assert len(long) == 1
    assert short.iloc[0].peak_cm == pytest.approx(30)
    assert short.iloc[0].above_hours == 2


def _synthetic_csv(path):
    path.write_text(
        "SID,SEASON,NAME,ISO_TIME,LAT,LON,BASIN,WMO_WIND,WMO_PRES,TRACK_TYPE\n"
        ",,,UTC,degrees_north,degrees_east,,kt,mb,\n"
        "1996001,1996,TEST_A,1996-08-01 12:00:00,24.45,118.067,WP,50,985,MAIN\n"
        "1997001,1997,TEST_B,1997-01-01 00:00:00,24.45,118.067,WP,50,985,MAIN\n",
        encoding="utf-8",
    )


def test_synthetic_cli_generates_small_outputs_and_refuses_overwrite(tmp_path, monkeypatch):
    station = replace(get_station_config("xiamen"), data_start_year=1995, train_start_year=1995,
                      train_end_year=1995, validation_year=1996, test_year=1997,
                      input_steps=2, grid_size=1)
    monkeypatch.setattr(station_config, "MODULE_ROOT", tmp_path)
    monkeypatch.setattr(statistics, "get_station_config", lambda name: station)
    root = station.dataset_dir
    root.mkdir(parents=True)
    times = pd.date_range("1995-01-01", "1998-01-01", freq="h", inclusive="left")
    surge = np.sin(np.arange(len(times)) / 7) * 0.05 + 0.1
    surge[np.asarray(times.year == 1997)] += 3.0
    np.save(root / "time.npy", times.to_numpy(dtype="datetime64[ns]"))
    np.save(root / "surge.npy", surge.astype("f4"))
    np.save(root / "atmosphere.npy", np.zeros((len(times), 3, 1, 1), dtype="f4"))
    (root / "dataset_metadata.json").write_text(json.dumps({"station_id": "xiamen"}), encoding="utf-8")
    track = tmp_path / "ibtracs.csv"
    _synthetic_csv(track)
    assert len(read_ibtracs_wp(track, (1995, 1997))) == 2
    output = tmp_path / "statistics"
    statistics.main(["--stations", "xiamen", "--track-csv", str(track), "--output-dir", str(output)])
    for name in ("station_typhoon_counts_by_split.csv", "station_typhoon_event_catalog.csv",
                 "station_extreme_sample_distribution.csv", "station_independent_peak_events.csv",
                 "station_typhoon_extreme_crosswalk.csv", "fig01_typhoon_counts_by_split.png",
                 "fig02_surge_threshold_distribution.png", "fig03_extreme_events_by_split.png",
                 "statistics_methodology.md", "statistics_summary.md"):
        assert (output / name).is_file()
    table = pd.read_csv(output / "station_extreme_sample_distribution.csv")
    assert table.loc[table.split.eq("train"), "training_p99_threshold_cm"].iloc[0] < 100
    assert table.loc[table.split.eq("test"), "p99_hours"].iloc[0] > 0
    counts = pd.read_csv(output / "station_typhoon_counts_by_split.csv").set_index("split")
    assert counts.loc["validation", "data_complete_same_split_candidates"] == 1
    assert counts.loc["test", "cross_split_window_candidates"] == 1
    with pytest.raises(SystemExit, match="2"):
        statistics.main(["--stations", "xiamen", "--track-csv", str(track), "--output-dir", str(output)])


def test_missing_prepared_data_names_paths(tmp_path, monkeypatch):
    station = replace(get_station_config("xiamen"), grid_size=1)
    monkeypatch.setattr(station_config, "MODULE_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="atmosphere.npy"):
        statistics.load_station_arrays(station)
