from pathlib import Path

import numpy as np

from caribbean_short_term_forecast.src.tide_gauge_loader import load_tide_gauge
from caribbean_short_term_forecast.src.tide_quality_control import quality_control


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_headerless_gesla_loading_and_qc() -> None:
    source = FIXTURES / "synthetic_gesla.txt"
    frame = load_tide_gauge(source, "test_station", source="GESLA 4.0", water_level_unit="m")
    assert list(frame.columns[:7]) == [
        "datetime", "water_level", "qc_flag", "use_flag", "source", "sensor", "station_id"
    ]
    clean, report = quality_control(frame)
    assert len(clean) == 2
    assert report["removed_duplicate_count"] == 1
    assert report["removed_quality_flag_count"] == 1
    assert report["selected_sensor"] == "primary"
    assert np.allclose(clean.water_level, [1.2, 1.3])


def test_csv_aliases_and_unit_conversion() -> None:
    source = FIXTURES / "synthetic_noc_table.txt"
    frame = load_tide_gauge(source, "test_station", source="PSMSL/NOC", water_level_unit="cm")
    assert np.isclose(frame.water_level.iloc[0], 1.25)
