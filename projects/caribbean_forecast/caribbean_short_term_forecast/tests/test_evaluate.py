import numpy as np
import pandas as pd

from caribbean_short_term_forecast.src.evaluate import (
    calculate_detailed_metrics,
    calculate_metrics,
)


def test_r2_and_reference_skill_score() -> None:
    observed = np.linspace(-0.2, 0.3, 100)
    predicted = observed + 0.01
    reference = observed + 0.02
    dates = pd.date_range("2018-01-01", periods=100, freq="h")
    overall = calculate_metrics(observed, predicted)
    detailed = calculate_detailed_metrics(observed, predicted, dates, reference)
    assert overall["r2"] > 0.99
    assert np.isclose(detailed["skill_score_vs_ridge"], 0.75)
    assert detailed["top_absolute_10_percent"]["n"] >= 10
    assert detailed["peak_events"]["event_count"] > 0
