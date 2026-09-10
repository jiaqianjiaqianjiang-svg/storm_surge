import numpy as np

from caribbean_short_term_forecast.src.physics_features import (
    inverse_barometer_height,
    project_wind_stress,
    wind_stress,
)


def test_inverse_barometer_one_hpa_low_is_about_one_centimetre_high():
    value = inverse_barometer_height(100_900.0, 101_000.0)
    assert np.isclose(value, 0.009946, rtol=2e-3)


def test_zero_and_opposite_winds_have_physical_stress_signs():
    speed, _, tau_east, tau_north, magnitude = wind_stress([0.0, 10.0, -10.0], [0.0, 0.0, 0.0])
    assert speed[0] == 0 and tau_east[0] == 0 and tau_north[0] == 0 and magnitude[0] == 0
    assert tau_east[1] > 0 and tau_east[2] < 0
    assert np.isclose(abs(tau_east[1]), abs(tau_east[2]))


def test_stress_grows_approximately_quadratically_with_speed():
    _, _, tau, _, _ = wind_stress([5.0, 10.0], [0.0, 0.0])
    assert tau[1] / tau[0] > 3.0


def test_onshore_projection_uses_offshore_to_bay_bearing():
    # 90 degrees clockwise from north points east.
    onshore, alongshore = project_wind_stress([2.0], [0.0], 90.0)
    assert np.isclose(onshore[0], 2.0)
    assert np.isclose(alongshore[0], 0.0, atol=1e-12)
