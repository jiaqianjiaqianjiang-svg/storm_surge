import numpy as np

from caribbean_short_term_forecast.src.physics_features import EARTH_RADIUS_M, pressure_gradients


def test_linear_pressure_field_gradient_has_pa_per_m_units():
    latitude = np.array([11.5, 12.0, 12.5])
    longitude = np.array([-62.5, -62.0, -61.5])
    lat_rad = np.deg2rad(latitude)
    lon_rad = np.deg2rad(longitude)
    north_expected = 2.0e-5
    east_expected = 3.0e-5
    pressure = (
        101_000.0
        + north_expected * EARTH_RADIUS_M * (lat_rad[:, None] - lat_rad[1])
        + east_expected * EARTH_RADIUS_M * np.cos(lat_rad[1]) * (lon_rad[None, :] - lon_rad[1])
    )
    east, north = pressure_gradients(pressure, latitude, longitude)
    assert np.isclose(east[1, 1], east_expected, rtol=2e-4)
    assert np.isclose(north[1, 1], north_expected, rtol=2e-4)


def test_latitude_order_and_longitude_convention_do_not_change_gradient():
    latitude = np.array([11.5, 12.0, 12.5])
    longitude = np.array([-62.5, -62.0, -61.5])
    pressure = 100_000 + np.arange(3)[:, None] * 20 + np.arange(3)[None, :] * 10
    east, north = pressure_gradients(pressure, latitude, longitude)
    east_desc, north_desc = pressure_gradients(pressure[::-1], latitude[::-1], longitude + 360)
    np.testing.assert_allclose(east, east_desc[::-1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(north, north_desc[::-1], rtol=1e-12, atol=1e-12)
