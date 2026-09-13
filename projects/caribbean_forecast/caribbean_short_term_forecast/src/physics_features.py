"""Leakage-safe physical features derived from U10, V10 and MSL.

All public functions use SI units. ERA5 input order is fixed to U10 [m s-1],
V10 [m s-1], MSL [Pa]. Accumulated stresses are returned in Pa s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_371_000.0
DEFAULT_RHO_AIR = 1.225
DEFAULT_RHO_WATER = 1025.0
DEFAULT_GRAVITY = 9.80665
SECONDS_PER_HOUR = 3600.0


def inverse_barometer_height(
    pressure_pa: object,
    reference_pressure_pa: object,
    rho_water: float = DEFAULT_RHO_WATER,
    gravity: float = DEFAULT_GRAVITY,
) -> np.ndarray:
    """Return inverse-barometer sea-level response in metres."""
    if rho_water <= 0 or gravity <= 0:
        raise ValueError("rho_water and gravity must be positive")
    return -(
        np.asarray(pressure_pa, dtype=np.float64)
        - np.asarray(reference_pressure_pa, dtype=np.float64)
    ) / (rho_water * gravity)


def drag_coefficient(speed_ms: object, law: str = "garratt", maximum: float = 0.003) -> np.ndarray:
    """Return dimensionless surface drag coefficient for non-negative wind speed.

    Garratt: ``(0.75 + 0.067 U) 1e-3``. Wu uses 1.2875e-3 below
    7.5 m/s and ``(0.8 + 0.065 U) 1e-3`` above it. Both are capped.
    """
    speed = np.asarray(speed_ms, dtype=np.float64)
    if np.any(speed < 0) or maximum <= 0:
        raise ValueError("Wind speed must be non-negative and maximum must be positive")
    key = law.lower()
    if key == "garratt":
        coefficient = (0.75 + 0.067 * speed) * 1e-3
    elif key == "wu":
        coefficient = np.where(speed < 7.5, 1.2875e-3, (0.8 + 0.065 * speed) * 1e-3)
    else:
        raise ValueError("drag law must be 'garratt' or 'wu'")
    return np.clip(coefficient, 0.0, float(maximum))


def wind_stress(
    u10_ms: object,
    v10_ms: object,
    rho_air: float = DEFAULT_RHO_AIR,
    drag_law: str = "garratt",
    maximum_drag: float = 0.003,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return speed [m/s], Cd [-], east/north stress and magnitude [Pa]."""
    if rho_air <= 0:
        raise ValueError("rho_air must be positive")
    u = np.asarray(u10_ms, dtype=np.float64)
    v = np.asarray(v10_ms, dtype=np.float64)
    speed = np.hypot(u, v)
    coefficient = drag_coefficient(speed, drag_law, maximum_drag)
    tau_east = rho_air * coefficient * speed * u
    tau_north = rho_air * coefficient * speed * v
    return speed, coefficient, tau_east, tau_north, np.hypot(tau_east, tau_north)


def bearing_unit_vectors(bearing_deg_clockwise_from_north: float) -> tuple[float, float, float, float]:
    """Return onshore east/north and alongshore east/north unit vectors."""
    theta = np.deg2rad(float(bearing_deg_clockwise_from_north))
    n_east, n_north = float(np.sin(theta)), float(np.cos(theta))
    return n_east, n_north, -n_north, n_east


def project_wind_stress(
    tau_east_pa: object,
    tau_north_pa: object,
    bearing_deg_clockwise_from_north: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project stress onto positive-offshore-to-bay and alongshore axes [Pa]."""
    n_east, n_north, s_east, s_north = bearing_unit_vectors(
        bearing_deg_clockwise_from_north
    )
    east = np.asarray(tau_east_pa, dtype=np.float64)
    north = np.asarray(tau_north_pa, dtype=np.float64)
    return east * n_east + north * n_north, east * s_east + north * s_north


def normalise_longitudes(longitude_deg: object) -> np.ndarray:
    """Normalise longitudes to [-180, 180) without changing array order."""
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    return (longitude + 180.0) % 360.0 - 180.0


def pressure_gradients(
    pressure_pa: object,
    latitude_deg: object,
    longitude_deg: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eastward and northward pressure gradients in Pa/m.

    Latitude may be ascending or descending. Longitude can use either the
    -180..180 or 0..360 convention, provided points follow a continuous axis.
    """
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    latitude = np.asarray(latitude_deg, dtype=np.float64)
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    if pressure.shape[-2:] != (len(latitude), len(longitude)):
        raise ValueError("pressure trailing dimensions must match latitude/longitude")
    if len(latitude) < 2 or len(longitude) < 2:
        raise ValueError("At least two latitude and longitude coordinates are required")
    lat_rad = np.deg2rad(latitude)
    lon_rad = np.unwrap(np.deg2rad(longitude))
    if np.any(np.diff(lat_rad) == 0) or np.any(np.diff(lon_rad) == 0):
        raise ValueError("Coordinate values must be unique")
    north = np.gradient(pressure, lat_rad, axis=-2, edge_order=1) / EARTH_RADIUS_M
    d_pressure_d_lon = np.gradient(pressure, lon_rad, axis=-1, edge_order=1)
    cos_lat = np.cos(lat_rad).reshape((1,) * (pressure.ndim - 2) + (len(latitude), 1))
    if np.any(np.abs(cos_lat) < 1e-6):
        raise ValueError("Longitude gradient is unstable at the poles")
    east = d_pressure_d_lon / (EARTH_RADIUS_M * cos_lat)
    return east, north


def local_pressure_gradient(
    pressure_pa: object,
    latitude_deg: object,
    longitude_deg: object,
    latitude_index: int,
    longitude_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return station-cell east/north gradients [Pa/m] without a full-grid copy."""
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    latitude = np.asarray(latitude_deg, dtype=np.float64)
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    if pressure.shape[-2:] != (len(latitude), len(longitude)):
        raise ValueError("pressure trailing dimensions must match latitude/longitude")
    lat_rad = np.deg2rad(latitude)
    lon_rad = np.unwrap(np.deg2rad(longitude))
    north_line = pressure[..., :, longitude_index]
    east_line = pressure[..., latitude_index, :]
    north = np.gradient(north_line, lat_rad, axis=-1, edge_order=1)[..., latitude_index] / EARTH_RADIUS_M
    east = np.gradient(east_line, lon_rad, axis=-1, edge_order=1)[..., longitude_index]
    east = east / (EARTH_RADIUS_M * np.cos(lat_rad[latitude_index]))
    return east, north


def accumulated_forcing(values: object, hours: int, step_seconds: float = SECONDS_PER_HOUR) -> np.ndarray:
    """Trailing integral ending at each row, with incomplete windows set NaN."""
    array = np.asarray(values, dtype=np.float64)
    if hours <= 0 or step_seconds <= 0:
        raise ValueError("hours and step_seconds must be positive")
    result = np.full(array.shape, np.nan, dtype=np.float64)
    if len(array) < hours:
        return result
    finite = np.isfinite(array)
    safe = np.where(finite, array, 0.0)
    prefix = np.concatenate([np.zeros((1,) + array.shape[1:]), np.cumsum(safe, axis=0)], axis=0)
    counts = np.concatenate([np.zeros((1,) + array.shape[1:], dtype=np.int64), np.cumsum(finite, axis=0)], axis=0)
    sums = prefix[hours:] - prefix[:-hours]
    valid = counts[hours:] - counts[:-hours] == hours
    result[hours - 1 :] = np.where(valid, sums * step_seconds, np.nan)
    return result


def assert_hourly_window(times: object, start: int, stop: int) -> bool:
    """Return whether [start, stop) is finite-length, in-bounds and exactly hourly."""
    index = pd.DatetimeIndex(pd.to_datetime(times))
    if start < 0 or stop > len(index) or stop <= start:
        return False
    return bool(np.all(np.diff(index[start:stop].values) == np.timedelta64(1, "h")))


def future_window(origin: int, lead_hours: int) -> slice:
    """Return the only permitted future forcing interval: t+1 through t+h."""
    if lead_hours <= 0:
        raise ValueError("lead_hours must be positive")
    return slice(int(origin) + 1, int(origin) + int(lead_hours) + 1)


def circular_lon_distance(longitude: np.ndarray, station_longitude: float) -> np.ndarray:
    return np.abs((normalise_longitudes(longitude) - normalise_longitudes(station_longitude) + 180) % 360 - 180)


def spatial_region_masks(
    latitude: object,
    longitude: object,
    station_latitude: float,
    station_longitude: float,
    regions: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Build simple latitude/longitude-radius masks, always including station cell."""
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    nearest_lat = int(np.argmin(np.abs(lat - float(station_latitude))))
    nearest_lon = int(np.argmin(circular_lon_distance(lon, float(station_longitude))))
    masks: dict[str, np.ndarray] = {}
    for name, radius in regions.items():
        radius = float(radius)
        if radius < 0:
            raise ValueError("Spatial region radius cannot be negative")
        mask = (
            (np.abs(lat[:, None] - station_latitude) <= radius)
            & (circular_lon_distance(lon[None, :], station_longitude) <= radius)
        )
        if radius == 0:
            mask[:] = False
        mask[nearest_lat, nearest_lon] = True
        masks[str(name)] = mask
    return masks


@dataclass
class HourlyPhysics:
    values: np.ndarray
    names: list[str]
    units: list[str]
    metadata: dict[str, Any]


def derive_hourly_physics(
    atmosphere: object,
    latitude: object,
    longitude: object,
    station_latitude: float,
    station_longitude: float,
    training_mask: object,
    config: Mapping[str, Any],
    chunk_hours: int = 168,
) -> HourlyPhysics:
    """Reduce 40x40 ERA5 grids to auditable hourly physical descriptors."""
    data = atmosphere
    if len(data.shape) != 4 or data.shape[1] != 3:
        raise ValueError("atmosphere must have shape (time, 3, latitude, longitude)")
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    if data.shape[-2:] != (len(lat), len(lon)):
        raise ValueError("ERA5 coordinate shape does not match prepared atmosphere")
    train = np.asarray(training_mask, dtype=bool)
    if train.shape != (len(data),) or not train.any():
        raise ValueError("training_mask must select at least one hour")
    rho_air = float(config.get("rho_air", DEFAULT_RHO_AIR))
    rho_water = float(config.get("rho_water", DEFAULT_RHO_WATER))
    gravity = float(config.get("gravity", DEFAULT_GRAVITY))
    law = str(config.get("drag_law", "garratt"))
    regions = config.get("spatial_regions") or {"station": 0.0, "regional": 5.0}
    masks = spatial_region_masks(lat, lon, station_latitude, station_longitude, regions)
    station_mask = next((mask for name, mask in masks.items() if name == "station"), None)
    if station_mask is None:
        station_mask = spatial_region_masks(lat, lon, station_latitude, station_longitude, {"station": 0})["station"]
    station_i, station_j = np.argwhere(station_mask)[0]

    # First pass obtains a training-only pressure reference.
    total = 0.0
    count = 0
    for start in range(0, len(data), chunk_hours):
        stop = min(len(data), start + chunk_hours)
        selected = train[start:stop]
        if not selected.any():
            continue
        local = np.asarray(data[start:stop, 2, station_i, station_j], dtype=np.float64)
        finite = selected & np.isfinite(local)
        total += float(local[finite].sum())
        count += int(finite.sum())
    if count == 0:
        raise ValueError("Training local pressure contains no finite values")
    climatology_pa = total / count

    names = [
        "pressure_local_pa", "pressure_regional_mean_pa", "pressure_anomaly_clim_pa",
        "pressure_anomaly_regional_pa", "inverse_barometer_clim_m",
        "inverse_barometer_regional_m", "pressure_gradient_east_pa_m",
        "pressure_gradient_north_pa_m", "wind_speed_local_ms", "drag_coefficient_local",
        "wind_stress_east_local_pa", "wind_stress_north_local_pa", "wind_stress_magnitude_local_pa",
    ]
    units = ["Pa", "Pa", "Pa", "Pa", "m", "m", "Pa/m", "Pa/m", "m/s", "1", "Pa", "Pa", "Pa"]
    for region_name in masks:
        if region_name == "station":
            continue
        names.extend([
            f"{region_name}_pressure_mean_pa", f"{region_name}_pressure_min_pa",
            f"{region_name}_wind_stress_mean_pa", f"{region_name}_wind_stress_max_pa",
        ])
        units.extend(["Pa", "Pa", "Pa", "Pa"])
    bearing = config.get("onshore_bearing_deg_clockwise_from_north")
    if bearing is not None:
        names.extend(["wind_stress_onshore_local_pa", "wind_stress_alongshore_local_pa"])
        units.extend(["Pa", "Pa"])
    output = np.empty((len(data), len(names)), dtype=np.float32)

    for start in range(0, len(data), chunk_hours):
        stop = min(len(data), start + chunk_hours)
        chunk = np.asarray(data[start:stop], dtype=np.float64)
        u, v, pressure = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        speed, cd, tau_e, tau_n, tau_mag = wind_stress(u, v, rho_air, law)
        grad_e_local, grad_n_local = local_pressure_gradient(
            pressure, lat, lon, station_i, station_j
        )
        p_local = pressure[:, station_i, station_j]
        p_regional = np.nanmean(pressure, axis=(1, 2))
        _, cd_local, tau_e_local, tau_n_local, tau_mag_local = wind_stress(
            u[:, station_i, station_j], v[:, station_i, station_j], rho_air, law
        )
        columns: list[np.ndarray] = [
            p_local, p_regional, p_local - climatology_pa, p_local - p_regional,
            inverse_barometer_height(p_local, climatology_pa, rho_water, gravity),
            inverse_barometer_height(p_local, p_regional, rho_water, gravity),
            grad_e_local, grad_n_local,
            speed[:, station_i, station_j], cd_local, tau_e_local, tau_n_local, tau_mag_local,
        ]
        for region_name, mask in masks.items():
            if region_name == "station":
                continue
            columns.extend([
                np.nanmean(pressure[:, mask], axis=1), np.nanmin(pressure[:, mask], axis=1),
                np.nanmean(tau_mag[:, mask], axis=1), np.nanmax(tau_mag[:, mask], axis=1),
            ])
        if bearing is not None:
            onshore, alongshore = project_wind_stress(tau_e_local, tau_n_local, float(bearing))
            columns.extend([onshore, alongshore])
        output[start:stop] = np.column_stack(columns).astype(np.float32)

    metadata = {
        "variable_order": ["U10", "V10", "MSL"],
        "source_units": ["m/s", "m/s", "Pa"],
        "rho_air_kg_m3": rho_air,
        "rho_water_kg_m3": rho_water,
        "gravity_m_s2": gravity,
        "drag_law": law,
        "pressure_climatology_pa": climatology_pa,
        "pressure_climatology_source": "training years only",
        "station_grid_index": [int(station_i), int(station_j)],
        "station_grid_coordinate": [float(lat[station_i]), float(normalise_longitudes(lon[station_j]))],
        "onshore_bearing_deg_clockwise_from_north": bearing,
        "onshore_features_available": bearing is not None,
        "spatial_regions": {name: {"radius_degrees": float(regions[name]), "grid_cells": int(mask.sum())} for name, mask in masks.items()},
    }
    return HourlyPhysics(output, names, units, metadata)


def build_origin_physics_features(
    hourly_values: object,
    hourly_names: Sequence[str],
    times: object,
    origins: object,
    lead_hours: int,
    past_windows: Sequence[int] = (3, 6, 12, 24),
) -> tuple[np.ndarray, list[str]]:
    """Build features at origins; future features stop exactly at t+lead."""
    values = np.asarray(hourly_values, dtype=np.float64)
    index = pd.DatetimeIndex(pd.to_datetime(times))
    origins_array = np.asarray(origins, dtype=np.int64)
    if values.shape != (len(index), len(hourly_names)):
        raise ValueError("hourly values, names and times do not align")
    name_to_index = {name: i for i, name in enumerate(hourly_names)}
    required = [
        "pressure_local_pa", "wind_stress_east_local_pa",
        "wind_stress_north_local_pa", "wind_stress_magnitude_local_pa",
    ]
    missing = [name for name in required if name not in name_to_index]
    if missing:
        raise ValueError(f"Hourly physics is missing required columns: {missing}")
    force_names = [name for name in hourly_names if "wind_stress_" in name and name.endswith("_pa")]
    force_names += [
        name for name in ("pressure_anomaly_clim_pa", "pressure_anomaly_regional_pa")
        if name in name_to_index
    ]
    feature_names = [f"current_{name}" for name in hourly_names]
    feature_names += [f"pressure_change_{hours}h_pa" for hours in (1, 3, 6)]
    for hours in past_windows:
        feature_names += [f"past_{hours}h_integral_{name}_pa_s" for name in force_names]
    feature_names += [
        f"future_{lead_hours}h_integral_{name}_pa_s" for name in force_names
    ]
    feature_names += [
        f"future_{lead_hours}h_min_pressure_pa",
        f"future_{lead_hours}h_min_pressure_offset_h",
        f"future_{lead_hours}h_max_wind_stress_pa",
        f"future_{lead_hours}h_max_wind_stress_offset_h",
    ]
    if "wind_stress_onshore_local_pa" in name_to_index:
        feature_names += [
            f"future_{lead_hours}h_positive_onshore_impulse_pa_s",
            f"future_{lead_hours}h_negative_onshore_impulse_pa_s",
        ]
    if not len(origins_array):
        return np.empty((0, len(feature_names)), dtype=np.float32), feature_names
    history_start = origins_array - max(max(past_windows), 6) + 1
    future_stop = origins_array + lead_hours + 1
    if history_start.min() < 0 or future_stop.max() > len(index):
        raise ValueError("An origin has an incomplete physics window")
    if not np.all(np.diff(index.values) == np.timedelta64(1, "h")):
        raise ValueError("Physics time axis contains a non-hourly interval")
    p = name_to_index["pressure_local_pa"]
    force_idx = np.asarray([name_to_index[name] for name in force_names], dtype=int)
    parts: list[np.ndarray] = [values[origins_array]]
    parts.append(np.column_stack([
        values[origins_array, p] - values[origins_array - hours, p]
        for hours in (1, 3, 6)
    ]))
    forces = values[:, force_idx]
    prefix = np.vstack([np.zeros((1, len(force_idx))), np.cumsum(forces, axis=0)])
    for hours in past_windows:
        parts.append(
            (prefix[origins_array + 1] - prefix[origins_array - hours + 1])
            * SECONDS_PER_HOUR
        )
    parts.append(
        (prefix[origins_array + lead_hours + 1] - prefix[origins_array + 1])
        * SECONDS_PER_HOUR
    )
    offsets = np.arange(1, lead_hours + 1, dtype=int)
    future_indices = origins_array[:, None] + offsets[None, :]
    future_pressure = values[future_indices, p]
    stress = values[future_indices, name_to_index["wind_stress_magnitude_local_pa"]]
    parts.append(np.column_stack([
        np.min(future_pressure, axis=1), np.argmin(future_pressure, axis=1) + 1,
        np.max(stress, axis=1), np.argmax(stress, axis=1) + 1,
    ]))
    if "wind_stress_onshore_local_pa" in name_to_index:
        onshore = values[future_indices, name_to_index["wind_stress_onshore_local_pa"]]
        parts.append(np.column_stack([
            np.maximum(onshore, 0).sum(axis=1) * SECONDS_PER_HOUR,
            np.minimum(onshore, 0).sum(axis=1) * SECONDS_PER_HOUR,
        ]))
    matrix = np.column_stack(parts).astype(np.float32)
    invalid = np.flatnonzero(~np.isfinite(matrix).all(axis=1))
    if len(invalid):
        raise ValueError(f"Non-finite physics feature at origin {index[origins_array[invalid[0]]]}")
    if matrix.shape[1] != len(feature_names):
        raise AssertionError("Physics feature names and matrix width disagree")
    return matrix, feature_names
