"""Configuration-driven station registry with strict missing-field checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = MODULE_ROOT / "configs" / "stations.yaml"


@dataclass(frozen=True)
class Station:
    station_id: str
    station_name: str
    source: str | None
    latitude: float | None
    longitude: float | None
    start_time: str | None
    end_time: str | None
    timezone: str | None
    water_level_unit: str | None
    file_path: str | None
    era5_path: str | None

    def require(self, *fields: str) -> "Station":
        missing = [name for name in fields if getattr(self, name, None) in (None, "")]
        if missing:
            names = ", ".join(missing)
            raise ValueError(
                f"Station '{self.station_id}' is missing required configuration: {names}. "
                "Run inspect_data.py, then update configs/stations.yaml with verified values."
            )
        return self


def load_station_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Station]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Station registry not found: {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    records = raw.get("stations")
    if not isinstance(records, dict):
        raise ValueError(f"Expected a 'stations' mapping in {path}")
    registry: dict[str, Station] = {}
    for key, values in records.items():
        values = values or {}
        values.setdefault("station_id", key)
        unknown = set(values) - set(Station.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown station fields for {key}: {sorted(unknown)}")
        registry[key] = Station(**values)
    return registry


def get_station(station_id: str, path: str | Path = DEFAULT_REGISTRY) -> Station:
    registry = load_station_registry(path)
    try:
        return registry[station_id]
    except KeyError as exc:
        raise KeyError(f"Unknown station '{station_id}'. Available: {sorted(registry)}") from exc
