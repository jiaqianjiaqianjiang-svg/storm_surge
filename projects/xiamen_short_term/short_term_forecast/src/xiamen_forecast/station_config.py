"""Central configuration for Chinese hourly storm-surge stations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StationConfig:
    station_id: str
    name: str
    latitude: float
    longitude: float
    gesla_file: Path
    era5_dir: Path
    era5_filenames: dict[str, str]
    data_start_year: int
    data_end_year: int
    train_start_year: int
    train_end_year: int
    validation_year: int
    test_year: int
    input_steps: int = 24
    grid_size: int = 40
    region_size_degrees: float = 10.0

    @property
    def era5_files(self) -> dict[str, Path]:
        return {
            variable: self.era5_dir / filename
            for variable, filename in self.era5_filenames.items()
        }

    @property
    def processed_root(self) -> Path:
        return MODULE_ROOT / "outputs" / "processed" / self.station_id

    @property
    def dataset_dir(self) -> Path:
        return self.processed_root / "aligned_dataset"

    @property
    def model_root(self) -> Path:
        return MODULE_ROOT / "models" / self.station_id

    @property
    def experiment_root(self) -> Path:
        return MODULE_ROOT / "outputs" / "experiments" / self.station_id

    def as_metadata(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "station_name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "gesla_file": str(self.gesla_file),
            "era5_dir": str(self.era5_dir),
            "era5_files": {
                name: str(path) for name, path in self.era5_files.items()
            },
            "data_years": [self.data_start_year, self.data_end_year],
            "training_years": [self.train_start_year, self.train_end_year],
            "validation_year": self.validation_year,
            "test_year": self.test_year,
            "input_steps": self.input_steps,
            "grid_size": self.grid_size,
            "variable_order": ["U10", "V10", "MSL"],
        }


def _era5_names(prefix: str, start_year: int, end_year: int) -> dict[str, str]:
    span = f"{start_year}_{end_year}"
    return {
        "U10": f"{prefix}_10u_{span}.nc",
        "V10": f"{prefix}_v10_{span}.nc",
        "MSL": f"{prefix}_slp_{span}.nc",
    }


STATIONS: dict[str, StationConfig] = {
    "xiamen": StationConfig(
        station_id="xiamen",
        name="Xiamen",
        latitude=24.450,
        longitude=118.067,
        gesla_file=Path(r"F:\GESLA\GESLA3\xiamen-376a-chn-uhslc"),
        era5_dir=Path(r"F:\ERA5-NEW\Xiamen"),
        era5_filenames=_era5_names("xiamen", 1970, 1997),
        data_start_year=1970,
        data_end_year=1997,
        train_start_year=1970,
        train_end_year=1995,
        validation_year=1996,
        test_year=1997,
    ),
    "lianyungang": StationConfig(
        station_id="lianyungang",
        name="Lianyungang",
        latitude=34.750,
        longitude=119.417,
        gesla_file=Path(r"F:\GESLA\GESLA3\lianyungang-639a-chn-uhslc"),
        era5_dir=Path(r"F:\ERA5-NEW\Lianyungang"),
        era5_filenames=_era5_names("lianyungang", 1975, 1997),
        data_start_year=1975,
        data_end_year=1997,
        train_start_year=1975,
        train_end_year=1995,
        validation_year=1996,
        test_year=1997,
    ),
    "beihai": StationConfig(
        station_id="beihai",
        name="Beihai",
        latitude=21.483,
        longitude=109.083,
        gesla_file=Path(r"F:\GESLA\GESLA3\beihai-636a-chn-uhslc"),
        era5_dir=Path(r"F:\ERA5-NEW\Beihai"),
        era5_filenames=_era5_names("beihai", 1975, 1997),
        data_start_year=1975,
        data_end_year=1997,
        train_start_year=1975,
        train_end_year=1995,
        validation_year=1996,
        test_year=1997,
    ),
}


def get_station_config(station: str) -> StationConfig:
    key = station.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"lian_yun_gang": "lianyungang", "bei_hai": "beihai"}
    key = aliases.get(key, key)
    if key not in STATIONS:
        raise ValueError(
            f"Unknown station '{station}'. Available stations: {', '.join(STATIONS)}"
        )
    return STATIONS[key]


def configured_year(value: int | None, station: StationConfig, field: str) -> int:
    return int(getattr(station, field) if value is None else value)


def apply_station_defaults(
    arguments: Any,
    field_map: dict[str, str],
) -> StationConfig:
    """Fill only omitted CLI values, preserving explicit compatibility overrides."""
    station = get_station_config(arguments.station)
    for argument_name, config_name in field_map.items():
        if getattr(arguments, argument_name, None) is None:
            setattr(arguments, argument_name, getattr(station, config_name))
    return station


def validate_dataset_identity(dataset_dir: str | Path, station_id: str) -> dict[str, Any]:
    """Require prepared metadata and reject accidental cross-station reads."""
    station = get_station_config(station_id)
    root = Path(dataset_dir)
    metadata_path = root / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Prepared dataset metadata is missing: {metadata_path}. "
            f"Run preparation and audit for station '{station.station_id}' first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = metadata.get("station_id")
    if actual != station.station_id:
        raise ValueError(
            f"Prepared dataset belongs to station '{actual}', not "
            f"'{station.station_id}': {root}"
        )
    return metadata


def validate_output_location(
    output_path: str | Path,
    allowed_root: str | Path,
    label: str,
) -> Path:
    """Reject CLI output paths outside the intended station-owned directory."""
    output = Path(output_path).resolve(strict=False)
    root = Path(allowed_root).resolve(strict=False)
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{label} must remain inside {root}; received {output}"
        ) from exc
    return output
