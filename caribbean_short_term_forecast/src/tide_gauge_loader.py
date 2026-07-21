"""Unified loader for GESLA 4.0 and PSMSL/NOC tide-gauge text files."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


DATE_TOKEN = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")
ALIASES = {
    "datetime": ("datetime", "date_time", "timestamp", "date", "time"),
    "water_level": (
        "water_level", "waterlevel", "sea_level", "sealevel", "level", "height", "value",
    ),
    "qc_flag": ("qc_flag", "qc", "quality_flag", "quality"),
    "use_flag": ("use_flag", "use", "valid", "valid_flag"),
    "sensor": ("sensor", "channel", "sensor_id", "instrument"),
    "tide_estimate": ("tide_estimate", "tide", "predicted_tide"),
}
OUTPUT_COLUMNS = [
    "datetime", "water_level", "qc_flag", "use_flag", "source", "sensor", "station_id",
    "tide_estimate",
]


def _normalise_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _find_column(columns: list[object], target: str) -> object | None:
    normalised = {_normalise_name(column): column for column in columns}
    return next((normalised[name] for name in ALIASES[target] if name in normalised), None)


def _read_gesla_rows(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = line.replace(",", " ").split()
            if len(tokens) < 3 or not DATE_TOKEN.match(tokens[0]):
                continue
            time_token = tokens[1] if ":" in tokens[1] else "00:00"
            try:
                value = float(tokens[2])
            except ValueError:
                continue
            rows.append(
                {
                    "datetime": f"{tokens[0]} {time_token}",
                    "water_level": value,
                    "qc_flag": tokens[3] if len(tokens) > 3 else np.nan,
                    "use_flag": tokens[4] if len(tokens) > 4 else np.nan,
                    "sensor": tokens[5] if len(tokens) > 5 else "default",
                    "tide_estimate": tokens[6] if len(tokens) > 6 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _read_table(path: Path) -> pd.DataFrame:
    attempts = ({"sep": None, "engine": "python"}, {"sep": r"\s+", "engine": "python"})
    errors: list[str] = []
    for kwargs in attempts:
        try:
            frame = pd.read_csv(path, comment="#", **kwargs)
            if len(frame.columns) >= 2:
                return frame
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            errors.append(str(exc))
    raise ValueError(f"Could not parse tide-gauge file {path}: {'; '.join(errors)}")


def _canonicalise_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(frame.columns)
    date_column = _find_column(columns, "datetime")
    value_column = _find_column(columns, "water_level")
    if date_column is None or value_column is None:
        raise ValueError(
            "Could not identify datetime and water-level columns. "
            f"Available columns: {list(map(str, columns))}"
        )
    normalised = {_normalise_name(column): column for column in columns}
    datetime_values = frame[date_column]
    if _normalise_name(date_column) == "date" and "time" in normalised and normalised["time"] != date_column:
        datetime_values = frame[date_column].astype(str).str.strip() + " " + frame[normalised["time"]].astype(str).str.strip()
    result = pd.DataFrame({"datetime": datetime_values, "water_level": frame[value_column]})
    for target in ("qc_flag", "use_flag", "sensor", "tide_estimate"):
        source = _find_column(columns, target)
        result[target] = frame[source] if source is not None else np.nan
    return result


def load_tide_gauge(
    path: str | Path,
    station_id: str,
    source: str = "auto",
    water_level_unit: str = "m",
) -> pd.DataFrame:
    """Load a station file without applying destructive quality filters.

    GESLA headerless records are recognised by their leading date token. CSV,
    tab-separated and whitespace-delimited tables are recognised by aliases.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tide-gauge file not found: {path}")
    source_name = source if source != "auto" else ("GESLA 4.0" if "gesla" in str(path).lower() else "PSMSL/NOC")
    frame = _read_gesla_rows(path) if "gesla" in source_name.lower() else pd.DataFrame()
    if frame.empty:
        try:
            frame = _canonicalise_table(_read_table(path))
        except ValueError:
            frame = _read_gesla_rows(path)
    if frame.empty:
        raise ValueError(f"No tide-gauge observations recognised in {path}")

    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["water_level"] = pd.to_numeric(frame["water_level"], errors="coerce")
    frame["tide_estimate"] = pd.to_numeric(frame["tide_estimate"], errors="coerce")
    unit = water_level_unit.strip().lower()
    factor = {"m": 1.0, "metre": 1.0, "meter": 1.0, "cm": 0.01, "mm": 0.001}.get(unit)
    if factor is None:
        raise ValueError(f"Unsupported water-level unit '{water_level_unit}'; expected m, cm, or mm")
    frame["water_level"] *= factor
    frame["tide_estimate"] *= factor
    frame["source"] = source_name
    frame["sensor"] = frame["sensor"].fillna("default").astype(str)
    frame["station_id"] = station_id
    return frame.reindex(columns=OUTPUT_COLUMNS)
