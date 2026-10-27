"""Freeze reviewed training-year typhoon exclusion windows; do not train models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .station_config import MODULE_ROOT, get_station_config


REPOSITORY = next(parent for parent in (MODULE_ROOT, *MODULE_ROOT.parents) if (parent / ".git").exists())
DEFAULT_CATALOG = REPOSITORY / "reports" / "experiment_results" / "china_three_station_event_distribution_v1" / "station_typhoon_event_catalog.csv"


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def verify_source_catalog(reviewed: pd.DataFrame, catalog: pd.DataFrame) -> None:
    originals = catalog.set_index("event_id")
    for _, row in reviewed.loc[reviewed.review_decision.astype(str).str.lower().str.strip() == "yes"].iterrows():
        if row.event_id not in originals.index:
            raise ValueError(f"Approved event not found in source catalog: {row.event_id}")
        original = originals.loc[row.event_id]
        for field in ("station", "storm_sid", "split", "window_start_utc", "window_end_exclusive_utc"):
            if str(getattr(row, field)) != str(original[field]):
                raise ValueError(f"Approved event {row.event_id} changed source {field}")


def make_exclusions(reviewed: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_id", "station", "storm_sid", "split", "window_start_utc",
        "window_end_exclusive_utc", "data_complete_same_split",
        "input_evaluation_cross_split", "review_decision", "review_note",
    }
    missing = required - set(reviewed.columns)
    if missing:
        raise ValueError(f"Reviewed event catalog is missing columns: {sorted(missing)}")
    approved = reviewed.loc[reviewed.review_decision.astype(str).str.strip().str.lower() == "yes"].copy()
    if approved.empty:
        raise ValueError("No manually reviewed training-year events with review_decision=yes")
    if approved.event_id.duplicated().any():
        raise ValueError("Duplicate approved event_id")
    rows = []
    for _, event in approved.iterrows():
        station = get_station_config(str(event.station))
        if event.split != "train" or not _is_true(event.data_complete_same_split) or _is_true(event.input_evaluation_cross_split):
            raise ValueError(f"Event is not complete and confined to training years: {event.event_id}")
        if not str(event.review_note).strip() or str(event.review_note).lower() == "nan":
            raise ValueError(f"Manual meteorology/surge attribution note required: {event.event_id}")
        start = pd.Timestamp(event.window_start_utc)
        end = pd.Timestamp(event.window_end_exclusive_utc)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if not start < end:
            raise ValueError(f"Invalid event window: {event.event_id}")
        excluded_start = start - pd.Timedelta(hours=station.input_steps)
        excluded_end = end + pd.Timedelta(hours=72)
        train_start = pd.Timestamp(station.train_start_year, 1, 1, tz="UTC")
        train_end = pd.Timestamp(station.train_end_year + 1, 1, 1, tz="UTC")
        if excluded_start < train_start or excluded_end > train_end:
            raise ValueError(f"History/event/72h buffer crosses the training split: {event.event_id}")
        rows.append({
            "event_id": event.event_id, "station": station.station_id,
            "storm_sid": event.storm_sid,
            "event_start_utc": start.isoformat(),
            "event_end_exclusive_utc": end.isoformat(),
            "exclude_start_utc": excluded_start.isoformat(),
            "exclude_end_exclusive_utc": excluded_end.isoformat(),
            "history_buffer_hours": station.input_steps,
            "forecast_buffer_hours": 72,
            "review_note": event.review_note,
        })
    result = pd.DataFrame(rows).sort_values(["station", "exclude_start_utc"])
    for station_id, group in result.groupby("station"):
        last_end = None
        for _, row in group.iterrows():
            start = pd.Timestamp(row.exclude_start_utc)
            if last_end is not None and start < last_end:
                raise ValueError(f"Overlapping holdout windows for {station_id}; review case independence")
            last_end = pd.Timestamp(row.exclude_end_exclusive_utc)
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-csv", type=Path, required=True)
    parser.add_argument("--catalog-csv", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if args.output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite fixed holdout plan: {args.output_csv}")
    reviewed = pd.read_csv(args.reviewed_csv, encoding="utf-8-sig")
    verify_source_catalog(reviewed, pd.read_csv(args.catalog_csv, encoding="utf-8-sig"))
    result = make_exclusions(reviewed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Frozen exclusion windows (no model training performed): {args.output_csv}")


if __name__ == "__main__":
    main()
