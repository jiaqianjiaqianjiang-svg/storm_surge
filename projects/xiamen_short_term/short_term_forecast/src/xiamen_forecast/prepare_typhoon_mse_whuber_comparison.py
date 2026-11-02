"""合并已审核的四场1997台风，为MSE/WHuber历史回算建立只读清单。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
RESULTS_ROOT = REPO_ROOT / "reports" / "experiment_results"
DEFAULT_SOURCES = (
    RESULTS_ROOT / "typhoon_events_1997_seed42_compact",
    RESULTS_ROOT / "typhoon_events_1997_amber_victor_seed42",
)
DEFAULT_OUTPUT = RESULTS_ROOT / "typhoon_events_1997_mse_whuber_comparison_seed42"
EXPECTED_EVENT_IDS = (
    "beihai_1997232N17115",
    "lianyungang_1997217N06168",
    "lianyungang_1997210N15120",
    "xiamen_1997232N14136",
)
PROTOCOL_FIELDS = (
    "best_track_sha256",
    "year",
    "seed",
    "fixed_origin_offsets_hours",
    "fixed_horizons_hours",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-dirs", nargs=2, type=Path, default=DEFAULT_SOURCES)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def _load_source(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    events_file = path / "confirmed_event_list.csv"
    protocol_file = path / "source_and_protocol.json"
    missing = [str(item) for item in (events_file, protocol_file) if not item.is_file()]
    if missing:
        raise FileNotFoundError("Missing reviewed source files: " + "; ".join(missing))
    events = pd.read_csv(events_file, encoding="utf-8-sig", dtype={"storm_sid": str})
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError(f"Protocol must be a JSON object: {protocol_file}")
    return events, protocol


def _check_protocols(protocols: list[dict[str, Any]]) -> None:
    reference = protocols[0]
    for field in PROTOCOL_FIELDS:
        if field not in reference:
            raise ValueError(f"First source protocol is missing {field}")
        for index, protocol in enumerate(protocols[1:], start=2):
            if protocol.get(field) != reference[field]:
                raise ValueError(
                    f"Source protocol mismatch for {field}: source 1 has "
                    f"{reference[field]!r}, source {index} has {protocol.get(field)!r}"
                )


def prepare(source_dirs: tuple[Path, Path], output_dir: Path) -> Path:
    """Create a new package without altering either reviewed source package."""

    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing comparison package: {output}")

    frames: list[pd.DataFrame] = []
    protocols: list[dict[str, Any]] = []
    resolved_sources: list[Path] = []
    for source_dir in source_dirs:
        source = source_dir.resolve()
        events, protocol = _load_source(source)
        approved = events.loc[
            events.review_decision.astype(str).str.lower().str.strip() == "yes"
        ].copy()
        approved["source_review_package"] = source.name
        frames.append(approved)
        protocols.append(protocol)
        resolved_sources.append(source)
    _check_protocols(protocols)

    merged = pd.concat(frames, ignore_index=True)
    if merged.event_id.duplicated().any():
        duplicates = merged.loc[merged.event_id.duplicated(), "event_id"].tolist()
        raise ValueError(f"Duplicate reviewed event IDs: {duplicates}")
    actual = set(merged.event_id.astype(str))
    expected = set(EXPECTED_EVENT_IDS)
    if actual != expected:
        raise ValueError(
            f"Reviewed event set mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    if merged.review_note.astype(str).str.strip().eq("").any():
        raise ValueError("Every reviewed event must retain a non-empty review_note")

    order = {event_id: index for index, event_id in enumerate(EXPECTED_EVENT_IDS)}
    merged["_event_order"] = merged.event_id.map(order)
    merged = merged.sort_values("_event_order").drop(columns="_event_order")

    reference = protocols[0]
    metadata = {
        **reference,
        "stations": list(dict.fromkeys(merged.station.astype(str))),
        "source_review_packages": [path.name for path in resolved_sources],
        "approved_event_ids": list(EXPECTED_EVENT_IDS),
        "comparison_models": [
            "cnn_gru",
            "cnn_gru_rollout6",
            "whuber_one_step",
            "whuber_rollout6",
        ],
        "purpose": "fixed MSE/MSE versus WHuber/WHuber 1997 historical-event comparison",
        "selection_warning": "Do not use these 1997 cases to retune loss parameters or select a new objective.",
    }

    output.mkdir(parents=True)
    merged.to_csv(output / "candidate_events.csv", index=False, encoding="utf-8-sig")
    (output / "source_and_protocol.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared reviewed comparison package: {output}")
    return output


def main() -> None:
    args = parser().parse_args()
    prepare(tuple(args.source_dirs), args.output_dir)


if __name__ == "__main__":
    main()
