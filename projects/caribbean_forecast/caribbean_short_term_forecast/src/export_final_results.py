"""Export a compact, Git-safe Prickly Bay final-result package."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--evaluation-year", type=int, default=2018)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rolling-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def copy_result_files(source: Path, destination: Path) -> list[str]:
    if not source.is_dir():
        raise FileNotFoundError(f"Required result directory is missing: {source}")
    copied: list[str] = []
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md", ".png"}:
            continue
        if "predictions" in path.name.lower():
            continue
        target = destination / path.name
        shutil.copy2(path, target)
        copied.append(target.name)
    return copied


def main() -> None:
    args = parse_args()
    experiments = MODULE_ROOT / "outputs" / "experiments" / args.station
    direct = experiments / f"direct_multimodel_{args.evaluation_year}_final_seed{args.seed}"
    rolling = args.rolling_dir or (
        experiments / f"rolling_72_with_rollout6_{args.evaluation_year}_seed{args.seed}"
    )
    output = args.output_dir or (
        REPOSITORY_ROOT / "reports" / "experiment_results" /
        f"{args.station}_short_term_{args.evaluation_year}_seed{args.seed}"
    )
    if output.exists():
        shutil.rmtree(output)
    direct_output = output / "direct_24h"
    rolling_output = output / "rolling_72h"
    direct_output.mkdir(parents=True)
    rolling_output.mkdir(parents=True)
    copied = {
        "direct_24h": copy_result_files(direct, direct_output),
        "rolling_72h": copy_result_files(rolling, rolling_output),
    }
    manifest = {
        "station": args.station,
        "evaluation_year": args.evaluation_year,
        "seed": args.seed,
        "training_years": [2011, 2016],
        "development_year": 2017,
        "independent_test_year": args.evaluation_year,
        "known_future_era5": True,
        "rolling_result_source": str(rolling),
        "contains_model_weights": False,
        "contains_raw_or_processed_data": False,
        "files": copied,
    }
    (output / "RESULT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Git-safe result package: {output}")
    print("Review it, then commit only this reports/experiment_results directory.")


if __name__ == "__main__":
    main()
