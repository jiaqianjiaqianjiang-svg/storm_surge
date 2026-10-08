"""Run the complete, resumable validation workflow for one configured station."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .station_config import MODULE_ROOT, STATIONS, get_station_config


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    command: tuple[str, ...]
    marker: Path | None = None
    required_json_keys: tuple[str, ...] = ()
    always_run: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", choices=tuple(STATIONS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run every step again even when its completion marker is valid.",
    )
    return parser.parse_args()


def require_passed_audit(station_id: str) -> dict[str, Any]:
    station = get_station_config(station_id)
    path = station.processed_root / "prepared_dataset_audit.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared-data audit is missing: {path}. Run preparation and audit first."
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read prepared-data audit: {path}") from exc
    if report.get("status") != "passed":
        errors = report.get("errors") or ["audit status is not 'passed'"]
        raise ValueError(
            f"Prepared-data audit failed for {station.name}: " + "; ".join(errors)
        )
    return report


def _json_contains(path: Path, keys: tuple[str, ...]) -> bool:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


def step_is_complete(step: WorkflowStep) -> bool:
    if step.always_run or step.marker is None or not step.marker.is_file():
        return False
    if step.required_json_keys:
        return _json_contains(step.marker, step.required_json_keys)
    return True


def build_validation_steps(
    station_id: str,
    seed: int = 42,
    device: str = "cuda",
    language: str = "en",
    epochs: int = 50,
    patience: int = 8,
    num_workers: int = 0,
) -> list[WorkflowStep]:
    station = get_station_config(station_id)
    python = sys.executable
    validation_year = str(station.validation_year)
    formal = station.model_root / f"formal_seed{seed}"
    comparison = station.experiment_root / "model_comparison"
    rolling = station.experiment_root / f"rolling_{validation_year}_seed{seed}"
    figures = (
        MODULE_ROOT
        / "outputs"
        / "journal_figures"
        / f"{station.station_id}_{validation_year}_seed{seed}"
    )
    repository = next(
        parent
        for parent in (MODULE_ROOT, *MODULE_ROOT.parents)
        if (parent / ".git").exists()
    )
    result_package = (
        repository
        / "reports"
        / "experiment_results"
        / f"{station.station_id}_short_term_{validation_year}_seed{seed}"
    )

    steps: list[WorkflowStep] = [
        WorkflowStep(
            "Persistence and Ridge baselines",
            (
                python,
                "-m",
                "src.xiamen_forecast.evaluate_baselines",
                "--station",
                station.station_id,
                "--validation-only",
            ),
            station.model_root / "baselines" / "baseline_metrics.json",
            ("metrics", "validation"),
        )
    ]
    model_batches = {
        "surge_mlp": 256,
        "era5_cnn": 256,
        "cnn": 256,
        "cnn_gru": 32,
    }
    for model, batch_size in model_batches.items():
        steps.append(
            WorkflowStep(
                f"Train {model}",
                (
                    python,
                    "-m",
                    "src.xiamen_forecast.train_xiamen",
                    "--station",
                    station.station_id,
                    "--model-type",
                    model,
                    "--validation-only",
                    "--device",
                    device,
                    "--batch-size",
                    str(batch_size),
                    "--epochs",
                    str(epochs),
                    "--patience",
                    str(patience),
                    "--seed",
                    str(seed),
                    "--num-workers",
                    str(num_workers),
                ),
                formal / model / "metrics.json",
                ("validation",),
            )
        )
    steps.extend(
        [
            WorkflowStep(
                "Compare one-step validation models",
                (
                    python,
                    "-m",
                    "src.xiamen_forecast.compare_models",
                    "--station",
                    station.station_id,
                    "--split",
                    "validation",
                    "--seed",
                    str(seed),
                ),
                comparison / "validation_model_metrics.csv",
                always_run=True,
            ),
            WorkflowStep(
                "Fine-tune CNN-GRU with rollout-6",
                (
                    python,
                    "-m",
                    "src.xiamen_forecast.train_rollout_temporal",
                    "--station",
                    station.station_id,
                    "--model-type",
                    "cnn_gru",
                    "--rollout-steps",
                    "6",
                    "--device",
                    device,
                    "--seed",
                    str(seed),
                    "--num-workers",
                    str(num_workers),
                ),
                formal / "cnn_gru_rollout6" / "training_metadata.json",
            ),
            WorkflowStep(
                "Run 72-hour validation rolling diagnostics",
                (
                    python,
                    "-m",
                    "src.xiamen_forecast.rolling_diagnostics",
                    "--station",
                    station.station_id,
                    "--evaluation-year",
                    validation_year,
                    "--split",
                    "validation",
                    "--models",
                    "surge_mlp",
                    "era5_cnn",
                    "cnn",
                    "cnn_gru",
                    "--device",
                    device,
                    "--seed",
                    str(seed),
                    "--rollout-checkpoint",
                    str(formal / "cnn_gru_rollout6" / "best_model.pth"),
                ),
                rolling / "rolling_rmse_table.csv",
            ),
            WorkflowStep(
                "Create validation figures",
                (
                    python,
                    "-m",
                    "src.short_term_forecast.journal_figures.make_xiamen_journal_figures",
                    "--station",
                    station.station_id,
                    "--validation-year",
                    validation_year,
                    "--split",
                    "validation",
                    "--seed",
                    str(seed),
                    "--language",
                    language,
                ),
                figures / "figure_manifest.csv",
            ),
            WorkflowStep(
                "Export Git-safe validation result package",
                (
                    python,
                    "-m",
                    "src.xiamen_forecast.export_final_results",
                    "--station",
                    station.station_id,
                    "--evaluation-year",
                    validation_year,
                    "--split",
                    "validation",
                    "--seed",
                    str(seed),
                ),
                result_package / "RESULTS_SUMMARY.md",
            ),
        ]
    )
    return steps


def main() -> None:
    args = parse_args()
    station = get_station_config(args.station)
    require_passed_audit(station.station_id)
    steps = build_validation_steps(
        station.station_id,
        seed=args.seed,
        device=args.device,
        language=args.language,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
    )
    print(
        f"Audit passed. Starting {station.name} {station.validation_year} "
        f"validation workflow ({len(steps)} steps).",
        flush=True,
    )
    for index, step in enumerate(steps, start=1):
        prefix = f"[{index}/{len(steps)}]"
        if not args.force and step_is_complete(step):
            print(f"{prefix} SKIP {step.name}: {step.marker}", flush=True)
            continue
        print(f"{prefix} START {step.name}", flush=True)
        print("  " + subprocess.list2cmdline(step.command), flush=True)
        subprocess.run(step.command, cwd=MODULE_ROOT, check=True)
        if step.marker is not None and not step.marker.is_file():
            raise FileNotFoundError(
                f"Step finished but its expected output is missing: {step.marker}"
            )
        print(f"{prefix} DONE {step.name}", flush=True)

    repository = next(
        parent
        for parent in (MODULE_ROOT, *MODULE_ROOT.parents)
        if (parent / ".git").exists()
    )
    relative_package = (
        Path("reports")
        / "experiment_results"
        / f"{station.station_id}_short_term_{station.validation_year}_seed{args.seed}"
    )
    print(f"\nWorkflow completed. Result package: {repository / relative_package}")
    print("Upload only the compact package:")
    print(f"  git add {relative_package}")
    print(
        f'  git commit -m "Archive {station.name} '
        f'{station.validation_year} validation results"'
    )
    print("  git push origin codex/china-multistation")


if __name__ == "__main__":
    main()
