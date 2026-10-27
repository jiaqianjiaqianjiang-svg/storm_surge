"""Audit and run the fixed three-station extreme-loss validation suite.

The default action is a real execution. Use ``--dry-run`` first on the laboratory
computer. Existing compatible artifacts are skipped; incomplete or conflicting
directories are never overwritten without an explicit recovery decision.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

from .extreme_loss_suite import (
    EXTREME_ALPHA,
    EXTREME_BETA,
    FORMAL_SEED,
    HUBER_DELTA_CM,
    LOSS_SCHEMES,
    ONE_STEP_BATCH_SIZE,
    ONE_STEP_EPOCHS,
    ONE_STEP_LEARNING_RATE,
    ONE_STEP_LOSSES,
    ONE_STEP_PATIENCE,
    ROLLOUT_BATCH_SIZE,
    ROLLOUT_EMBEDDING_BATCH_SIZE,
    ROLLOUT_EPOCHS,
    ROLLOUT_LEARNING_RATE,
    ROLLOUT_PATIENCE,
    ROLLOUT_STEPS,
    ROLLOUT_TEACHER_FORCING_EPOCHS,
    SCHEMES_BY_ID,
    SUITE_STATIONS,
    LossScheme,
    inspect_one_step_artifact,
    inspect_rollout_artifact,
    one_step_path,
    reusable,
    rollout_path,
)
from .station_config import MODULE_ROOT, StationConfig, get_station_config


@dataclass(frozen=True)
class Task:
    task_id: str
    description: str
    command: tuple[str, ...]
    output_dir: Path
    inspect: Callable[[], dict[str, object]]


def repository_root() -> Path:
    return next(
        parent for parent in (MODULE_ROOT, *MODULE_ROOT.parents)
        if (parent / ".git").exists()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stations", nargs="+", choices=SUITE_STATIONS,
        default=list(SUITE_STATIONS),
    )
    parser.add_argument(
        "--schemes", nargs="+", choices=tuple(SCHEMES_BY_ID),
        default=list(SCHEMES_BY_ID),
    )
    parser.add_argument(
        "--stage", choices=("status", "train", "evaluate", "summarize", "all"), default="all"
    )
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--accept-legacy-metadata",
        action="store_true",
        help=(
            "Reuse a complete older artifact whose essential identity is valid but "
            "whose metadata predates the full formal-parameter record."
        ),
    )
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help=(
            "Explicitly restart a task inside an existing incomplete directory. "
            "Configuration-mismatched directories remain blocked."
        ),
    )
    parser.add_argument("--status-csv", type=Path)
    return parser.parse_args()


def _join(values: object) -> str:
    if not values:
        return ""
    return "; ".join(str(item) for item in values)


def _csv_contains(path: Path, column: str, value: str) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return any(row.get(column) == value for row in csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error):
        return False


def compact_evidence(station_id: str, scheme: LossScheme) -> tuple[bool, bool, str]:
    """Report Git-safe evidence without treating it as a local checkpoint."""

    reports = repository_root() / "reports" / "experiment_results"
    if station_id == "lianyungang":
        one = reports / "lianyungang_extreme_loss_ablation_1996_seed42"
        rolling = reports / "lianyungang_extreme_rollout_validation_1996_seed42"
        one_ok = _csv_contains(
            one / "validation_extreme_loss_summary.csv",
            "experiment",
            scheme.one_step_loss,
        )
        rolling_ok = _csv_contains(
            rolling / "rolling_metrics_long.csv", "model", scheme.rolling_label
            if scheme.scheme_id != "mse/mse" else "cnn_gru_rollout6"
        )
        return one_ok, rolling_ok, "archived Lianyungang extreme-loss package"
    core = reports / f"{station_id}_short_term_1996_seed42"
    one_ok = scheme.scheme_id == "mse/mse" and _csv_contains(
        core / "validation_model_metrics.csv", "model", "cnn_gru"
    )
    rolling_ok = scheme.scheme_id == "mse/mse" and _csv_contains(
        core / "rolling_metrics_long.csv", "model", "cnn_gru_rollout6"
    )
    return one_ok, rolling_ok, "archived core validation package"


def build_status_rows(
    stations: list[str], schemes: list[LossScheme], seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for station_id in stations:
        station = get_station_config(station_id)
        for scheme in schemes:
            one = inspect_one_step_artifact(station, scheme.one_step_loss, seed)
            rollout = inspect_rollout_artifact(station, scheme, seed)
            compact_one, compact_rollout, evidence_note = compact_evidence(
                station_id, scheme
            )
            rows.append(
                {
                    "station": station_id,
                    "scheme": scheme.scheme_id,
                    "one_step_loss": scheme.one_step_loss,
                    "rollout_loss": scheme.rollout_loss,
                    "one_step_directory": scheme.one_step_directory,
                    "one_step_checkpoint_on_this_computer": (
                        Path(str(one["path"])) / "best_model.pth"
                    ).is_file(),
                    "one_step_status_on_this_computer": one["status"],
                    "one_step_missing_files": _join(one["missing_files"]),
                    "one_step_metadata_issues": _join(
                        [*one["mismatches"], *one["missing_metadata"]]
                    ),
                    "one_step_1996_compact_evidence": compact_one,
                    "rollout_directory": scheme.rollout_directory,
                    "rollout_checkpoint_on_this_computer": (
                        Path(str(rollout["path"])) / "best_model.pth"
                    ).is_file(),
                    "rollout_status_on_this_computer": rollout["status"],
                    "rollout_missing_files": _join(rollout["missing_files"]),
                    "rollout_metadata_issues": _join(
                        [*rollout["mismatches"], *rollout["missing_metadata"]]
                    ),
                    "rollout_1996_compact_evidence": compact_rollout,
                    "compact_evidence_note": evidence_note,
                    "laboratory_ignored_checkpoint_status": "unknown until laboratory dry-run",
                    "next_action": (
                        "inspect laboratory metadata; reuse when compatible"
                        if compact_one or compact_rollout
                        else "train on laboratory computer after dry-run"
                    ),
                }
            )
    return rows


def write_status(rows: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def one_step_task(
    station: StationConfig,
    loss_type: str,
    seed: int,
    device: str,
    workers: int,
) -> Task:
    output = one_step_path(station, loss_type, seed)
    command = (
        sys.executable, "-m", "src.xiamen_forecast.train_xiamen",
        "--station", station.station_id,
        "--model-type", "cnn_gru",
        "--loss-type", loss_type,
        "--validation-only",
        "--split-mode", "years",
        "--train-start-year", str(station.train_start_year),
        "--train-end-year", str(station.train_end_year),
        "--validation-year", str(station.validation_year),
        "--test-year", str(station.test_year),
        "--input-steps", "24",
        "--epochs", str(ONE_STEP_EPOCHS),
        "--patience", str(ONE_STEP_PATIENCE),
        "--batch-size", str(ONE_STEP_BATCH_SIZE),
        "--lr", str(ONE_STEP_LEARNING_RATE),
        "--seed", str(seed),
        "--extreme-alpha", str(EXTREME_ALPHA),
        "--extreme-beta", str(EXTREME_BETA),
        "--huber-delta-cm", str(HUBER_DELTA_CM),
        "--device", device,
        "--num-workers", str(workers),
        "--dataset-path", str(station.dataset_dir),
        "--baseline-dir", str(station.model_root / "baselines"),
        "--output-dir", str(output),
    )
    return Task(
        f"{station.station_id}:one-step:{loss_type}",
        f"{station.name} CNN-GRU one-step {loss_type.upper()}",
        command,
        output,
        lambda: inspect_one_step_artifact(station, loss_type, seed),
    )


def rollout_task(
    station: StationConfig,
    scheme: LossScheme,
    seed: int,
    device: str,
    workers: int,
) -> Task:
    output = rollout_path(station, scheme, seed)
    base = one_step_path(station, scheme.one_step_loss, seed) / "best_model.pth"
    command = (
        sys.executable, "-m", "src.xiamen_forecast.train_rollout_temporal",
        "--station", station.station_id,
        "--model-type", "cnn_gru",
        "--loss-type", scheme.rollout_loss,
        "--train-start-year", str(station.train_start_year),
        "--train-end-year", str(station.train_end_year),
        "--validation-year", str(station.validation_year),
        "--rollout-steps", str(ROLLOUT_STEPS),
        "--epochs", str(ROLLOUT_EPOCHS),
        "--patience", str(ROLLOUT_PATIENCE),
        "--teacher-forcing-epochs", str(ROLLOUT_TEACHER_FORCING_EPOCHS),
        "--batch-size", str(ROLLOUT_BATCH_SIZE),
        "--embedding-batch-size", str(ROLLOUT_EMBEDDING_BATCH_SIZE),
        "--learning-rate", str(ROLLOUT_LEARNING_RATE),
        "--seed", str(seed),
        "--extreme-alpha", str(EXTREME_ALPHA),
        "--extreme-beta", str(EXTREME_BETA),
        "--huber-delta-cm", str(HUBER_DELTA_CM),
        "--device", device,
        "--num-workers", str(workers),
        "--dataset-path", str(station.dataset_dir),
        "--base-checkpoint", str(base),
        "--output-dir", str(output),
    )
    return Task(
        f"{station.station_id}:rollout:{scheme.scheme_id}",
        f"{station.name} rollout-6 {scheme.display_name}",
        command,
        output,
        lambda: inspect_rollout_artifact(station, scheme, seed),
    )


def command_text(command: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(command)


def inspect_output_files(root: Path, required: tuple[str, ...]) -> dict[str, object]:
    missing = [name for name in required if not (root / name).is_file()]
    return {
        "status": (
            "compatible" if not missing else ("incomplete" if root.exists() else "absent")
        ),
        "path": str(root),
        "missing_files": missing,
        "mismatches": [],
        "missing_metadata": [],
    }


def run_task(
    task: Task,
    dry_run: bool,
    accept_legacy: bool,
    restart_incomplete: bool,
    log_root: Path,
    state: dict[str, object],
) -> None:
    inspection = task.inspect()
    status = str(inspection["status"])
    print(f"[{status}] {task.description}", flush=True)
    if reusable(status, accept_legacy):
        print(f"  skip: {task.output_dir}", flush=True)
        state[task.task_id] = {"status": f"reused_{status}", "path": str(task.output_dir)}
        return
    if status == "legacy_metadata":
        if dry_run:
            print("  blocked: review metadata; optionally use --accept-legacy-metadata", flush=True)
            state[task.task_id] = {
                "status": "blocked_legacy_metadata", "path": str(task.output_dir)
            }
            return
        raise RuntimeError(
            f"{task.task_id} is complete but its formal parameters are not fully "
            "recorded. Review the status CSV, then rerun with "
            "--accept-legacy-metadata only if the laboratory record confirms it."
        )
    if status in {"configuration_mismatch", "configuration_invalid"}:
        if dry_run:
            print(f"  blocked: {status}; issues={inspection['mismatches']}", flush=True)
            state[task.task_id] = {"status": f"blocked_{status}", "path": str(task.output_dir)}
            return
        raise RuntimeError(
            f"Refusing to overwrite {task.output_dir}: {status}; "
            f"issues={inspection['mismatches']}. Move it to an archival name or "
            "use a clean experiment directory after review."
        )
    if status == "incomplete" and not restart_incomplete:
        if dry_run:
            print("  blocked: incomplete directory requires an explicit recovery decision", flush=True)
            state[task.task_id] = {
                "status": "blocked_incomplete", "path": str(task.output_dir)
            }
            return
        raise RuntimeError(
            f"Incomplete directory exists: {task.output_dir}. Inspect its log and "
            "files; pass --restart-incomplete only to explicitly restart this task."
        )
    print("  " + command_text(task.command), flush=True)
    if dry_run:
        state[task.task_id] = {"status": "planned", "path": str(task.output_dir)}
        return
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{task.task_id.replace(':', '_').replace('/', '_')}.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {command_text(task.command)}\n")
        process = subprocess.Popen(
            task.command,
            cwd=MODULE_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        state[task.task_id] = {
            "status": "failed", "return_code": return_code, "log": str(log_path)
        }
        raise RuntimeError(f"Task failed ({return_code}); see {log_path}")
    verified = task.inspect()
    if verified["status"] != "compatible":
        raise RuntimeError(
            f"Task finished but artifact verification returned {verified['status']}: "
            f"{task.output_dir}"
        )
    state[task.task_id] = {
        "status": "completed", "path": str(task.output_dir), "log": str(log_path)
    }


def evaluation_commands(station: StationConfig, seed: int, device: str) -> list[Task]:
    formal = station.model_root / f"formal_seed{seed}"
    one_output = station.experiment_root / f"extreme_loss_suite_one_step_{station.validation_year}_seed{seed}"
    rolling_output = station.experiment_root / f"extreme_loss_suite_rolling_{station.validation_year}_seed{seed}"
    compare = Task(
        f"{station.station_id}:evaluate:one-step",
        f"{station.name} common-timestamp one-step comparison",
        (
            sys.executable, "-m", "src.xiamen_forecast.compare_extreme_losses",
            "--station", station.station_id,
            "--validation-year", str(station.validation_year),
            "--seed", str(seed),
            "--model-root", str(formal),
            "--output-dir", str(one_output),
        ),
        one_output,
        lambda: inspect_output_files(
            one_output,
            (
                "validation_extreme_loss_summary.csv",
                "validation_positive_peak_events.csv",
                "validation_extreme_loss_details.json",
                "experiment_metadata.json",
                "validation_extreme_loss_comparison.png",
            ),
        ),
    )
    extras: list[str] = []
    for scheme in LOSS_SCHEMES[1:]:
        extras.extend(
            ["--extra-checkpoint", f"{scheme.rolling_label}={rollout_path(station, scheme, seed) / 'best_model.pth'}"]
        )
    rolling = Task(
        f"{station.station_id}:evaluate:rolling",
        f"{station.name} common-origin rollout comparison",
        (
            sys.executable, "-m", "src.xiamen_forecast.rolling_diagnostics",
            "--station", station.station_id,
            "--train-start-year", str(station.train_start_year),
            "--evaluation-year", str(station.validation_year),
            "--split", "validation",
            "--models", "cnn_gru",
            "--input-steps", "24",
            "--max-lead", "72",
            "--seed", str(seed),
            "--device", device,
            "--dataset-path", str(station.dataset_dir),
            "--checkpoint-root", str(formal),
            "--ridge-path", str(station.model_root / "baselines" / "ridge_pipeline.joblib"),
            "--rollout-checkpoint", str(rollout_path(station, LOSS_SCHEMES[0], seed) / "best_model.pth"),
            *extras,
            "--output-dir", str(rolling_output),
        ),
        rolling_output,
        lambda: inspect_output_files(
            rolling_output,
            (
                "rolling_metrics_long.csv",
                "rolling_rmse_table.csv",
                "rolling_predictions_selected_leads.csv",
                "rmse_vs_lead.png",
                "experiment_metadata.json",
                "rolling_diagnostic_report.md",
            ),
        ),
    )
    return [compare, rolling]


def summary_task(seed: int) -> Task:
    output = (
        repository_root() / "reports" / "experiment_results"
        / f"three_station_extreme_loss_validation_1996_seed{seed}"
    )
    return Task(
        "three-station:summarize",
        "Three-station unified extreme-loss report",
        (
            sys.executable, "-m",
            "src.xiamen_forecast.summarize_three_station_extreme_losses",
            "--stations", *SUITE_STATIONS,
            "--seed", str(seed),
            "--output-dir", str(output),
        ),
        output,
        lambda: inspect_output_files(
            output,
            (
                "three_station_extreme_loss_validation.csv",
                "three_station_extreme_rollout_validation.csv",
                "three_station_extreme_loss_comparison.png",
                "three_station_extreme_rollout_comparison.png",
                "three_station_extreme_loss_report.md",
                "experiment_metadata.json",
            ),
        ),
    )


def require_all_artifacts(
    station: StationConfig, seed: int, accept_legacy: bool
) -> None:
    problems: list[str] = []
    for loss_type in ONE_STEP_LOSSES:
        status = inspect_one_step_artifact(station, loss_type, seed)["status"]
        if not reusable(str(status), accept_legacy):
            problems.append(f"one-step {loss_type}: {status}")
    for scheme in LOSS_SCHEMES:
        status = inspect_rollout_artifact(station, scheme, seed)["status"]
        if not reusable(str(status), accept_legacy):
            problems.append(f"rollout {scheme.scheme_id}: {status}")
    if problems:
        raise RuntimeError(
            f"Cannot evaluate {station.station_id}; required artifacts are not ready: "
            + "; ".join(problems)
        )


def save_state(path: Path, state: dict[str, object], dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "tasks": state,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.seed != FORMAL_SEED:
        raise ValueError("The formal suite is fixed to seed 42")
    selected_schemes = [SCHEMES_BY_ID[item] for item in dict.fromkeys(args.schemes)]
    selected_stations = list(dict.fromkeys(args.stations))
    suite_root = MODULE_ROOT / "outputs" / "experiments" / f"three_station_extreme_loss_suite_seed{args.seed}"
    status_csv = args.status_csv or suite_root / "three_station_extreme_experiment_status.csv"
    rows = build_status_rows(selected_stations, selected_schemes, args.seed)
    write_status(rows, status_csv)
    print(f"Status table: {status_csv}", flush=True)
    if args.stage == "status":
        return

    state_path = suite_root / ("dry_run_state.json" if args.dry_run else "suite_state.json")
    state: dict[str, object] = {}
    log_root = suite_root / "logs"
    try:
        if args.stage in {"train", "all"}:
            for station_id in selected_stations:
                station = get_station_config(station_id)
                unique_losses = list(dict.fromkeys(scheme.one_step_loss for scheme in selected_schemes))
                for loss_type in unique_losses:
                    task = one_step_task(
                        station, loss_type, args.seed, args.device, args.num_workers
                    )
                    run_task(
                        task, args.dry_run, args.accept_legacy_metadata,
                        args.restart_incomplete, log_root, state,
                    )
                for scheme in selected_schemes:
                    base_status = inspect_one_step_artifact(
                        station, scheme.one_step_loss, args.seed
                    )["status"]
                    if not args.dry_run and not reusable(
                        str(base_status), args.accept_legacy_metadata
                    ):
                        raise RuntimeError(
                            f"Dependency is not reusable for {scheme.scheme_id}: {base_status}"
                        )
                    task = rollout_task(
                        station, scheme, args.seed, args.device, args.num_workers
                    )
                    run_task(
                        task, args.dry_run, args.accept_legacy_metadata,
                        args.restart_incomplete, log_root, state,
                    )

        if args.stage in {"evaluate", "all"}:
            if set(selected_schemes) != set(LOSS_SCHEMES):
                raise ValueError(
                    "Unified evaluation requires all five fixed schemes; omit "
                    "--schemes or pass the complete matrix."
                )
            for station_id in selected_stations:
                station = get_station_config(station_id)
                if not args.dry_run:
                    require_all_artifacts(
                        station, args.seed, args.accept_legacy_metadata
                    )
                for task in evaluation_commands(station, args.seed, args.device):
                    run_task(
                        task, args.dry_run, True, args.restart_incomplete,
                        log_root, state,
                    )
        if args.stage in {"summarize", "all"}:
            if set(selected_stations) != set(SUITE_STATIONS):
                raise ValueError(
                    "The cross-station report requires xiamen, lianyungang and beihai."
                )
            task = summary_task(args.seed)
            run_task(
                task, args.dry_run, True, args.restart_incomplete,
                log_root, state,
            )
        save_state(state_path, state, args.dry_run)
        write_status(
            build_status_rows(selected_stations, selected_schemes, args.seed),
            status_csv,
        )
    except Exception:
        save_state(state_path, state, args.dry_run)
        write_status(
            build_status_rows(selected_stations, selected_schemes, args.seed),
            status_csv,
        )
        raise
    print(f"Suite state: {state_path}", flush=True)


if __name__ == "__main__":
    main()
