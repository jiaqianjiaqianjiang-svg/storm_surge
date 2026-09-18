"""Export compact Xiamen result summaries that are safe to commit to Git."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = (
    "surge_mlp",
    "era5_cnn",
    "cnn",
    "cnn_lstm",
    "cnn_gru",
    "tcn",
    "transformer",
)


def repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="xiamen")
    parser.add_argument("--validation-year", type=int, default=1996)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required result is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_optional(source: Path, destination: Path) -> bool:
    if not source.is_file():
        print(f"warning: optional result missing: {source}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.{decimals}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_summary(
    destination: Path,
    validation_metrics: pd.DataFrame,
    rolling_rmse: pd.DataFrame,
    rollout_metadata: dict[str, object],
    validation_year: int,
    seed: int,
) -> None:
    ranked = validation_metrics.sort_values("rmse_cm").copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked = ranked[
        [
            "rank",
            "model",
            "rmse_cm",
            "mae_cm",
            "pearson_r",
            "skill_score_vs_ridge",
        ]
    ]
    rollout_column = next(
        column for column in rolling_rmse.columns if "rollout" in column
    )
    base_column = str(rollout_metadata.get("base_model_type", "cnn_gru"))
    gains = rolling_rmse[["lead_hours", base_column, rollout_column]].copy()
    gains["rmse_reduction_percent"] = (
        100 * (1 - gains[rollout_column] / gains[base_column])
    )
    content = f"""# Xiamen Short-Term Forecast Results

## Experiment

- Validation year: {validation_year}
- Random seed: {seed}
- Input window: 24 hours
- Rolling evaluation: known future ERA5 forcing; historical hindcast, not an operational forecast
- Test-year observations were not loaded during rollout fine-tuning

## One-Step Validation Ranking

{markdown_table(ranked)}

## Recursive RMSE

{markdown_table(rolling_rmse)}

## Rollout Improvement Over Base Model

{markdown_table(gains)}

## Rollout Training

- Base recursive validation RMSE: {float(rollout_metadata['base_validation_rmse_cm']):.4f} cm
- Best recursive validation RMSE: {float(rollout_metadata['best_validation_rmse_cm']):.4f} cm
- Best epoch: {int(rollout_metadata['best_epoch'])}
- Improved over base: {bool(rollout_metadata['fine_tuned_improved_over_base'])}

Only compact metrics, reports, and figures are archived here. Raw data, prediction tables,
and model weights remain excluded from Git.
"""
    (destination / "RESULTS_SUMMARY.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repository = repository_root(MODULE_ROOT)
    destination = args.output_dir or (
        repository
        / "reports"
        / "experiment_results"
        / f"xiamen_short_term_{args.validation_year}_seed{args.seed}"
    )
    destination.mkdir(parents=True, exist_ok=True)

    experiment_root = MODULE_ROOT / "outputs" / "experiments" / args.station
    comparison = experiment_root / "model_comparison"
    rolling = experiment_root / f"rolling_{args.validation_year}_seed{args.seed}"
    model_root = MODULE_ROOT / "models" / args.station
    formal = model_root / f"formal_seed{args.seed}"
    rollout = formal / "cnn_gru_rollout6"

    validation_source = comparison / "validation_model_metrics.csv"
    rolling_source = rolling / "rolling_rmse_table.csv"
    rollout_metadata_source = rollout / "training_metadata.json"
    copy_required(validation_source, destination / "validation_model_metrics.csv")
    copy_required(rolling_source, destination / "rolling_rmse_table.csv")
    copy_required(
        rollout_metadata_source, destination / "rollout_training_metadata.json"
    )

    selected_optional = (
        (
            comparison / "validation_model_comparison.png",
            "validation_model_comparison.png",
        ),
        (rolling / "rolling_metrics_long.csv", "rolling_metrics_long.csv"),
        (rolling / "rmse_vs_lead.png", "rmse_vs_lead.png"),
        (rolling / "rolling_diagnostic_report.md", "rolling_diagnostic_report.md"),
        (
            rolling / "experiment_metadata.json",
            "rolling_experiment_metadata.json",
        ),
        (rollout / "loss_history.csv", "rollout_loss_history.csv"),
        (rollout / "loss_curve.png", "rollout_loss_curve.png"),
        (
            model_root / "baselines" / "baseline_metrics.json",
            "baseline_metrics.json",
        ),
    )
    for source, name in selected_optional:
        copy_optional(source, destination / name)
    for source in sorted(rolling.glob("strong_event_*.png")):
        copy_optional(source, destination / source.name)
    for model_name in MODEL_NAMES:
        copy_optional(
            formal / model_name / "metrics.json",
            destination / f"metrics_{model_name}.json",
        )

    validation_metrics = pd.read_csv(validation_source)
    rolling_rmse = pd.read_csv(rolling_source)
    rollout_metadata = json.loads(
        rollout_metadata_source.read_text(encoding="utf-8")
    )
    build_summary(
        destination,
        validation_metrics,
        rolling_rmse,
        rollout_metadata,
        args.validation_year,
        args.seed,
    )
    print(f"Git-safe result package: {destination}")
    print("Review it, then commit only this reports/experiment_results directory.")


if __name__ == "__main__":
    main()
