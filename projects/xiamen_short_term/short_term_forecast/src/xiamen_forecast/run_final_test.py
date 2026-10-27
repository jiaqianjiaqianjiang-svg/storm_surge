"""Run a fixed station independent-test workflow without retraining models."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from .station_config import STATIONS, get_station_config


MODULE_ROOT = Path(__file__).resolve().parents[2]
FORMAL_MODELS = (
    "surge_mlp",
    "era5_cnn",
    "cnn",
    "cnn_lstm",
    "cnn_gru",
    "tcn",
    "transformer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", choices=(*STATIONS, "shanwei"), default="xiamen")
    parser.add_argument("--year", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--rollout-checkpoint", type=Path)
    parser.add_argument("--profile", choices=("full", "core"), default="full")
    return parser.parse_args()


def build_commands(
    year: int,
    seed: int,
    device: str,
    language: str,
    rollout_checkpoint: Path,
    station: str = "xiamen",
    profile: str = "full",
) -> list[list[str]]:
    if profile not in {"full", "core"}:
        raise ValueError(f"Unknown model profile: {profile}")
    models = FORMAL_MODELS if profile == "full" else ("cnn_gru",)
    python = sys.executable
    common_year = str(year)
    baseline_command = [
        python, "-m", "src.xiamen_forecast.evaluate_baselines", "--station", station,
    ]
    if profile == "core":
        baseline_command.extend([
            "--existing-ridge",
            str(get_station_config(station).model_root / "baselines" / "ridge_pipeline.joblib"),
        ])
    return [
        baseline_command,
        [
            python,
            "-m",
            "src.xiamen_forecast.evaluate_checkpoints",
            "--station",
            station,
            "--split",
            "test",
            "--models",
            *models,
            "--seed",
            str(seed),
            "--year",
            common_year,
            "--device",
            device,
        ],
        [
            python,
            "-m",
            "src.xiamen_forecast.compare_models",
            "--station",
            station,
            "--split",
            "test",
            "--seed",
            str(seed),
        ],
        [
            python,
            "-m",
            "src.xiamen_forecast.rolling_diagnostics",
            "--station",
            station,
            "--models",
            *models,
            "--evaluation-year",
            common_year,
            "--split",
            "test",
            "--device",
            device,
            "--seed",
            str(seed),
            "--rollout-checkpoint",
            str(rollout_checkpoint),
        ],
        [
            python,
            "-m",
            "src.short_term_forecast.journal_figures.make_xiamen_journal_figures",
            "--station",
            station,
            "--validation-year",
            common_year,
            "--split",
            "test",
            "--seed",
            str(seed),
            "--language",
            language,
        ],
        [
            python,
            "-m",
            "src.xiamen_forecast.export_final_results",
            "--station",
            station,
            "--evaluation-year",
            common_year,
            "--split",
            "test",
            "--seed",
            str(seed),
        ],
    ]


def main() -> None:
    args = parse_args()
    station = get_station_config(args.station)
    year = station.test_year if args.year is None else args.year
    if year != station.test_year:
        raise ValueError(
            f"The {station.name} independent-test workflow is fixed to "
            f"{station.test_year}"
        )
    checkpoint = args.rollout_checkpoint or (
        MODULE_ROOT
        / "models"
        / station.station_id
        / f"formal_seed{args.seed}"
        / "cnn_gru_rollout6"
        / "best_model.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Rollout checkpoint not found: {checkpoint}")

    commands = build_commands(
        year, args.seed, args.device, args.language, checkpoint, station.station_id,
        args.profile,
    )
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=MODULE_ROOT, check=True)

    print(f"{station.name} {year} independent-test workflow completed.", flush=True)


if __name__ == "__main__":
    main()
