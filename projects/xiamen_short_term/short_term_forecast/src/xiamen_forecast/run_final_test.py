"""Run the fixed Xiamen independent-test workflow without retraining models."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


MODULE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=1997)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--rollout-checkpoint", type=Path)
    return parser.parse_args()


def build_commands(
    year: int,
    seed: int,
    device: str,
    language: str,
    rollout_checkpoint: Path,
) -> list[list[str]]:
    python = sys.executable
    common_year = str(year)
    return [
        [
            python,
            "-m",
            "src.xiamen_forecast.compare_models",
            "--split",
            "test",
        ],
        [
            python,
            "-m",
            "src.xiamen_forecast.rolling_diagnostics",
            "--evaluation-year",
            common_year,
            "--split",
            "test",
            "--device",
            device,
            "--rollout-checkpoint",
            str(rollout_checkpoint),
        ],
        [
            python,
            "-m",
            "src.short_term_forecast.journal_figures.make_xiamen_journal_figures",
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
    if args.year != 1997:
        raise ValueError("The formal independent-test workflow is fixed to 1997")
    checkpoint = args.rollout_checkpoint or (
        MODULE_ROOT
        / "models"
        / "xiamen"
        / f"formal_seed{args.seed}"
        / "cnn_gru_rollout6"
        / "best_model.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Rollout checkpoint not found: {checkpoint}")

    commands = build_commands(
        args.year, args.seed, args.device, args.language, checkpoint
    )
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=MODULE_ROOT, check=True)

    print("1997 independent-test workflow completed.", flush=True)


if __name__ == "__main__":
    main()
