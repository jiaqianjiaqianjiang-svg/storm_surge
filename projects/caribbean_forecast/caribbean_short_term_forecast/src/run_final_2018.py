"""Run the locked 2018 Prickly Bay direct and recursive final evaluations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--evaluation-year", type=int, default=2018)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-rolling", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def run(command: list[str], step: int, total: int) -> None:
    print(f"[{step}/{total}] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=MODULE_ROOT)


def main() -> None:
    args = parse_args()
    if args.evaluation_year <= 2017:
        raise ValueError("The final workflow requires an evaluation year after 2017")
    experiments = MODULE_ROOT / "outputs" / "experiments" / args.station
    rollout = MODULE_ROOT / "models" / args.station / f"rollout6_seed{args.seed}" / "best_model.pth"
    if not rollout.is_file():
        raise FileNotFoundError(f"Missing locked rollout checkpoint: {rollout}")

    commands: list[list[str]] = []
    if not args.skip_direct:
        commands.append([
            sys.executable, str(MODULE_ROOT / "src" / "evaluate_direct_2018.py"),
            "--station", args.station, "--evaluation-year", str(args.evaluation_year),
            "--device", args.device,
        ])
    if not args.skip_rolling:
        commands.append([
            sys.executable, str(MODULE_ROOT / "src" / "rolling_72_comparison.py"),
            "--station", args.station, "--evaluation-year", str(args.evaluation_year),
            "--device", args.device, "--rollout-checkpoint", str(rollout),
            "--output-dir", str(
                experiments / f"rolling_72_with_rollout6_{args.evaluation_year}_seed{args.seed}"
            ),
        ])
    if not args.skip_export:
        commands.append([
            sys.executable, str(MODULE_ROOT / "src" / "export_final_results.py"),
            "--station", args.station, "--evaluation-year", str(args.evaluation_year),
            "--seed", str(args.seed),
        ])
    for index, command in enumerate(commands, start=1):
        run(command, index, len(commands))
    print("Prickly Bay independent-test workflow completed.")


if __name__ == "__main__":
    main()
