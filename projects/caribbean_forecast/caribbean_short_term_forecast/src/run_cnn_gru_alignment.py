"""Train and evaluate the aligned Prickly Bay CNN-GRU experiment."""

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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-one-step-training", action="store_true")
    parser.add_argument("--skip-rollout-training", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def run(command: list[str], step: int, total: int) -> None:
    print(f"[{step}/{total}] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=MODULE_ROOT)


def main() -> None:
    args = parse_args()
    if args.evaluation_year <= 2017:
        raise ValueError("Independent evaluation year must be after 2017")
    model_root = MODULE_ROOT / "models" / args.station / f"formal_seed{args.seed}"
    cnn_gru = model_root / "cnn_gru" / "best_model.pth"
    rollout = model_root / "cnn_gru_rollout6" / "best_model.pth"
    experiments = MODULE_ROOT / "outputs" / "experiments" / args.station
    rolling_output = experiments / f"rolling_72_core_aligned_{args.evaluation_year}_seed{args.seed}"
    repository_root = MODULE_ROOT.parents[2]
    export_output = (
        repository_root / "reports" / "experiment_results" /
        f"{args.station}_core_aligned_{args.evaluation_year}_seed{args.seed}"
    )
    commands: list[list[str]] = []
    if not args.skip_one_step_training:
        commands.append([
            sys.executable, "-m", "src.train_station", "--station", args.station,
            "--split-mode", "years", "--train-start-year", "2011",
            "--train-end-year", "2016", "--validation-year", "2017",
            "--test-year", str(args.evaluation_year), "--input-steps", "24",
            "--model-type", "cnn_gru", "--seed", str(args.seed),
            "--device", args.device, "--num-workers", str(args.num_workers),
            "--output-dir", str(model_root / "cnn_gru"),
        ])
    if not args.skip_rollout_training:
        commands.append([
            sys.executable, "-m", "src.train_rollout_temporal",
            "--station", args.station, "--train-start-year", "2011",
            "--train-end-year", "2016", "--validation-year", "2017",
            "--base-checkpoint", str(cnn_gru), "--rollout-steps", "6",
            "--seed", str(args.seed), "--device", args.device,
            "--num-workers", str(args.num_workers),
            "--output-dir", str(model_root / "cnn_gru_rollout6"),
        ])
    commands.append([
        sys.executable, "-m", "src.rolling_72_comparison",
        "--station", args.station, "--evaluation-year", str(args.evaluation_year),
        "--device", args.device, "--cnn-gru-checkpoint", str(cnn_gru),
        "--cnn-gru-rollout-checkpoint", str(rollout),
        "--output-dir", str(rolling_output),
    ])
    if not args.skip_export:
        commands.append([
            sys.executable, "-m", "src.export_final_results",
            "--station", args.station, "--evaluation-year", str(args.evaluation_year),
            "--seed", str(args.seed), "--rolling-dir", str(rolling_output),
            "--output-dir", str(export_output),
        ])
    for index, command in enumerate(commands, 1):
        run(command, index, len(commands))
    print("Prickly Bay CNN-GRU alignment workflow completed.")


if __name__ == "__main__":
    main()
