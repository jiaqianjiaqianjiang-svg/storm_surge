"""Run selected station model trainings sequentially and resumably."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .station_config import STATIONS


MODULE_ROOT = Path(__file__).resolve().parents[2]
FORMAL_MODELS = ("cnn", "cnn_lstm", "cnn_gru", "tcn", "transformer")
DEFAULT_BATCH_SIZES = {
    "cnn": 256,
    "cnn_lstm": 32,
    "cnn_gru": 32,
    "tcn": 32,
    "transformer": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", choices=tuple(STATIONS), default="xiamen")
    parser.add_argument("--models", nargs="+", choices=FORMAL_MODELS, default=FORMAL_MODELS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also evaluate the independent test year; omit during model selection.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_root = (
        MODULE_ROOT / "models" / args.station / f"formal_seed{args.seed}"
    )
    for name in args.models:
        metrics = checkpoint_root / name / "metrics.json"
        if metrics.is_file() and not args.force:
            print(f"skip {name}: completed metrics already exist at {metrics}", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "src.xiamen_forecast.train_xiamen",
            "--station",
            args.station,
            "--model-type",
            name,
            "--batch-size",
            str(DEFAULT_BATCH_SIZES[name]),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--device",
            args.device,
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
        ]
        if not args.include_test:
            command.append("--validation-only")
        print(f"start {name}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=MODULE_ROOT, check=True)
    print("formal model suite complete", flush=True)


if __name__ == "__main__":
    main()
