"""Train leakage-safe one-hour Xiamen models from prepared aligned arrays."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .dataset_builder import build_datasets, build_year_datasets
    from .evaluate import calculate_detailed_metrics
    from .forecast_model import MODEL_TYPES, create_model
    from .plotting import loss_curve, observed_vs_predicted, validation_scatter
except ImportError:
    from dataset_builder import build_datasets, build_year_datasets
    from evaluate import calculate_detailed_metrics
    from forecast_model import MODEL_TYPES, create_model
    from plotting import loss_curve, observed_vs_predicted, validation_scatter


MODULE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="xiamen")
    parser.add_argument("--start-year", type=int, default=1970)
    parser.add_argument("--end-year", type=int, default=1997)
    parser.add_argument("--split-mode", choices=["ratio", "years"], default="years")
    parser.add_argument("--train-start-year", type=int, default=1970)
    parser.add_argument("--train-end-year", type=int, default=1995)
    parser.add_argument("--validation-year", type=int, default=1996)
    parser.add_argument("--test-year", type=int, default=1997)
    parser.add_argument("--input-steps", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-type", choices=sorted(MODEL_TYPES), default="dual")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prepared(path: Path, start_year: int, end_year: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    if path.is_dir():
        required = {name: path / f"{name}.npy" for name in ("atmosphere", "surge", "time")}
        missing = [str(value) for value in required.values() if not value.is_file()]
        if missing:
            raise ValueError(f"Prepared dataset directory is missing: {missing}")
        atmosphere = np.load(required["atmosphere"], mmap_mode="r", allow_pickle=False)
        surge = np.load(required["surge"], mmap_mode="r", allow_pickle=False)
        times = pd.DatetimeIndex(pd.to_datetime(np.load(required["time"], mmap_mode="r", allow_pickle=False)))
    elif path.is_file():
        with np.load(path, mmap_mode="r", allow_pickle=False) as data:
            required_names = {"atmosphere", "surge", "time"}
            if not required_names <= set(data.files):
                raise ValueError(f"{path} must contain arrays: {sorted(required_names)}")
            atmosphere, surge = data["atmosphere"], data["surge"]
            times = pd.DatetimeIndex(pd.to_datetime(data["time"]))
    else:
        raise FileNotFoundError(
            f"Prepared dataset not found: {path}. Run inspect_data.py, QC/UTide, ERA5 loading, "
            "then run prepare_xiamen.py first."
        )
    positions = np.flatnonzero((times.year >= start_year) & (times.year <= end_year))
    if not len(positions):
        raise ValueError(f"No prepared observations found for {start_year}-{end_year}")
    start, stop = int(positions[0]), int(positions[-1]) + 1
    return atmosphere[start:stop], surge[start:stop], times[start:stop]


def main() -> None:
    args = parse_args(); set_seed(args.seed)
    dataset_path = args.dataset_path or MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    output = args.output_dir or (
        MODULE_ROOT / "models" / args.station / "formal_seed42" / args.model_type
    )
    output.mkdir(parents=True, exist_ok=True)
    if args.split_mode == "years" and not args.smoke_test:
        load_start, load_end = args.train_start_year, args.test_year
    else:
        load_start, load_end = args.start_year, args.end_year
    atmosphere, surge, times = load_prepared(dataset_path, load_start, load_end)
    if args.smoke_test:
        limit = min(len(times), max(args.input_steps + 16, 64)); atmosphere, surge, times = atmosphere[:limit], surge[:limit], times[:limit]
        args.epochs = min(args.epochs, 2)
    if args.split_mode == "years" and not args.smoke_test:
        train_set, validation_set, test_set, dataset_report = build_year_datasets(
            atmosphere, surge, times, args.input_steps,
            args.train_start_year, args.train_end_year,
            args.validation_year, args.test_year,
            args.model_type,
        )
    else:
        train_set, validation_set, dataset_report = build_datasets(
            atmosphere, surge, times, args.input_steps, model_type=args.model_type
        )
        dataset_report["split_mode"] = "ratio"
        test_set = None
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size) if test_set else None
    variables = ("U10", "V10", "MSL")
    model = create_model(args.model_type, args.input_steps, variables, atmosphere.shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    best_loss, stale, history = float("inf"), 0, []
    checkpoint_path = output / "best_model.pth"
    config = vars(args).copy(); config.update({"dataset_path": str(dataset_path), "output_dir": str(output), "device": str(device)})
    surge_scale_m = float(dataset_report["scalers"]["surge"]["scale"][0])
    for epoch in range(1, args.epochs + 1):
        model.train(); total, count = 0.0, 0
        for weather, surge_history, target in train_loader:
            weather, surge_history, target = weather.to(device), surge_history.to(device), target.to(device)
            optimizer.zero_grad(); loss = criterion(model(weather, surge_history), target); loss.backward(); optimizer.step()
            total += loss.item() * len(target); count += len(target)
        model.eval(); val_total, val_count = 0.0, 0
        with torch.no_grad():
            for weather, surge_history, target in validation_loader:
                target = target.to(device); predictions = model(weather.to(device), surge_history.to(device))
                val_total += criterion(predictions, target).item() * len(target); val_count += len(target)
        train_loss, val_loss = total / count, val_total / val_count
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_rmse_cm": float(np.sqrt(train_loss) * surge_scale_m * 100),
            "validation_rmse_cm": float(np.sqrt(val_loss) * surge_scale_m * 100),
        })
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} validation_loss={val_loss:.6f} "
            f"validation_rmse_cm={history[-1]['validation_rmse_cm']:.4f}"
        )
        if val_loss < best_loss - 1e-8:
            best_loss, stale = val_loss, 0
            checkpoint = {
                **model.architecture_config(), "model_state_dict": model.state_dict(),
                "scalers": dataset_report["scalers"], "station_id": args.station,
                "training_time_range": dataset_report["train_time_range"],
                "dataset_split": dataset_report,
                "selection_metric": (
                    f"minimum {args.validation_year} validation scaled MSE "
                    "(equivalent to RMSE)"
                ),
                "best_validation_rmse_cm": history[-1]["validation_rmse_cm"],
                "training_config": config,
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}"); break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    scale = dataset_report["scalers"]["surge"]["scale"][0]; mean = dataset_report["scalers"]["surge"]["mean"][0]

    baseline_dir = args.baseline_dir or MODULE_ROOT / "models" / args.station / "baselines"

    def evaluate_split(loader: DataLoader, dataset: object, name: str) -> dict[str, object]:
        scaled_predictions, scaled_observed = [], []
        with torch.no_grad():
            for weather, surge_history, target in loader:
                predictions = model(weather.to(device), surge_history.to(device))
                scaled_predictions.extend(predictions.cpu().numpy())
                scaled_observed.extend(target.numpy())
        predicted = np.asarray(scaled_predictions) * scale + mean
        observed = np.asarray(scaled_observed) * scale + mean
        dates = dataset.times[dataset.targets]
        frame = pd.DataFrame(
            {"datetime": dates, "observed_m": observed, "predicted_m": predicted}
        )
        frame.to_csv(output / f"{name}_predictions.csv", index=False)
        observed_vs_predicted(
            dates, observed, predicted, output / f"{name}_observed_vs_predicted.png"
        )
        validation_scatter(
            observed, predicted, output / f"{name}_scatter.png"
        )
        reference_path = baseline_dir / f"{name}_baseline_predictions.csv"
        ridge_reference = None
        if reference_path.is_file():
            reference = pd.read_csv(reference_path)
            reference["datetime"] = pd.to_datetime(reference.datetime)
            reference = reference.set_index("datetime").reindex(pd.DatetimeIndex(dates))
            if reference.ridge_m.isna().any():
                if args.split_mode == "years" and not args.smoke_test:
                    raise ValueError(
                        f"Ridge reference {reference_path} does not cover all {name} timestamps"
                    )
            else:
                ridge_reference = reference.ridge_m.to_numpy(dtype=float)
        return calculate_detailed_metrics(
            observed, predicted, dates, ridge_reference
        )

    metrics = {"validation": evaluate_split(validation_loader, validation_set, "validation")}
    if test_loader is not None and test_set is not None:
        metrics["test"] = evaluate_split(test_loader, test_set, "test")
    pd.DataFrame(history).to_csv(output / "loss_history.csv", index=False)
    loss_curve(pd.DataFrame(history), output / "loss_curve.png")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "training_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (output / "dataset_report.json").write_text(
        json.dumps(dataset_report, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2)); print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
