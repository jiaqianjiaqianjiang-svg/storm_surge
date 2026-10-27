"""Shared definitions and artifact checks for the three-station loss suite."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .extreme_losses import one_step_experiment_directory, rollout_experiment_directory
from .station_config import StationConfig


SUITE_STATIONS = ("xiamen", "lianyungang", "beihai")
FORMAL_SEED = 42
ONE_STEP_EPOCHS = 50
ONE_STEP_PATIENCE = 8
ONE_STEP_BATCH_SIZE = 32
ONE_STEP_LEARNING_RATE = 0.001
ROLLOUT_EPOCHS = 20
ROLLOUT_PATIENCE = 5
ROLLOUT_STEPS = 6
ROLLOUT_TEACHER_FORCING_EPOCHS = 5
ROLLOUT_BATCH_SIZE = 256
ROLLOUT_EMBEDDING_BATCH_SIZE = 1024
ROLLOUT_LEARNING_RATE = 1e-4
EXTREME_ALPHA = 2.0
EXTREME_BETA = 1.0
HUBER_DELTA_CM = 5.0


@dataclass(frozen=True)
class LossScheme:
    """One-step objective and dependent rollout objective."""

    scheme_id: str
    display_name: str
    one_step_loss: str
    rollout_loss: str

    @property
    def one_step_directory(self) -> str:
        return one_step_experiment_directory("cnn_gru", self.one_step_loss)

    @property
    def rollout_directory(self) -> str:
        return rollout_experiment_directory(
            self.one_step_directory, self.rollout_loss, ROLLOUT_STEPS
        )

    @property
    def rolling_label(self) -> str:
        return self.scheme_id.replace("/", "_")


LOSS_SCHEMES = (
    LossScheme("mse/mse", "MSE/MSE", "mse", "mse"),
    LossScheme("wmse/mse", "WMSE/MSE", "wmse", "mse"),
    LossScheme("wmse/wmse", "WMSE/WMSE", "wmse", "wmse"),
    LossScheme("awmse/awmse", "AWMSE/AWMSE", "awmse", "awmse"),
    LossScheme("whuber/whuber", "WHuber/WHuber", "whuber", "whuber"),
)
SCHEMES_BY_ID = {scheme.scheme_id: scheme for scheme in LOSS_SCHEMES}
ONE_STEP_LOSSES = tuple(dict.fromkeys(scheme.one_step_loss for scheme in LOSS_SCHEMES))


ONE_STEP_REQUIRED_FILES = (
    "best_model.pth",
    "validation_predictions.csv",
    "metrics.json",
    "training_config.json",
    "dataset_report.json",
)
ROLLOUT_REQUIRED_FILES = (
    "best_model.pth",
    "training_metadata.json",
    "loss_history.csv",
)


def formal_root(station: StationConfig, seed: int = FORMAL_SEED) -> Path:
    return station.model_root / f"formal_seed{seed}"


def one_step_path(station: StationConfig, loss_type: str, seed: int = FORMAL_SEED) -> Path:
    return formal_root(station, seed) / one_step_experiment_directory("cnn_gru", loss_type)


def rollout_path(station: StationConfig, scheme: LossScheme, seed: int = FORMAL_SEED) -> Path:
    return formal_root(station, seed) / scheme.rollout_directory


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"missing {path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"invalid {path.name}: {exc}"
    if not isinstance(value, dict):
        return None, f"{path.name} must contain a JSON object"
    return value, None


def _different(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) > max(1e-12, abs(expected) * 1e-9)
        except (TypeError, ValueError):
            return True
    return actual != expected


def _compare_fields(
    metadata: dict[str, Any], expected: dict[str, Any]
) -> tuple[list[str], list[str]]:
    mismatches: list[str] = []
    missing: list[str] = []
    for key, value in expected.items():
        if key not in metadata:
            missing.append(key)
        elif _different(metadata[key], value):
            mismatches.append(f"{key}={metadata[key]!r}, expected {value!r}")
    return mismatches, missing


def inspect_one_step_artifact(
    station: StationConfig,
    loss_type: str,
    seed: int = FORMAL_SEED,
) -> dict[str, Any]:
    """Classify a one-step checkpoint directory without loading its large weight."""

    root = one_step_path(station, loss_type, seed)
    missing_files = [name for name in ONE_STEP_REQUIRED_FILES if not (root / name).is_file()]
    if missing_files:
        return {
            "status": "absent" if not root.exists() else "incomplete",
            "path": str(root),
            "missing_files": missing_files,
            "mismatches": [],
            "missing_metadata": [],
        }
    config, error = _read_json(root / "training_config.json")
    loss_path = root / "extreme_loss_config.json"
    loss_config, loss_error = (
        _read_json(loss_path) if loss_type != "mse" or loss_path.exists() else (None, None)
    )
    if error or loss_error:
        return {
            "status": "configuration_invalid",
            "path": str(root),
            "missing_files": [],
            "mismatches": [item for item in (error, loss_error) if item],
            "missing_metadata": [],
        }
    expected = {
        "station": station.station_id,
        "model_type": "cnn_gru",
        "loss_type": loss_type,
        "validation_only": True,
        "seed": seed,
        "input_steps": 24,
        "batch_size": ONE_STEP_BATCH_SIZE,
        "epochs": ONE_STEP_EPOCHS,
        "patience": ONE_STEP_PATIENCE,
        "lr": ONE_STEP_LEARNING_RATE,
        "train_start_year": station.train_start_year,
        "train_end_year": station.train_end_year,
        "validation_year": station.validation_year,
    }
    mismatches, missing_metadata = _compare_fields(config or {}, expected)
    if loss_config is not None:
        expected_loss = {"loss_type": loss_type}
        if loss_type != "mse":
            expected_loss.update(
                alpha=EXTREME_ALPHA, beta=EXTREME_BETA,
                huber_delta_cm=HUBER_DELTA_CM,
            )
        loss_mismatches, loss_missing = _compare_fields(loss_config, expected_loss)
        mismatches.extend(f"loss.{item}" for item in loss_mismatches)
        missing_metadata.extend(f"loss.{item}" for item in loss_missing)
    status = "compatible"
    if mismatches:
        status = "configuration_mismatch"
    elif missing_metadata:
        status = "legacy_metadata"
    return {
        "status": status,
        "path": str(root),
        "missing_files": [],
        "mismatches": mismatches,
        "missing_metadata": missing_metadata,
    }


def inspect_rollout_artifact(
    station: StationConfig,
    scheme: LossScheme,
    seed: int = FORMAL_SEED,
) -> dict[str, Any]:
    """Classify a rollout checkpoint and verify its one-step dependency."""

    root = rollout_path(station, scheme, seed)
    missing_files = [name for name in ROLLOUT_REQUIRED_FILES if not (root / name).is_file()]
    if missing_files:
        return {
            "status": "absent" if not root.exists() else "incomplete",
            "path": str(root),
            "missing_files": missing_files,
            "mismatches": [],
            "missing_metadata": [],
        }
    metadata, error = _read_json(root / "training_metadata.json")
    loss_path = root / "extreme_loss_config.json"
    loss_config, loss_error = (
        _read_json(loss_path)
        if scheme.rollout_loss != "mse" or loss_path.exists()
        else (None, None)
    )
    if error or loss_error:
        return {
            "status": "configuration_invalid",
            "path": str(root),
            "missing_files": [],
            "mismatches": [item for item in (error, loss_error) if item],
            "missing_metadata": [],
        }
    expected = {
        "station": station.station_id,
        "training_years": [station.train_start_year, station.train_end_year],
        "validation_year": station.validation_year,
        "test_year_loaded": False,
        "base_model_type": "cnn_gru",
        "base_experiment": scheme.one_step_directory,
        "rollout_steps": ROLLOUT_STEPS,
        "teacher_forcing_epochs": ROLLOUT_TEACHER_FORCING_EPOCHS,
        "seed": seed,
        "epochs": ROLLOUT_EPOCHS,
        "patience": ROLLOUT_PATIENCE,
        "batch_size": ROLLOUT_BATCH_SIZE,
        "embedding_batch_size": ROLLOUT_EMBEDDING_BATCH_SIZE,
        "learning_rate": ROLLOUT_LEARNING_RATE,
    }
    mismatches, missing_metadata = _compare_fields(metadata or {}, expected)
    if loss_config is not None:
        expected_loss = {"loss_type": scheme.rollout_loss}
        if scheme.rollout_loss != "mse":
            expected_loss.update(
                alpha=EXTREME_ALPHA, beta=EXTREME_BETA,
                huber_delta_cm=HUBER_DELTA_CM,
            )
        loss_mismatches, loss_missing = _compare_fields(loss_config, expected_loss)
        mismatches.extend(f"loss.{item}" for item in loss_mismatches)
        missing_metadata.extend(f"loss.{item}" for item in loss_missing)
    expected_base = one_step_path(station, scheme.one_step_loss, seed) / "best_model.pth"
    recorded_base = Path(str((metadata or {}).get("base_checkpoint", "")))
    if not recorded_base.name:
        missing_metadata.append("base_checkpoint")
    elif (
        recorded_base.name != expected_base.name
        or recorded_base.parent.name != expected_base.parent.name
    ):
        mismatches.append(
            f"base_checkpoint={recorded_base}, expected experiment {expected_base.parent.name}"
        )
    status = "compatible"
    if mismatches:
        status = "configuration_mismatch"
    elif missing_metadata:
        status = "legacy_metadata"
    return {
        "status": status,
        "path": str(root),
        "missing_files": [],
        "mismatches": mismatches,
        "missing_metadata": missing_metadata,
    }


def reusable(status: str, accept_legacy_metadata: bool = False) -> bool:
    return status == "compatible" or (
        accept_legacy_metadata and status == "legacy_metadata"
    )
