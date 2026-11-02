"""MSE/WHuber历史台风对照的路径、权重身份与准备流程测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.xiamen_forecast import run_typhoon_events as workflow
from src.xiamen_forecast.forecast_model import create_model
from src.xiamen_forecast.prepare_typhoon_mse_whuber_comparison import (
    EXPECTED_EVENT_IDS,
    prepare,
)
from src.xiamen_forecast.rolling_diagnostics import recursive_predictions


def _checkpoint(
    station: str = "xiamen",
    *,
    whuber: bool = False,
    rollout: bool = False,
    scalers: dict[str, object] | None = None,
) -> dict[str, object]:
    checkpoint: dict[str, object] = {
        "station_id": station,
        "model_name": "CNNGRUForecastModel",
        "input_steps": 24,
        "scalers": scalers or {
            "atmosphere": {"mean": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
            "surge": {"mean": [0.0], "scale": [1.0]},
        },
    }
    if whuber:
        checkpoint["loss_config"] = {"loss_type": "whuber"}
        checkpoint["training_config"] = {"seed": 42}
    else:
        checkpoint["loss_config"] = {"loss_type": "mse"}
    if rollout:
        checkpoint["rollout_training"] = {
            "base_model_type": "cnn_gru",
            "base_experiment": "cnn_gru_extreme_whuber" if whuber else "cnn_gru",
            "rollout_steps": 6,
            "seed": 42,
            "loss_config": {"loss_type": "whuber" if whuber else "mse"},
        }
    return checkpoint


def _four_checkpoints(station: str = "xiamen") -> dict[str, dict[str, object]]:
    return {
        "cnn_gru": _checkpoint(station),
        "cnn_gru_rollout6": _checkpoint(station, rollout=True),
        "whuber_one_step": _checkpoint(station, whuber=True),
        "whuber_rollout6": _checkpoint(station, whuber=True, rollout=True),
    }


@pytest.mark.parametrize("station_id", ("xiamen", "lianyungang", "beihai"))
def test_three_station_whuber_paths(station_id):
    ridge, one_step, rollout, extras = workflow.model_files(station_id, 42)
    formal = one_step.parents[1]
    assert ridge.name == "ridge_pipeline.joblib"
    assert one_step == formal / "cnn_gru" / "best_model.pth"
    assert rollout == formal / "cnn_gru_rollout6" / "best_model.pth"
    assert extras == {
        "whuber_one_step": formal / "cnn_gru_extreme_whuber" / "best_model.pth",
        "whuber_rollout6": (
            formal / "cnn_gru_extreme_whuber_rollout6_whuber" / "best_model.pth"
        ),
    }


def test_checkpoint_identity_and_scalers_are_strict():
    checkpoints = _four_checkpoints()
    workflow.validate_event_checkpoints(checkpoints, "xiamen", 42)

    wrong_station = _four_checkpoints()
    wrong_station["whuber_one_step"]["station_id"] = "beihai"
    with pytest.raises(ValueError, match="station_id mismatch"):
        workflow.validate_event_checkpoints(wrong_station, "xiamen", 42)

    wrong_scaler = _four_checkpoints()
    wrong_scaler["whuber_rollout6"]["scalers"] = {
        "atmosphere": {"mean": np.array([1.0, 0.0, 0.0]), "scale": [1, 1, 1]},
        "surge": {"mean": [0.0], "scale": [1.0]},
    }
    with pytest.raises(ValueError, match="scalers inconsistent"):
        workflow.validate_event_checkpoints(wrong_scaler, "xiamen", 42)

    wrong_base = _four_checkpoints()
    wrong_base["whuber_rollout6"]["rollout_training"]["base_experiment"] = "cnn_gru"
    with pytest.raises(ValueError, match="base experiment mismatch"):
        workflow.validate_event_checkpoints(wrong_base, "xiamen", 42)


def test_load_event_models_loads_exact_four_models(monkeypatch, tmp_path):
    station = SimpleNamespace(model_root=tmp_path / "models")
    monkeypatch.setattr(workflow, "get_station_config", lambda station_id: station)
    model = create_model("cnn_gru", 24, ("U10", "V10", "MSL"), 8)
    architecture = model.architecture_config()
    checkpoints = _four_checkpoints()
    formal = station.model_root / "formal_seed42"
    directories = {
        "cnn_gru": "cnn_gru",
        "cnn_gru_rollout6": "cnn_gru_rollout6",
        "whuber_one_step": "cnn_gru_extreme_whuber",
        "whuber_rollout6": "cnn_gru_extreme_whuber_rollout6_whuber",
    }
    for label, directory in directories.items():
        path = formal / directory / "best_model.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **architecture,
                **checkpoints[label],
                "model_state_dict": model.state_dict(),
            },
            path,
        )
    loaded_models, loaded_checkpoints, loaded_paths = workflow.load_event_models(
        "xiamen", 42, torch.device("cpu")
    )
    assert set(loaded_models) == set(workflow.MAIN_MODELS)
    assert set(loaded_checkpoints) == set(workflow.MAIN_MODELS)
    assert set(loaded_paths) == set(workflow.MAIN_MODELS)
    assert all(model.training is False for model in loaded_models.values())


def _write_review_source(
    root: Path,
    event_ids: list[str],
    *,
    best_track_sha256: str = "same-track",
) -> None:
    root.mkdir()
    rows = []
    for event_id in event_ids:
        station, storm_sid = event_id.split("_", 1)
        rows.append({
            "event_id": event_id,
            "station": station,
            "storm_name": storm_sid,
            "storm_sid": storm_sid,
            "closest_time_utc": "1997-08-01T00:00:00",
            "review_decision": "yes",
            "review_note": f"verified note for {event_id}",
        })
    pd.DataFrame(rows).to_csv(
        root / "confirmed_event_list.csv", index=False, encoding="utf-8-sig"
    )
    (root / "source_and_protocol.json").write_text(
        json.dumps({
            "best_track_sha256": best_track_sha256,
            "year": 1997,
            "seed": 42,
            "stations": sorted({row["station"] for row in rows}),
            "initial_distance_km": 500,
            "fixed_origin_offsets_hours": [-72, -48, -24],
            "fixed_horizons_hours": [24, 48, 72],
        }),
        encoding="utf-8",
    )


def test_prepare_merges_four_reviews_and_refuses_overwrite(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _write_review_source(first, list(EXPECTED_EVENT_IDS[:2]))
    _write_review_source(second, list(EXPECTED_EVENT_IDS[2:]))
    output = tmp_path / "comparison"
    prepare((first, second), output)
    merged = pd.read_csv(output / "candidate_events.csv", encoding="utf-8-sig")
    assert merged.event_id.tolist() == list(EXPECTED_EVENT_IDS)
    assert merged.review_decision.str.lower().eq("yes").all()
    assert merged.review_note.str.startswith("verified note").all()
    protocol = json.loads((output / "source_and_protocol.json").read_text())
    assert protocol["fixed_origin_offsets_hours"] == [-72, -48, -24]
    assert protocol["fixed_horizons_hours"] == [24, 48, 72]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare((first, second), output)


def test_prepare_rejects_protocol_mismatch(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _write_review_source(first, list(EXPECTED_EVENT_IDS[:2]))
    _write_review_source(
        second, list(EXPECTED_EVENT_IDS[2:]), best_track_sha256="changed-track"
    )
    with pytest.raises(ValueError, match="best_track_sha256"):
        prepare((first, second), tmp_path / "comparison")


class _ConstantRidge:
    def predict(self, features):
        return np.zeros(len(features), dtype=np.float32)


class _LastHistoryModel(torch.nn.Module):
    def forward(self, atmosphere, history):
        return history[:, -1]


def test_event_recursion_does_not_use_future_observed_surge():
    origin = 30
    atmosphere = np.zeros((80, 3, 1, 1), dtype=np.float32)
    summaries = np.zeros((80, 1, 1), dtype=np.float32)
    first = np.zeros(80, dtype=np.float32)
    second = first.copy()
    second[origin:] = 999.0
    scalers = {
        "atmosphere": {"mean": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        "surge": {"mean": [0.0], "scale": [1.0]},
    }
    models = {name: _LastHistoryModel() for name in workflow.MAIN_MODELS}
    shared = (
        np.array([origin]), summaries, _ConstantRidge(), models,
        scalers, 24, 12, 1, torch.device("cpu"),
    )
    predicted_first = recursive_predictions(atmosphere, first, *shared)
    predicted_second = recursive_predictions(atmosphere, second, *shared)
    for model in ("ridge", *workflow.MAIN_MODELS):
        assert np.array_equal(predicted_first[model], predicted_second[model])
