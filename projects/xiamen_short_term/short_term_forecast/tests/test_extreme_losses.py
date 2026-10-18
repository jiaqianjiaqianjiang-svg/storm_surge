import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.xiamen_forecast.compare_extreme_losses import (
    collect_common_predictions,
    plot_summary,
    summarise_predictions,
)
from src.xiamen_forecast.dataset_builder import build_train_validation_year_datasets
from src.xiamen_forecast.evaluate import calculate_detailed_metrics
from src.xiamen_forecast.extreme_losses import (
    build_training_loss,
    extreme_ratio_from_scaled_targets,
    fit_positive_extreme_thresholds,
    one_step_experiment_directory,
    rollout_experiment_directory,
)
from src.xiamen_forecast.rolling_diagnostics import parse_extra_checkpoints
from src.xiamen_forecast.train_xiamen import raw_training_targets


def test_extreme_ratio_uses_raw_metres_after_inverse_scaling() -> None:
    mean_m, scale_m = -0.10, 0.20
    physical_m = torch.tensor([-0.20, 0.10, 0.30, 0.50])
    scaled = (physical_m - mean_m) / scale_m
    ratio = extreme_ratio_from_scaled_targets(
        scaled, mean_m, scale_m, q90_m=0.10, q99_m=0.50
    )
    assert torch.allclose(ratio, torch.tensor([0.0, 0.0, 0.5, 1.0]))


def test_wmse_matches_weighted_formula_and_weight_range() -> None:
    training = np.linspace(-0.4, 0.6, 101)
    criterion, metadata = build_training_loss(
        "wmse", training, surge_mean_m=0.0, surge_scale_m=1.0, alpha=2.0
    )
    target = torch.tensor([metadata["q90_m"], metadata["q99_m"]])
    prediction = target + torch.tensor([1.0, 1.0])
    loss = criterion(prediction, target)
    expected = (1.0 * 1.0 + 3.0 * 1.0) / 4.0
    assert loss.item() == pytest.approx(expected)
    assert metadata["threshold_unit"] == "m"
    assert metadata["threshold_sample_count"] == 101


def test_nonpositive_q90_is_protected_and_negative_targets_keep_unit_weight() -> None:
    training = np.concatenate(
        [np.linspace(-1.0, -0.1, 95), np.array([0.2, 0.4, 0.6, 0.8, 1.0])]
    )
    metadata = fit_positive_extreme_thresholds(training)
    assert metadata["raw_q90_m"] <= 0
    assert metadata["q90_m"] == 0.0
    assert metadata["positive_threshold_protected"] is True
    ratio = extreme_ratio_from_scaled_targets(
        torch.tensor([-0.4, 0.0]), 0.0, 1.0, metadata["q90_m"], metadata["q99_m"]
    )
    assert ratio.tolist() == [0.0, 0.0]


def test_invalid_or_zero_threshold_interval_fails_clearly() -> None:
    with pytest.raises(ValueError, match="P99 must be greater"):
        fit_positive_extreme_thresholds(np.ones(100) * -0.2)
    with pytest.raises(ValueError, match="q99_m must be greater"):
        extreme_ratio_from_scaled_targets(torch.zeros(2), 0.0, 1.0, 0.2, 0.2)


def test_awmse_penalises_positive_extreme_underprediction_more_than_overprediction() -> None:
    training = np.linspace(-0.2, 0.8, 201)
    criterion, metadata = build_training_loss(
        "awmse", training, 0.0, 1.0, alpha=2.0, beta=1.0
    )
    target = torch.tensor([metadata["q99_m"]])
    under = criterion(target - 0.1, target)
    over = criterion(target + 0.1, target)
    assert under > over


@pytest.mark.parametrize("loss_type", ["mse", "wmse", "awmse", "whuber"])
def test_all_losses_have_finite_gradients_for_multistep_tensors(loss_type: str) -> None:
    training = np.linspace(-0.3, 0.7, 301)
    criterion, metadata = build_training_loss(
        loss_type, training, surge_mean_m=0.05, surge_scale_m=0.2
    )
    prediction = torch.zeros((4, 6), requires_grad=True)
    target_m = torch.linspace(-0.1, 0.6, 24).reshape(4, 6)
    target = (target_m - 0.05) / 0.2
    loss = criterion(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert metadata["huber_delta_scaled"] == pytest.approx(0.05 / 0.2)


def test_thresholds_use_only_training_targets_not_validation_values() -> None:
    times = pd.date_range("1995-01-01", "1996-01-03", freq="1h")
    atmosphere = np.zeros((len(times), 3, 2, 2), dtype=np.float32)
    surge = np.linspace(-0.2, 0.2, len(times), dtype=np.float32)
    training_mask = times.year == 1995
    surge[~training_mask] = 50.0
    train, _, report = build_train_validation_year_datasets(
        atmosphere,
        surge,
        times,
        input_steps=24,
        train_start_year=1995,
        train_end_year=1995,
        validation_year=1996,
        model_type="cnn_gru",
    )
    targets = raw_training_targets(train)
    metadata = fit_positive_extreme_thresholds(targets)
    assert metadata["q99_m"] < 1.0
    assert report["test_year_loaded"] is False
    assert max(train.times[train.targets]).year == 1995


def test_checkpoint_directories_are_isolated_and_mse_stays_compatible() -> None:
    assert one_step_experiment_directory("cnn_gru", "mse") == "cnn_gru"
    names = {
        one_step_experiment_directory("cnn_gru", loss)
        for loss in ("mse", "wmse", "awmse", "whuber")
    }
    assert len(names) == 4
    assert rollout_experiment_directory("cnn_gru", "mse", 6) == "cnn_gru_rollout6"
    rollout_names = {
        rollout_experiment_directory("cnn_gru_extreme_wmse", "mse", 6),
        rollout_experiment_directory("cnn_gru_extreme_wmse", "wmse", 6),
        rollout_experiment_directory("cnn_gru_extreme_awmse", "awmse", 6),
    }
    assert len(rollout_names) == 3


def test_positive_metrics_are_distinct_from_legacy_absolute_metrics() -> None:
    observed = np.array([-1.0, -0.9, -0.8, 0.1, 0.2, 0.9])
    predicted = np.array([-1.0, -0.9, -0.8, 0.1, 0.2, 0.7])
    times = pd.date_range("1996-01-01", periods=len(observed), freq="1h")
    metrics = calculate_detailed_metrics(observed, predicted, times)
    assert metrics["top_absolute_5_percent"]["threshold_cm"] > 90
    assert metrics["positive_extreme_p95"]["threshold_cm"] > 0
    assert metrics["positive_extreme_p95"]["bias_cm"] < 0
    assert metrics["positive_peak_events"]["event_selection"] == (
        "largest positive observed peaks"
    )
    assert "peak_events" in metrics


def test_extra_checkpoint_parser_requires_unique_labels() -> None:
    parsed = parse_extra_checkpoints(["wmse=models/wmse.pth", "rollout-6=models/r6.pth"])
    assert set(parsed) == {"wmse", "rollout-6"}
    with pytest.raises(ValueError, match="Duplicate"):
        parse_extra_checkpoints(["wmse=a", "wmse=b"])
    with pytest.raises(ValueError, match="LABEL=PATH"):
        parse_extra_checkpoints(["missing_separator"])


def test_validation_summary_uses_common_timestamps_and_writes_expected_metrics(
    tmp_path: Path,
) -> None:
    times = pd.date_range("1996-01-01", periods=200, freq="1h")
    observed = np.sin(np.arange(200) / 10) * 0.2
    model_root = tmp_path / "formal_seed42"
    for directory, offset in (
        ("cnn_gru", -0.02),
        ("cnn_gru_extreme_wmse", -0.01),
        ("cnn_gru_extreme_awmse", 0.0),
    ):
        path = model_root / directory
        path.mkdir(parents=True)
        pd.DataFrame(
            {
                "datetime": times,
                "observed_m": observed,
                "predicted_m": observed + offset,
            }
        ).to_csv(path / "validation_predictions.csv", index=False)
    with pytest.warns(UserWarning, match="whuber"):
        common, sources = collect_common_predictions(model_root, 1996)
    summary, details = summarise_predictions(common)
    assert len(common) == 200
    assert set(summary.experiment) == {"mse", "wmse", "awmse"}
    assert "positive_p99_rmse_cm" in summary
    assert "positive_peak_event_rows" in details
    assert set(sources) == {"mse", "wmse", "awmse"}
    figure = tmp_path / "comparison.png"
    plot_summary(summary, figure, "Synthetic station")
    assert figure.is_file() and figure.stat().st_size > 0


def test_loss_metadata_is_json_serialisable() -> None:
    _, metadata = build_training_loss(
        "whuber", np.linspace(-0.2, 0.6, 100), 0.1, 0.2
    )
    encoded = json.dumps(metadata)
    assert "huber_delta_cm" in encoded
    assert "q99_m" in encoded
