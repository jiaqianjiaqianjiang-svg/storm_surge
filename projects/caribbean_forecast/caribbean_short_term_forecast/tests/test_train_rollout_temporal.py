import torch

from src.forecast_model import CNNGRUForecastModel
from src.train_rollout_temporal import (
    scheduled_teacher_ratio,
    temporal_rollout_forward,
)


def test_temporal_rollout_returns_each_recursive_step() -> None:
    model = CNNGRUForecastModel(6, ("U10", "V10", "MSL"), 40)
    weather = torch.randn(2, 3, 6, 64)
    history = torch.randn(2, 6)
    targets = torch.randn(2, 3)
    predicted = temporal_rollout_forward(model, weather, history, targets, 0.0)
    assert predicted.shape == (2, 3)


def test_teacher_forcing_schedule_reaches_recursive_training() -> None:
    assert scheduled_teacher_ratio(1, 5) == 1.0
    assert scheduled_teacher_ratio(5, 5) == 0.0
    assert scheduled_teacher_ratio(10, 5) == 0.0
