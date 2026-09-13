import numpy as np
import pandas as pd
import torch

from src.forecast_model import CaribbeanSurgeCNN
from src.train_rollout_dual import rollout_forward, rollout_origins


def test_rollout_origins_require_complete_history_and_targets():
    times = pd.date_range("2017-01-01", periods=80, freq="h")
    valid = np.ones(80, dtype=bool)
    surge = np.ones(80, dtype=np.float32)
    surge[30] = np.nan
    origins = rollout_origins(times, valid, surge, {2017}, 6)
    assert 30 not in origins
    assert 31 not in origins
    assert 55 in origins  # the 24-hour history no longer includes index 30


def test_rollout_forward_returns_every_unrolled_step():
    model = CaribbeanSurgeCNN()
    weather = torch.randn(2, 3, 128)
    history = torch.randn(2, 24)
    targets = torch.randn(2, 3)
    predicted = rollout_forward(model, weather, history, targets, 0.0)
    assert predicted.shape == (2, 3)
