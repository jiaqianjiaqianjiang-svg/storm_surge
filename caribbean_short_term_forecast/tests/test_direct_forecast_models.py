import numpy as np
import torch

from src.direct_forecast_models import DirectDualCNN, DirectGRU, DirectMLP, GridDirectDataset


def test_direct_model_output_shapes():
    assert DirectMLP(20)(torch.zeros(2, 20)).shape == (2, 24)
    assert DirectGRU()(torch.zeros(2, 48, 14)).shape == (2, 24)
    assert DirectDualCNN(24)(torch.zeros(2, 72, 40, 40), torch.zeros(2, 24)).shape == (2, 24)
    assert DirectDualCNN(48)(torch.zeros(2, 144, 40, 40), torch.zeros(2, 24)).shape == (2, 24)


def test_grid_dataset_past_and_future_boundaries():
    atmosphere = np.arange(70 * 3 * 2 * 2, dtype=np.float32).reshape(70, 3, 2, 2)
    surge = np.arange(70, dtype=np.float32)
    origins = np.asarray([30])
    labels = np.arange(31, 55, dtype=np.float32)[None, :]
    common = dict(
        atmosphere=atmosphere,
        surge=surge,
        origins=origins,
        labels=labels,
        atmosphere_mean=np.zeros(3),
        atmosphere_scale=np.ones(3),
        surge_mean=0.0,
        surge_scale=1.0,
        input_steps=24,
        output_steps=24,
    )
    past_weather, history, target = GridDirectDataset(include_future=False, **common)[0]
    future_weather, _, _ = GridDirectDataset(include_future=True, **common)[0]
    assert past_weather.shape == (72, 2, 2)
    assert future_weather.shape == (144, 2, 2)
    assert history.tolist() == list(range(7, 31))
    assert target.tolist() == list(range(31, 55))
