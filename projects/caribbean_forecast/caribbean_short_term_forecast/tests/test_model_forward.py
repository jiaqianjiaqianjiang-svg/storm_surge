import torch

from caribbean_short_term_forecast.src.forecast_model import (
    CaribbeanSurgeCNN,
    ERA5OnlyCNN,
    SurgeHistoryMLP,
)


def test_model_forward_required_shape() -> None:
    model = CaribbeanSurgeCNN(input_steps=24, variables=("U10", "V10", "MSL"), grid_size=40)
    output = model(torch.randn(2, 72, 40, 40), torch.randn(2, 24))
    assert tuple(output.shape) == (2,)

def test_channels_follow_input_steps() -> None:
    model = CaribbeanSurgeCNN(input_steps=6, variables=("U10", "V10", "MSL"), grid_size=40)
    output = model(torch.randn(2, 18, 40, 40), torch.randn(2, 6))
    assert tuple(output.shape) == (2,)


def test_ablation_models_follow_common_forward_interface() -> None:
    atmosphere = torch.randn(2, 18, 40, 40)
    history = torch.randn(2, 6)
    weather_only = ERA5OnlyCNN(6, ("U10", "V10", "MSL"), 40)
    history_only = SurgeHistoryMLP(6, ("U10", "V10", "MSL"), 40)
    assert tuple(weather_only(atmosphere, history).shape) == (2,)
    assert tuple(history_only(atmosphere, history).shape) == (2,)
