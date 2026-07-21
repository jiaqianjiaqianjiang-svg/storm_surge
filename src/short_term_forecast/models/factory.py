"""模型工厂，保证训练和滚动预报使用同一套模型名。"""

from __future__ import annotations

from torch import nn

from models.cnn import ForecastCNN
from models.cnn_gru import CNNGRUForecastModel
from models.cnn_lstm import CNNLSTMForecastModel
from models.tcn import TCNForecastModel
from models.transformer import SimpleTransformerForecastModel


MODEL_CHOICES = ("cnn", "cnn_lstm", "cnn_gru", "tcn", "transformer")


def build_model(model_name: str, input_steps: int, n_variables: int = 3, grid_size: int = 40) -> nn.Module:
    """根据模型名创建短时预报模型。"""

    if model_name == "cnn":
        return ForecastCNN(input_steps=input_steps, n_variables=n_variables, grid_size=grid_size)
    if model_name == "cnn_lstm":
        return CNNLSTMForecastModel(input_steps=input_steps, n_variables=n_variables, grid_size=grid_size)
    if model_name == "cnn_gru":
        return CNNGRUForecastModel(input_steps=input_steps, n_variables=n_variables, grid_size=grid_size)
    if model_name == "tcn":
        return TCNForecastModel(input_steps=input_steps, n_variables=n_variables, grid_size=grid_size)
    if model_name == "transformer":
        return SimpleTransformerForecastModel(input_steps=input_steps, n_variables=n_variables, grid_size=grid_size)
    raise ValueError(f"未知模型: {model_name}，可选 {MODEL_CHOICES}")
