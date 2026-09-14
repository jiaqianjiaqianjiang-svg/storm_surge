"""厦门站短时风暴潮预报模型集合。"""

from models.cnn import ForecastCNN
from models.cnn_gru import CNNGRUForecastModel
from models.cnn_lstm import CNNLSTMForecastModel
from models.factory import MODEL_CHOICES, build_model
from models.tcn import TCNForecastModel
from models.transformer import SimpleTransformerForecastModel

__all__ = [
    "ForecastCNN",
    "CNNLSTMForecastModel",
    "CNNGRUForecastModel",
    "TCNForecastModel",
    "SimpleTransformerForecastModel",
    "MODEL_CHOICES",
    "build_model",
]
