from pathlib import Path

from src.direct_forecast_models import DirectDualCNN, DirectGRU, DirectMLP
from src.evaluate_direct_2018 import model_from_checkpoint
from src.export_final_results import copy_result_files


def test_model_factory_reconstructs_direct_models():
    mlp = model_from_checkpoint(
        {"model_name": "DirectMLP", "input_features": 600, "output_steps": 24}
    )
    gru = model_from_checkpoint(
        {
            "model_name": "DirectGRU", "input_features": 14,
            "hidden_size": 32, "num_layers": 1, "output_steps": 24,
        }
    )
    dual = model_from_checkpoint(
        {
            "model_name": "DirectDualCNN", "atmosphere_steps": 24,
            "input_steps": 24, "variables": 3, "grid_size": 40,
            "output_steps": 24,
        }
    )
    assert isinstance(mlp, DirectMLP)
    assert isinstance(gru, DirectGRU)
    assert isinstance(dual, DirectDualCNN)


def test_git_safe_export_omits_predictions_and_weights(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    for name in (
        "metrics.csv", "metadata.json", "figure.png", "report.md",
        "test_predictions.csv", "best_model.pth", "model.joblib",
    ):
        (source / name).write_text("test", encoding="utf-8")
    copied = copy_result_files(source, destination)
    assert copied == ["figure.png", "metadata.json", "metrics.csv", "report.md"]
    assert not (destination / "test_predictions.csv").exists()
    assert not (destination / "best_model.pth").exists()
