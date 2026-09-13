import numpy as np

from src.rolling_72_comparison import build_metrics


def test_unified_metrics_use_identical_origins_and_same_lead_targets():
    surge = np.linspace(-0.1, 0.2, 180, dtype=np.float32)
    origins = np.asarray([30, 31, 32, 33])
    predictions = {
        name: np.stack([surge[origins + lead] for lead in range(72)], axis=1)
        for name in ("persistence", "ridge", "xgboost", "dual_cnn")
    }
    frame = build_metrics(surge, origins, predictions)
    assert len(frame) == 7 * 4
    assert np.allclose(frame.rmse_cm, 0.0)
