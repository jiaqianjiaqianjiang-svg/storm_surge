import numpy as np
import pandas as pd

from src.era5_ablation import incremental_skill


def test_incremental_skill_uses_mse_ratio_and_rmse_change():
    rows = []
    values = {
        "xgb_surge": 2.0,
        "xgb_past_era5": 1.8,
        "xgb_future_era5": 1.6,
        "cnn_surge": 3.0,
        "cnn_past_era5": 2.7,
        "cnn_future_era5": 2.4,
    }
    for lead in range(1, 25):
        rows.extend(
            {"lead_hours": lead, "method": method, "rmse_cm": rmse}
            for method, rmse in values.items()
        )
    result = incremental_skill(pd.DataFrame(rows))
    xgb_past = result[
        (result.lead_hours == 1)
        & (result.family == "XGBoost")
        & (result.comparison == "Past ERA5 vs surge-only")
    ].iloc[0]
    assert np.isclose(xgb_past.mse_skill, 1 - (1.8 / 2.0) ** 2)
    assert np.isclose(xgb_past.rmse_improvement_percent, 10.0)
