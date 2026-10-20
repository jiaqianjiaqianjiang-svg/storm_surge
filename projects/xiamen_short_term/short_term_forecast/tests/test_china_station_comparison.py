import json
from pathlib import Path
import sys

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from compare_china_short_term_stations import MODEL_COLORS, load_comparison, write_outputs
from src.short_term_forecast.journal_figures.style import model_color, model_line_style


STATIONS = ("xiamen", "lianyungang", "beihai")
ONE_STEP_MODELS = (
    "persistence", "ridge", "surge_mlp", "era5_cnn", "cnn",
    "cnn_lstm", "cnn_gru", "tcn", "transformer",
)
ROLLING_MODELS = (*ONE_STEP_MODELS, "cnn_gru_rollout6")


def test_station_and_journal_figures_share_model_identity() -> None:
    for model, color in MODEL_COLORS.items():
        assert model_color(model) == color
    assert model_line_style("cnn_gru") != model_line_style("cnn_gru_rollout6")
    assert model_line_style("persistence")["linestyle"] != model_line_style("cnn_gru")["linestyle"]


def _write_package(root: Path, station: str) -> None:
    package = root / f"{station}_short_term_1997_seed42"
    figures = package / "journal_figures"
    figures.mkdir(parents=True)
    pd.DataFrame([
        {
            "model": model, "n": 100, "pearson_r": 0.98,
            "r2": 0.95, "rmse_cm": 3.0 + index / 10,
            "mae_cm": 2.0 + index / 10, "bias_cm": 0.1,
            "rrmse_percent": 20.0 + index,
        }
        for index, model in enumerate(ONE_STEP_MODELS)
    ]).to_csv(package / "test_model_metrics.csv", index=False)
    pd.DataFrame([
        {
            "lead_hours": lead, "model": model, "n": 90,
            "pearson_r": 0.9, "r2": 0.8,
            "rmse_cm": 2.0 + lead / 10 + index / 10,
            "mae_cm": 1.0 + lead / 10, "bias_cm": 0.1,
            "rrmse_percent": 20.0 + lead / 10,
        }
        for lead in (1, 3, 6, 12, 24, 48, 72)
        for index, model in enumerate(ROLLING_MODELS)
    ]).to_csv(package / "rolling_metrics_long.csv", index=False)
    figures.joinpath("peak_metrics.json").write_text(json.dumps([
        {
            "percentile": percentile, "threshold_cm": percentile / 2,
            "n_peak": 10, "peak_rmse": percentile / 20,
            "peak_mae": percentile / 25, "peak_bias": -0.5,
            "underestimation_ratio": 0.6, "peak_correlation": 0.9,
        }
        for percentile in (90.0, 95.0, 99.0)
    ]), encoding="utf-8")


def test_three_station_comparison_writes_tables_and_figures(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    for station in STATIONS:
        _write_package(reports, station)

    one_step, rolling, peak, packages = load_comparison(reports)
    output = tmp_path / "comparison"
    summary = write_outputs(one_step, rolling, peak, packages, output)

    assert len(one_step) == 3 * len(ONE_STEP_MODELS)
    assert len(rolling) == 3 * 7 * len(ROLLING_MODELS)
    assert len(peak) == 9
    assert len(summary) == 3
    assert set(summary.station_id) == set(STATIONS)
    for name in (
        "china_three_station_summary.csv",
        "china_three_station_rolling_metrics.csv",
        "CHINA_THREE_STATION_COMPARISON.md",
        "fig01_one_step_rmse.png",
        "fig02_rolling_rmse.png",
        "fig03_p99_comparison.png",
        "figure_manifest.csv",
    ):
        assert (output / name).stat().st_size > 0
