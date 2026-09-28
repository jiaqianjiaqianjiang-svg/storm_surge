from pathlib import Path
import sys

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from compare_station_core_models import build_comparison, write_outputs


def _metrics(path: Path, name_column: str, names: list[str]) -> None:
    rows = []
    for lead in (1, 24, 72):
        for index, name in enumerate(names):
            rows.append({
                "lead_hours": lead, name_column: name,
                "rmse_cm": 2.0 + index, "rrmse_percent": 20.0 + index,
                "pearson_r": 0.9 - index / 100,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_aligned_station_table_contains_all_core_models(tmp_path: Path) -> None:
    xiamen = tmp_path / "xiamen.csv"
    prickly = tmp_path / "prickly.csv"
    _metrics(xiamen, "model", ["persistence", "ridge", "cnn", "cnn_gru", "cnn_gru_rollout6"])
    _metrics(prickly, "method", ["persistence", "ridge", "dual_cnn", "cnn_gru", "cnn_gru_rollout6"])
    frame = build_comparison(xiamen, prickly)
    assert len(frame) == 30
    assert frame.skill_score_vs_persistence.notna().all()
    output = tmp_path / "output"
    write_outputs(frame, output)
    assert (output / "core_model_comparison_long.csv").is_file()
    assert (output / "core_model_comparison.md").is_file()
