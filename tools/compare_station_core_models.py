"""Build aligned Xiamen and Prickly Bay core-model comparison tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LEADS = (1, 24, 72)
MODELS = ("Persistence", "Ridge", "Fusion CNN", "CNN-GRU", "Rollout-6")
SOURCE_NAMES = {
    "Xiamen": {
        "persistence": "Persistence", "ridge": "Ridge", "cnn": "Fusion CNN",
        "cnn_gru": "CNN-GRU", "cnn_gru_rollout6": "Rollout-6",
    },
    "Prickly Bay": {
        "persistence": "Persistence", "ridge": "Ridge", "dual_cnn": "Fusion CNN",
        "cnn_gru": "CNN-GRU", "cnn_gru_rollout6": "Rollout-6",
    },
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xiamen-metrics", type=Path,
        default=root / "reports" / "experiment_results" /
        "xiamen_short_term_1997_seed42" / "rolling_metrics_long.csv",
    )
    parser.add_argument(
        "--prickly-metrics", type=Path,
        default=root / "reports" / "experiment_results" /
        "prickly_bay_core_aligned_2018_seed42" / "rolling_72h" /
        "metrics_selected_leads.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "reports" / "experiment_results" / "station_core_comparison",
    )
    return parser.parse_args()


def normalise_station_metrics(
    path: Path, station: str, source_column: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {station} metrics: {path}")
    frame = pd.read_csv(path)
    required = {"lead_hours", source_column, "rmse_cm", "rrmse_percent", "pearson_r"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[frame.lead_hours.isin(LEADS)].copy()
    frame["model"] = frame[source_column].map(SOURCE_NAMES[station])
    frame = frame[frame.model.notna()].copy()
    frame["station"] = station
    persistence = (
        frame[frame.model == "Persistence"]
        .set_index("lead_hours")["rmse_cm"]
    )
    frame["skill_score_vs_persistence"] = [
        1 - (row.rmse_cm / persistence.loc[row.lead_hours]) ** 2
        for row in frame.itertuples()
    ]
    frame["rmse_reduction_vs_persistence_percent"] = [
        100 * (1 - row.rmse_cm / persistence.loc[row.lead_hours])
        for row in frame.itertuples()
    ]
    return frame[[
        "station", "lead_hours", "model", "rmse_cm", "rrmse_percent",
        "pearson_r", "skill_score_vs_persistence",
        "rmse_reduction_vs_persistence_percent",
    ]]


def build_comparison(xiamen_path: Path, prickly_path: Path) -> pd.DataFrame:
    frame = pd.concat([
        normalise_station_metrics(xiamen_path, "Xiamen", "model"),
        normalise_station_metrics(prickly_path, "Prickly Bay", "method"),
    ], ignore_index=True)
    frame["model"] = pd.Categorical(frame.model, MODELS, ordered=True)
    return frame.sort_values(["lead_hours", "station", "model"]).reset_index(drop=True)


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "# Aligned core-model comparison",
        "",
        "Skill score is `1 - model MSE / persistence MSE` at the same station and lead.",
    ]
    for lead in LEADS:
        lines.extend(["", f"## {lead} h", ""])
        subset = frame[frame.lead_hours == lead].copy()
        for column, title in (
            ("rmse_cm", "RMSE (cm)"),
            ("rrmse_percent", "RRMSE (%)"),
            ("pearson_r", "Pearson r"),
            ("skill_score_vs_persistence", "Skill vs persistence"),
        ):
            pivot = subset.pivot(index="station", columns="model", values=column)
            pivot = pivot.reindex(columns=MODELS)
            rounded = pivot.round(4)
            headers = ["Station", *rounded.columns.astype(str)]
            lines.extend([
                f"**{title}**", "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ])
            for station, row in rounded.iterrows():
                values = [station] + [
                    "" if pd.isna(value) else f"{value:.4f}" for value in row
                ]
                lines.append("| " + " | ".join(values) + " |")
            lines.append("")
    return "\n".join(lines)


def write_outputs(frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "core_model_comparison_long.csv", index=False)
    for lead in LEADS:
        frame[frame.lead_hours == lead].to_csv(
            output / f"core_model_comparison_{lead}h.csv", index=False
        )
    (output / "core_model_comparison.md").write_text(
        markdown_table(frame), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    frame = build_comparison(args.xiamen_metrics, args.prickly_metrics)
    write_outputs(frame, args.output_dir)
    expected = 2 * len(LEADS) * len(MODELS)
    if len(frame) != expected:
        missing = expected - len(frame)
        print(f"WARNING: comparison is incomplete ({missing} station/lead/model rows missing)")
    print(frame.to_string(index=False)); print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
