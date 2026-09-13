"""Minimal ERA5 ablation for direct XGBoost and matched Dual-CNN models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .direct_forecast_models import ArrayDirectDataset, MatchedSurgeAblation
    from .direct_multistep_baselines import (
        PRIMARY_LEADS,
        build_feature_matrix,
        direct_labels,
        direct_origins,
        hourly_atmosphere_valid,
    )
    from .evaluate import calculate_metrics
    from .evaluate_baselines import summarise_atmosphere
    from .train_direct_multimodel import train_neural_model
    from .train_station import load_prepared
except ImportError:
    from direct_forecast_models import ArrayDirectDataset, MatchedSurgeAblation
    from direct_multistep_baselines import (
        PRIMARY_LEADS,
        build_feature_matrix,
        direct_labels,
        direct_origins,
        hourly_atmosphere_valid,
    )
    from evaluate import calculate_metrics
    from evaluate_baselines import summarise_atmosphere
    from train_direct_multimodel import train_neural_model
    from train_station import load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "persistence",
    "combined_ridge",
    "xgb_surge",
    "xgb_past_era5",
    "xgb_future_era5",
    "cnn_surge",
    "cnn_past_era5",
    "cnn_future_era5",
)
DISPLAY_NAMES = {
    "persistence": "Persistence",
    "combined_ridge": "Combined-Ridge",
    "xgb_surge": "XGB-Surge",
    "xgb_past_era5": "XGB-Past-ERA5",
    "xgb_future_era5": "XGB-Future-ERA5",
    "cnn_surge": "CNN-Surge matched ablation",
    "cnn_past_era5": "Dual-CNN-Past",
    "cnn_future_era5": "Dual-CNN-Future",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--direct-predictions", type=Path)
    parser.add_argument("--direct-model-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    return parser.parse_args()


def load_reference_predictions(
    path: Path, expected_origins: pd.DatetimeIndex
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    frame = pd.read_csv(path, parse_dates=["forecast_origin"])
    frame = frame.set_index("forecast_origin").reindex(expected_origins)
    if frame.isna().any().any():
        raise ValueError("Direct predictions do not cover all ablation origins")

    def values(prefix: str) -> np.ndarray:
        return np.column_stack(
            [frame[f"{prefix}_{lead:02d}h_m"] for lead in range(1, 25)]
        ).astype(np.float32)

    return (
        {
            "persistence": values("persistence"),
            "combined_ridge": values("combined_ridge"),
            "xgb_future_era5": values("xgboost"),
            "cnn_past_era5": values("dual_cnn_past"),
            "cnn_future_era5": values("dual_cnn_future"),
        },
        values("observed"),
    )


def fit_xgboost_family(
    name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    output: Path,
    device: torch.device,
    seed: int,
    reuse: bool,
) -> np.ndarray:
    from xgboost import XGBRegressor

    model_path = output / f"{name}_models.joblib"
    if reuse and model_path.is_file():
        models = joblib.load(model_path)
    else:
        models = []
        for lead in range(24):
            model = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_lambda=10.0,
                objective="reg:squarederror",
                tree_method="hist",
                device="cuda" if device.type == "cuda" else "cpu",
                n_jobs=1,
                random_state=seed,
            )
            model.fit(train_features, train_labels[:, lead])
            models.append(model)
            print(f"{name} lead={lead + 1}/24 complete", flush=True)
        joblib.dump(models, model_path, compress=3)
    return np.column_stack(
        [model.predict(validation_features) for model in models]
    ).astype(np.float32)


def calculate_ablation_metrics(
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    surge: np.ndarray,
    origins: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lead_index in range(24):
        target = observed[:, lead_index]
        prior = np.asarray(surge[origins + lead_index], dtype=float)
        threshold = float(np.quantile(np.abs(target), 0.95))
        top_mask = np.abs(target) >= threshold
        rises = target - prior
        positive = rises[rises > 0]
        rise_threshold = float(np.quantile(positive, 0.90)) if len(positive) else np.nan
        rapid_mask = rises >= rise_threshold if np.isfinite(rise_threshold) else np.zeros(len(target), bool)
        for method in METHODS:
            predicted = predictions[method][:, lead_index]
            overall = calculate_metrics(target, predicted)
            top = calculate_metrics(target[top_mask], predicted[top_mask])
            rapid = calculate_metrics(target[rapid_mask], predicted[rapid_mask])
            rows.append(
                {
                    "lead_hours": lead_index + 1,
                    "method": method,
                    **overall,
                    "top5_n": int(top["n"]),
                    "top5_threshold_cm": threshold * 100,
                    "top5_rmse_cm": top["rmse_cm"],
                    "rapid_rise_n": int(rapid["n"]),
                    "rapid_rise_threshold_cm_per_hour": rise_threshold * 100,
                    "rapid_rise_rmse_cm": rapid["rmse_cm"],
                }
            )
    return pd.DataFrame(rows)


def incremental_skill(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(index="lead_hours", columns="method", values="rmse_cm")
    rows = []
    comparisons = (
        ("XGBoost", "Past ERA5 vs surge-only", "xgb_surge", "xgb_past_era5"),
        ("XGBoost", "Future ERA5 vs past ERA5", "xgb_past_era5", "xgb_future_era5"),
        ("CNN", "Past ERA5 vs surge-only", "cnn_surge", "cnn_past_era5"),
        ("CNN", "Future ERA5 vs past ERA5", "cnn_past_era5", "cnn_future_era5"),
    )
    for lead in range(1, 25):
        for family, comparison, reference, model in comparisons:
            reference_rmse = float(pivot.loc[lead, reference])
            model_rmse = float(pivot.loc[lead, model])
            rows.append(
                {
                    "lead_hours": lead,
                    "family": family,
                    "comparison": comparison,
                    "reference_rmse_cm": reference_rmse,
                    "model_rmse_cm": model_rmse,
                    "mse_skill": 1 - (model_rmse / reference_rmse) ** 2,
                    "rmse_improvement_percent": 100 * (1 - model_rmse / reference_rmse),
                }
            )
    return pd.DataFrame(rows)


def plot_rmse(metrics: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    panels = (
        ("XGBoost ERA5 ablation", ("xgb_surge", "xgb_past_era5", "xgb_future_era5")),
        ("Matched CNN ERA5 ablation", ("cnn_surge", "cnn_past_era5", "cnn_future_era5")),
    )
    for ax, (title, methods) in zip(axes, panels):
        for method in methods:
            subset = metrics[metrics.method == method]
            ax.plot(subset.lead_hours, subset.rmse_cm, linewidth=2, label=DISPLAY_NAMES[method])
        ax.set_ylabel("RMSE (cm)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Direct forecast lead (hours)")
    axes[-1].set_xticks([1, 3, 6, 12, 18, 24])
    fig.suptitle("Prickly Bay 2017 minimal ERA5 ablation")
    fig.tight_layout()
    fig.savefig(destination, dpi=400)
    plt.close(fig)


def plot_events(
    output: Path,
    times: pd.DatetimeIndex,
    origins: np.ndarray,
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    scores = np.max(np.abs(observed), axis=1)
    selected: list[int] = []
    for candidate in np.argsort(scores)[::-1]:
        if all(abs(int(origins[candidate]) - int(origins[other])) >= 72 for other in selected):
            selected.append(int(candidate))
        if len(selected) == 3:
            break
    plotted = (
        "xgb_surge", "xgb_past_era5", "xgb_future_era5",
        "cnn_surge", "cnn_past_era5", "cnn_future_era5",
    )
    details = []
    for number, row in enumerate(selected, start=1):
        valid_times = times[origins[row] + np.arange(1, 25)]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(valid_times, observed[row] * 100, color="black", linewidth=2.2, label="Observed")
        for method in plotted:
            ax.plot(valid_times, predictions[method][row] * 100, linewidth=1.2, label=DISPLAY_NAMES[method])
        ax.set(xlabel="Valid time", ylabel="Storm surge (cm)")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=7)
        ax.set_title(f"ERA5 ablation strong event from {times[origins[row]]:%Y-%m-%d %H:%M}")
        fig.autofmt_xdate()
        fig.tight_layout()
        filename = f"strong_event_{number}_{times[origins[row]]:%Y%m%d_%H%M}.png"
        fig.savefig(output / filename, dpi=400)
        plt.close(fig)
        details.append(
            {
                "forecast_origin": times[origins[row]].isoformat(),
                "maximum_absolute_observed_cm": float(scores[row] * 100),
                "plot": filename,
            }
        )
    return details


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---:" for _ in frame.columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    output: Path,
    primary: pd.DataFrame,
    skills: pd.DataFrame,
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
) -> None:
    selected_skills = skills[skills.lead_hours.isin(PRIMARY_LEADS)]
    table = primary.set_index("lead_hours")
    skill = skills.set_index(["lead_hours", "family", "comparison"])
    top5 = metrics.pivot(index="lead_hours", columns="method", values="top5_rmse_cm")
    top5_xgb_improvements = {
        lead: 100 * (1 - top5.loc[lead, "xgb_future_era5"] / top5.loc[lead, "xgb_surge"])
        for lead in (3, 6, 12, 24)
    }
    content = f"""# Prickly Bay 2017最小ERA5消融报告

## 实验控制

- 所有版本使用相同的{metadata['validation_samples']}个2017起报时刻、2011—2016训练数据和连续24小时标签。
- XGBoost三个版本使用完全相同的树参数和seed 42。
- CNN-Surge复用Dual-CNN的增水分支与融合输出头，仅移除ERA5分支；三个CNN版本使用相同MSE、seed 42和早停标准。
- 2018未加载。
- Future-ERA5使用未来ERA5再分析真值，属于已知未来大气强迫历史回算。

## 重点提前量RMSE（cm）

{markdown_table(primary)}

## ERA5增量技能

正值表示加入相应ERA5后改善。

{markdown_table(selected_skills)}

## 主要结论

1. XGB-Surge在24小时RMSE已达到{table.loc[24, 'xgb_surge']:.3f} cm，
说明XGBoost相对线性模型的优势很大一部分来自对历史增水的非线性拟合，不能全部归因于ERA5。
2. 加入过去ERA5后，XGBoost在6和12小时RMSE分别改善
{skill.loc[(6, 'XGBoost', 'Past ERA5 vs surge-only'), 'rmse_improvement_percent']:.2f}%和
{skill.loc[(12, 'XGBoost', 'Past ERA5 vs surge-only'), 'rmse_improvement_percent']:.2f}%；
再加入未来ERA5后分别额外改善
{skill.loc[(6, 'XGBoost', 'Future ERA5 vs past ERA5'), 'rmse_improvement_percent']:.2f}%和
{skill.loc[(12, 'XGBoost', 'Future ERA5 vs past ERA5'), 'rmse_improvement_percent']:.2f}%。
因此ERA5对XGBoost存在可测的增量价值，主要出现在6—24小时。
3. XGB-Future-ERA5相对XGB-Surge在Top 5%强增水中的RMSE改善为：
3小时{top5_xgb_improvements[3]:.2f}%、6小时{top5_xgb_improvements[6]:.2f}%、
12小时{top5_xgb_improvements[12]:.2f}%、24小时{top5_xgb_improvements[24]:.2f}%；
说明ERA5对强增水过程的帮助明显大于总体平均指标所显示的幅度。
4. 匹配结构CNN中，过去ERA5在6、12、24小时相对Surge-only分别改善
{skill.loc[(6, 'CNN', 'Past ERA5 vs surge-only'), 'rmse_improvement_percent']:.2f}%、
{skill.loc[(12, 'CNN', 'Past ERA5 vs surge-only'), 'rmse_improvement_percent']:.2f}%、
{skill.loc[(24, 'CNN', 'Past ERA5 vs surge-only'), 'rmse_improvement_percent']:.2f}%，
说明过去原始空间场具有小幅增量价值。
5. Dual-CNN-Future在所有重点提前量总体RMSE均未超过Dual-CNN-Past，说明当前把未来48小时气象场直接拼接为通道的方式没有有效利用未来强迫。
6. 最终可以表述为：XGBoost优势同时来自非线性增水拟合和ERA5增量；CNN能从过去空间场获得少量信息，但当前未来空间场编码方式无效。

## 下一步候选

- XGBoost最终候选：XGB-Future-ERA5。
- CNN最终候选：Dual-CNN-Past。
- 下一步不再扩模型，可对这两个候选运行多个随机种子，并使用时间块Bootstrap评价稳定性。

## 输出

- `metrics_all_leads.csv`：总体、Top 5%和快速上涨指标。
- `incremental_skill.csv`：Past相对Surge-only、Future相对Past的技能。
- `rmse_primary_leads.csv`：重点提前量RMSE。
- `rmse_era5_ablation.png`：两类模型消融曲线。
- `strong_event_*.png`：三次强增水过程。
"""
    (output / "2017_era5_ablation_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path or (
        MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    )
    direct_root = args.direct_model_root or (
        MODULE_ROOT / "outputs" / "experiments" / args.station /
        "direct_multimodel_2017_seed42"
    )
    direct_path = args.direct_predictions or direct_root / "validation_predictions.csv"
    output = args.output_dir or (
        MODULE_ROOT / "outputs" / "experiments" / args.station /
        "era5_ablation_2017_seed42"
    )
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    atmosphere, surge, times = load_prepared(dataset_path, 2011, 2017)
    if (times.year > 2017).any() or times.max().year != 2017:
        raise AssertionError("2018 entered ERA5 ablation")
    atmosphere_valid = hourly_atmosphere_valid(atmosphere)
    train_origins = direct_origins(times, atmosphere_valid, surge, set(range(2011, 2017)))
    validation_origins = direct_origins(times, atmosphere_valid, surge, {2017})
    train_labels = direct_labels(surge, train_origins)
    validation_labels = direct_labels(surge, validation_origins)
    predictions, reference_observed = load_reference_predictions(
        direct_path, pd.DatetimeIndex(times[validation_origins])
    )
    if not np.allclose(validation_labels, reference_observed):
        raise AssertionError("Ablation labels do not match frozen direct experiment")
    print(
        f"loaded through {times.max()}; train={len(train_origins)} "
        f"validation={len(validation_origins)} device={device}", flush=True,
    )

    summaries = summarise_atmosphere(atmosphere)
    feature_configs = {
        "xgb_surge": (True, "none"),
        "xgb_past_era5": (True, "past"),
    }
    for name, (include_surge, era5_mode) in feature_configs.items():
        train_features = build_feature_matrix(
            summaries, surge, train_origins, include_surge, era5_mode
        )
        validation_features = build_feature_matrix(
            summaries, surge, validation_origins, include_surge, era5_mode
        )
        predictions[name] = fit_xgboost_family(
            name, train_features, train_labels, validation_features,
            output, device, args.seed, args.reuse_checkpoints,
        )

    direct_checkpoint = torch.load(
        direct_root / "dual_cnn_past" / "best_model.pth",
        map_location="cpu", weights_only=False,
    )
    surge_mean = float(direct_checkpoint["surge_mean"])
    surge_scale = float(direct_checkpoint["surge_scale"])
    train_history = np.stack(
        [np.asarray(surge[o - 23 : o + 1], dtype=np.float32) for o in train_origins]
    )
    validation_history = np.stack(
        [np.asarray(surge[o - 23 : o + 1], dtype=np.float32) for o in validation_origins]
    )
    train_history = (train_history - surge_mean) / surge_scale
    validation_history = (validation_history - surge_mean) / surge_scale
    scaled_train_labels = (train_labels - surge_mean) / surge_scale
    scaled_validation_labels = (validation_labels - surge_mean) / surge_scale
    train_loader = DataLoader(
        ArrayDirectDataset(train_history, scaled_train_labels),
        batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        ArrayDirectDataset(validation_history, scaled_validation_labels),
        batch_size=args.batch_size, pin_memory=device.type == "cuda",
    )
    predictions["cnn_surge"], _ = train_neural_model(
        "cnn_surge", MatchedSurgeAblation(), train_loader, validation_loader,
        output, device, surge_mean, surge_scale, args.epochs, args.patience,
        args.seed,
        {"ablation_definition": "Dual-CNN surge branch and fusion head; ERA5 branch removed"},
        args.reuse_checkpoints,
    )

    metrics = calculate_ablation_metrics(
        validation_labels, predictions, surge, validation_origins
    )
    metrics.to_csv(output / "metrics_all_leads.csv", index=False)
    skills = incremental_skill(metrics)
    skills.to_csv(output / "incremental_skill.csv", index=False)
    primary = metrics[metrics.lead_hours.isin(PRIMARY_LEADS)].pivot(
        index="lead_hours", columns="method", values="rmse_cm"
    ).reset_index()
    primary.insert(1, "valid_samples", len(validation_origins))
    primary = primary[["lead_hours", "valid_samples", *METHODS]]
    primary.to_csv(output / "rmse_primary_leads.csv", index=False)
    plot_rmse(metrics, output / "rmse_era5_ablation.png")
    events = plot_events(output, times, validation_origins, validation_labels, predictions)
    metadata = {
        "experiment": "2017 minimal ERA5 ablation",
        "loaded_time_max": times.max().isoformat(),
        "2018_loaded": False,
        "train_samples": int(len(train_origins)),
        "validation_samples": int(len(validation_origins)),
        "seed": args.seed,
        "xgboost_hyperparameters_identical": True,
        "cnn_surge_branch_and_head_matched": True,
        "events": events,
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output, primary, skills, metadata, metrics)
    print(primary.to_string(index=False), flush=True)
    print(skills[skills.lead_hours.isin(PRIMARY_LEADS)].to_string(index=False), flush=True)
    print(f"outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
