"""Compare direct and recursive 24-hour forecasts on identical 2017 origins."""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .dataset_builder import valid_targets
    from .direct_multistep_baselines import (
        PRIMARY_LEADS,
        direct_labels,
        direct_origins,
        hourly_atmosphere_valid,
    )
    from .evaluate import calculate_metrics
    from .evaluate_baselines import build_ridge_features, summarise_atmosphere
    from .forecast_model import model_from_checkpoint
    from .train_station import load_prepared
except ImportError:
    from dataset_builder import valid_targets
    from direct_multistep_baselines import (
        PRIMARY_LEADS,
        direct_labels,
        direct_origins,
        hourly_atmosphere_valid,
    )
    from evaluate import calculate_metrics
    from evaluate_baselines import build_ridge_features, summarise_atmosphere
    from forecast_model import model_from_checkpoint
    from train_station import load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "persistence",
    "combined_ridge_direct",
    "combined_ridge_rolling",
    "xgboost_direct",
    "xgboost_rolling",
    "dual_cnn_past_direct",
    "dual_cnn_future_direct",
    "dual_cnn_rolling",
)
DISPLAY_NAMES = {
    "persistence": "Persistence",
    "combined_ridge_direct": "Ridge direct",
    "combined_ridge_rolling": "Ridge rolling",
    "xgboost_direct": "XGBoost direct",
    "xgboost_rolling": "XGBoost rolling",
    "dual_cnn_past_direct": "Dual-CNN direct past",
    "dual_cnn_future_direct": "Dual-CNN direct future",
    "dual_cnn_rolling": "Dual-CNN rolling",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--direct-predictions", type=Path)
    parser.add_argument("--ridge-model", type=Path)
    parser.add_argument("--dual-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reuse-xgboost", action="store_true")
    return parser.parse_args()


def recursive_tabular_predictions(
    model: Any,
    summaries: np.ndarray,
    initial_histories: np.ndarray,
    origins: np.ndarray,
    output_steps: int = 24,
) -> np.ndarray:
    """Roll one-step tabular model forward without future observed surge."""
    histories = np.asarray(initial_histories, dtype=np.float32).copy()
    weather_width = 24 * summaries.shape[1] * summaries.shape[2]
    predictions = np.empty((len(origins), output_steps), dtype=np.float32)
    features = np.empty((len(origins), weather_width + 24), dtype=np.float32)
    for lead in range(1, output_steps + 1):
        targets = origins + lead
        for row, target in enumerate(targets):
            features[row, :weather_width] = summaries[target - 24 : target].reshape(-1)
        features[:, weather_width:] = histories
        values = np.asarray(model.predict(features), dtype=np.float32).reshape(-1)
        predictions[:, lead - 1] = values
        histories[:, :-1] = histories[:, 1:]
        histories[:, -1] = values
    return predictions


def recursive_dual_predictions(
    atmosphere: np.ndarray,
    origins: np.ndarray,
    initial_histories: np.ndarray,
    checkpoint: dict[str, Any],
    device: torch.device,
    batch_size: int,
    output_steps: int = 24,
) -> np.ndarray:
    """Roll the seed-42 one-step Dual-CNN on the same direct origins."""
    model = model_from_checkpoint(checkpoint).to(device).eval()
    histories = np.asarray(initial_histories, dtype=np.float32).copy()
    predictions = np.empty((len(origins), output_steps), dtype=np.float32)
    atmosphere_mean = np.asarray(
        checkpoint["scalers"]["atmosphere"]["mean"], dtype=np.float32
    ).reshape(1, 1, -1, 1, 1)
    atmosphere_scale = np.asarray(
        checkpoint["scalers"]["atmosphere"]["scale"], dtype=np.float32
    ).reshape(1, 1, -1, 1, 1)
    surge_mean = float(checkpoint["scalers"]["surge"]["mean"][0])
    surge_scale = float(checkpoint["scalers"]["surge"]["scale"][0])
    with torch.inference_mode():
        for lead in range(1, output_steps + 1):
            targets = origins + lead
            for start in range(0, len(origins), batch_size):
                stop = min(len(origins), start + batch_size)
                batch_targets = targets[start:stop]
                raw_weather = np.stack(
                    [
                        np.asarray(atmosphere[target - 24 : target], dtype=np.float32)
                        for target in batch_targets
                    ]
                )
                scaled_weather = (raw_weather - atmosphere_mean) / atmosphere_scale
                weather = torch.from_numpy(
                    scaled_weather.reshape(
                        len(batch_targets), -1, atmosphere.shape[-2], atmosphere.shape[-1]
                    )
                ).to(device, non_blocking=True)
                history = torch.from_numpy(
                    (histories[start:stop] - surge_mean) / surge_scale
                ).to(device, non_blocking=True)
                # Keep full precision to reproduce the original rolling
                # diagnostic checkpoints exactly on their overlapping origins.
                scaled = model(weather, history)
                predictions[start:stop, lead - 1] = (
                    scaled.float().cpu().numpy() * surge_scale + surge_mean
                )
            histories[:, :-1] = histories[:, 1:]
            histories[:, -1] = predictions[:, lead - 1]
            print(f"dual rolling lead={lead}/24 complete", flush=True)
    return predictions


def load_direct_predictions(
    path: Path,
    expected_origins: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path, parse_dates=["forecast_origin"])
    frame = frame.set_index("forecast_origin").reindex(expected_origins)
    if frame.isna().any().any():
        raise ValueError("Direct predictions do not cover every common origin")

    def values(prefix: str) -> np.ndarray:
        return np.column_stack(
            [frame[f"{prefix}_{lead:02d}h_m"] for lead in range(1, 25)]
        ).astype(np.float32)

    return {
        "persistence": values("persistence"),
        "combined_ridge_direct": values("combined_ridge"),
        "xgboost_direct": values("xgboost"),
        "dual_cnn_past_direct": values("dual_cnn_past"),
        "dual_cnn_future_direct": values("dual_cnn_future"),
    }


def calculate_comparison_metrics(
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    surge: np.ndarray,
    origins: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lead_index in range(24):
        target = observed[:, lead_index]
        prior = np.asarray(surge[origins + lead_index], dtype=float)
        top_threshold = float(np.quantile(np.abs(target), 0.95))
        top_mask = np.abs(target) >= top_threshold
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
                    "top5_threshold_cm": top_threshold * 100,
                    "top5_rmse_cm": top["rmse_cm"],
                    "rapid_rise_n": int(rapid["n"]),
                    "rapid_rise_threshold_cm_per_hour": rise_threshold * 100,
                    "rapid_rise_rmse_cm": rapid["rmse_cm"],
                }
            )
    return pd.DataFrame(rows)


def method_change_table(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(index="lead_hours", columns="method", values="rmse_cm")
    rows = []
    pairs = {
        "Combined-Ridge": ("combined_ridge_direct", "combined_ridge_rolling"),
        "XGBoost": ("xgboost_direct", "xgboost_rolling"),
        "Dual-CNN (matched future forcing)": (
            "dual_cnn_future_direct", "dual_cnn_rolling"
        ),
        "Dual-CNN (best direct past)": (
            "dual_cnn_past_direct", "dual_cnn_rolling"
        ),
    }
    for lead in PRIMARY_LEADS:
        for model, (direct, rolling) in pairs.items():
            direct_rmse = float(pivot.loc[lead, direct])
            rolling_rmse = float(pivot.loc[lead, rolling])
            rows.append(
                {
                    "lead_hours": lead,
                    "model": model,
                    "direct_rmse_cm": direct_rmse,
                    "rolling_rmse_cm": rolling_rmse,
                    "rolling_minus_direct_percent": 100 * (rolling_rmse / direct_rmse - 1),
                }
            )
    return pd.DataFrame(rows)


def plot_rmse(metrics: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    panels = (
        ("Combined-Ridge", "combined_ridge_direct", "combined_ridge_rolling"),
        ("XGBoost", "xgboost_direct", "xgboost_rolling"),
        ("Dual-CNN", "dual_cnn_future_direct", "dual_cnn_rolling"),
    )
    for ax, (title, direct, rolling) in zip(axes, panels):
        for method, style in ((direct, "-"), (rolling, "--")):
            subset = metrics[metrics.method == method]
            ax.plot(
                subset.lead_hours, subset.rmse_cm, style,
                linewidth=2, label=DISPLAY_NAMES[method],
            )
        persistence = metrics[metrics.method == "persistence"]
        ax.plot(
            persistence.lead_hours, persistence.rmse_cm,
            color="grey", linewidth=1.2, alpha=0.7, label="Persistence",
        )
        ax.set_ylabel("RMSE (cm)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Forecast lead (hours)")
    axes[-1].set_xticks([1, 3, 6, 12, 18, 24])
    fig.suptitle("2017 direct versus rolling forecast on identical origins")
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
        "combined_ridge_direct", "combined_ridge_rolling",
        "xgboost_direct", "xgboost_rolling",
        "dual_cnn_future_direct", "dual_cnn_rolling",
    )
    details = []
    for number, row in enumerate(selected, start=1):
        valid_times = times[origins[row] + np.arange(1, 25)]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(valid_times, observed[row] * 100, color="black", linewidth=2.2, label="Observed")
        for method in plotted:
            ax.plot(valid_times, predictions[method][row] * 100, linewidth=1.25, label=DISPLAY_NAMES[method])
        ax.set(xlabel="Valid time", ylabel="Storm surge (cm)")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=7)
        ax.set_title(f"Direct vs rolling from {times[origins[row]]:%Y-%m-%d %H:%M}")
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


def write_report(output: Path, primary: pd.DataFrame, changes: pd.DataFrame, metadata: dict[str, Any]) -> None:
    table = primary.set_index("lead_hours")
    change = changes.set_index(["lead_hours", "model"])
    content = f"""# Prickly Bay 2017直接与滚动24小时统一比较

## 实验边界

- 所有方法严格使用相同的{metadata['common_origin_count']}个2017起报时刻。
- 训练数据为2011—2016，2018未加载。
- 滚动模型每一步只回填自身预测增水，不使用未来真实增水。
- 滚动过程中读取各目标时次ERA5再分析真值，属于已知未来大气强迫历史回算，不是业务预报。
- Ridge和XGBoost的直接版本使用过去＋未来ERA5统计特征。
- Dual-CNN滚动与Dual-CNN-Future直接版本信息条件匹配；Dual-CNN-Past作为此前最优直接空间模型另外保留。

## 重点提前量RMSE（cm）

{markdown_table(primary)}

## 滚动相对直接的变化

正值表示滚动RMSE更高，负值表示滚动更低。

{markdown_table(changes)}

## 主要结论

1. XGBoost直接预测在12和24小时保持最优，RMSE分别为
{table.loc[12, 'xgboost_direct']:.3f}和{table.loc[24, 'xgboost_direct']:.3f} cm。
滚动后分别升至{table.loc[12, 'xgboost_rolling']:.3f}和
{table.loc[24, 'xgboost_rolling']:.3f} cm，即恶化
{change.loc[(12, 'XGBoost'), 'rolling_minus_direct_percent']:.2f}%和
{change.loc[(24, 'XGBoost'), 'rolling_minus_direct_percent']:.2f}%。
2. Dual-CNN滚动在1、3、6小时最低，说明专门训练的一步模型在短提前量更有优势；
到12小时与Dual-CNN-Past直接预测基本持平，到24小时滚动比Direct-Past高
{change.loc[(24, 'Dual-CNN (best direct past)'), 'rolling_minus_direct_percent']:.2f}%。
3. Combined-Ridge直接与滚动差异较小，24小时滚动反而低
{abs(change.loc[(24, 'Combined-Ridge'), 'rolling_minus_direct_percent']):.2f}%，
说明线性模型在本实验中没有出现明显递归失控。
4. XGBoost滚动Bias由1小时接近0逐渐变为24小时负偏，表明自身预测回填会累积低估；
直接XGBoost在中长提前量和强增水样本中更稳定。
5. 信息条件严格匹配时，应比较Dual-CNN-Future直接与Dual-CNN滚动；
Dual-CNN-Past虽然是更好的直接空间模型，但它没有使用未来ERA5，需作为补充比较而非唯一公平参照。

## 解释限制

- 直接模型为多输出或逐提前量模型，滚动模型为专门的一步模型；两者训练目标和可用训练样本数不同。因此差异代表两套完整预测方案的效果，不能全部归因于递归回填本身。
- 当前结果仍不能证明XGBoost的提升来自ERA5。下一项实验应是最小ERA5消融，而不是继续增加模型类型。
- 本轮只用于2017模型开发，2018仍未加载。

## 输出

- `metrics_all_leads.csv`：所有方法1—24小时完整指标。
- `rmse_primary_leads.csv`：重点提前量统一RMSE表。
- `rolling_vs_direct_change.csv`：滚动相对直接的RMSE变化百分比。
- `predictions.csv`：全部共同起报时刻的观测、直接预测和滚动预测。
- `rmse_direct_vs_rolling.png`：三类模型直接与滚动误差曲线。
- `strong_event_*.png`：三次典型强增水过程。
"""
    (output / "2017_direct_vs_rolling_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path or (
        MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    )
    direct_path = args.direct_predictions or (
        MODULE_ROOT / "outputs" / "experiments" / args.station /
        "direct_multimodel_2017_seed42" / "validation_predictions.csv"
    )
    ridge_path = args.ridge_model or (
        MODULE_ROOT / "models" / args.station / "baselines" / "ridge_pipeline.joblib"
    )
    dual_path = args.dual_checkpoint or (
        MODULE_ROOT / "models" / args.station / "formal_seed42" / "dual" / "best_model.pth"
    )
    output = args.output_dir or (
        MODULE_ROOT / "outputs" / "experiments" / args.station /
        "rolling_24_comparison_2017_seed42"
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
        raise AssertionError("2018 entered rolling comparison")
    atmosphere_valid = hourly_atmosphere_valid(atmosphere)
    origins = direct_origins(times, atmosphere_valid, surge, {2017})
    observed = direct_labels(surge, origins)
    initial_histories = np.stack(
        [np.asarray(surge[o - 23 : o + 1], dtype=np.float32) for o in origins]
    )
    predictions = load_direct_predictions(direct_path, pd.DatetimeIndex(times[origins]))
    print(
        f"loaded through {times.max()}; common origins={len(origins)} device={device}",
        flush=True,
    )

    summaries = summarise_atmosphere(atmosphere)
    ridge = joblib.load(ridge_path)
    predictions["combined_ridge_rolling"] = recursive_tabular_predictions(
        ridge, summaries, initial_histories, origins
    )

    xgboost_path = output / "xgboost_one_step.joblib"
    if args.reuse_xgboost and xgboost_path.is_file():
        xgboost = joblib.load(xgboost_path)
    else:
        from xgboost import XGBRegressor

        targets, _ = valid_targets(times, atmosphere, surge, 24)
        train_targets = [target for target in targets if 2011 <= times[target].year <= 2016]
        features = build_ridge_features(summaries, surge, train_targets, 24)
        labels = np.asarray(surge[train_targets], dtype=np.float32)
        print(f"fitting one-step XGBoost on {len(train_targets)} samples", flush=True)
        xgboost = XGBRegressor(
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
            random_state=args.seed,
        )
        xgboost.fit(features, labels)
        joblib.dump(xgboost, xgboost_path, compress=3)
    predictions["xgboost_rolling"] = recursive_tabular_predictions(
        xgboost, summaries, initial_histories, origins
    )

    checkpoint = torch.load(dual_path, map_location=device, weights_only=False)
    predictions["dual_cnn_rolling"] = recursive_dual_predictions(
        atmosphere, origins, initial_histories, checkpoint, device, args.batch_size
    )

    metrics = calculate_comparison_metrics(observed, predictions, surge, origins)
    metrics.to_csv(output / "metrics_all_leads.csv", index=False)
    primary = metrics[metrics.lead_hours.isin(PRIMARY_LEADS)].pivot(
        index="lead_hours", columns="method", values="rmse_cm"
    ).reset_index()
    primary.insert(1, "valid_samples", len(origins))
    primary = primary[["lead_hours", "valid_samples", *METHODS]]
    primary.to_csv(output / "rmse_primary_leads.csv", index=False)
    changes = method_change_table(metrics)
    changes.to_csv(output / "rolling_vs_direct_change.csv", index=False)

    prediction_data: dict[str, Any] = {"forecast_origin": times[origins]}
    for lead in range(1, 25):
        prediction_data[f"valid_time_{lead:02d}h"] = times[origins + lead]
        prediction_data[f"observed_{lead:02d}h_m"] = observed[:, lead - 1]
        for method in METHODS:
            prediction_data[f"{method}_{lead:02d}h_m"] = predictions[method][:, lead - 1]
    pd.DataFrame(prediction_data).to_csv(output / "predictions.csv", index=False)
    plot_rmse(metrics, output / "rmse_direct_vs_rolling.png")
    events = plot_events(output, times, origins, observed, predictions)
    metadata = {
        "experiment": "2017 direct versus rolling 24-hour comparison",
        "loaded_time_max": times.max().isoformat(),
        "2018_loaded": False,
        "common_origin_count": int(len(origins)),
        "train_years": [2011, 2016],
        "validation_year": 2017,
        "future_era5": "known reanalysis truth at each future valid time",
        "future_observed_surge_used_in_rolling": False,
        "xgboost_one_step_model": str(xgboost_path),
        "ridge_one_step_model": str(ridge_path),
        "dual_one_step_checkpoint": str(dual_path),
        "events": events,
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output, primary, changes, metadata)
    print(primary.to_string(index=False), flush=True)
    print(changes.to_string(index=False), flush=True)
    print(f"outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
