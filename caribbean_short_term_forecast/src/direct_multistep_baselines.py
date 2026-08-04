"""Leakage-safe direct 24-hour ridge baselines evaluated on 2017 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .evaluate import calculate_metrics
    from .evaluate_baselines import summarise_atmosphere
    from .train_station import load_prepared
except ImportError:
    from evaluate import calculate_metrics
    from evaluate_baselines import summarise_atmosphere
    from train_station import load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_LEADS = (1, 3, 6, 12, 24)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0, 1_000_000.0)
MODEL_NAMES = (
    "persistence",
    "ar_ridge",
    "era5_ridge_past",
    "era5_ridge_past_future",
    "combined_ridge_past",
    "combined_ridge_past_future",
)
DISPLAY_NAMES = {
    "persistence": "Persistence",
    "ar_ridge": "AR-Ridge",
    "era5_ridge_past": "ERA5-Ridge (past)",
    "era5_ridge_past_future": "ERA5-Ridge (past+future)",
    "combined_ridge_past": "Combined-Ridge (past)",
    "combined_ridge_past_future": "Combined-Ridge (past+future)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--input-steps", type=int, default=24)
    parser.add_argument("--output-steps", type=int, default=24)
    parser.add_argument("--validation-year", type=int, default=2017)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rolling-predictions", type=Path)
    return parser.parse_args()


def hourly_atmosphere_valid(atmosphere: np.ndarray, chunk_hours: int = 168) -> np.ndarray:
    valid = np.zeros(len(atmosphere), dtype=bool)
    for start in range(0, len(atmosphere), chunk_hours):
        chunk = np.asarray(atmosphere[start : start + chunk_hours])
        valid[start : start + len(chunk)] = np.isfinite(chunk).all(axis=(1, 2, 3))
    return valid


def direct_origins(
    times: pd.DatetimeIndex,
    atmosphere_valid: np.ndarray,
    surge: np.ndarray,
    years: set[int],
    input_steps: int = 24,
    output_steps: int = 24,
) -> np.ndarray:
    """Find origins with complete history, future forcing and 24 labels.

    The origin is the latest observed input time ``t``. Inputs cover
    ``t-23...t`` and labels/future ERA5 cover ``t+1...t+24``. All labels must
    remain in the origin calendar year, preventing a target sequence from
    crossing a split-year boundary.
    """
    values = np.asarray(surge, dtype=float)
    one_hour = np.timedelta64(1, "h")
    origins: list[int] = []
    for origin in np.flatnonzero(np.isin(times.year, list(years))):
        history_start = int(origin) - input_steps + 1
        future_stop = int(origin) + output_steps + 1
        if history_start < 0 or future_stop > len(times):
            continue
        if times[future_stop - 1].year != times[origin].year:
            continue
        window_times = times[history_start:future_stop]
        if not np.all(np.diff(window_times.values) == one_hour):
            continue
        if not atmosphere_valid[history_start:future_stop].all():
            continue
        if not np.isfinite(values[history_start : origin + 1]).all():
            continue
        if not np.isfinite(values[origin + 1 : future_stop]).all():
            continue
        origins.append(int(origin))
    return np.asarray(origins, dtype=np.int64)


def direct_labels(surge: np.ndarray, origins: np.ndarray, output_steps: int = 24) -> np.ndarray:
    return np.stack(
        [np.asarray(surge[o + 1 : o + output_steps + 1], dtype=np.float32) for o in origins]
    )


def build_feature_matrix(
    summaries: np.ndarray,
    surge: np.ndarray,
    origins: np.ndarray,
    include_surge: bool,
    era5_mode: str,
    input_steps: int = 24,
    output_steps: int = 24,
) -> np.ndarray:
    """Build compact ERA5 and/or surge features for direct prediction."""
    if era5_mode not in {"none", "past", "past_future"}:
        raise ValueError("era5_mode must be none, past or past_future")
    weather_per_hour = summaries.shape[1] * summaries.shape[2]
    weather_hours = 0 if era5_mode == "none" else input_steps
    if era5_mode == "past_future":
        weather_hours += output_steps
    width = weather_hours * weather_per_hour + (input_steps if include_surge else 0)
    features = np.empty((len(origins), width), dtype=np.float32)
    for row, origin in enumerate(origins):
        cursor = 0
        if era5_mode != "none":
            past = summaries[origin - input_steps + 1 : origin + 1].reshape(-1)
            features[row, cursor : cursor + len(past)] = past
            cursor += len(past)
        if era5_mode == "past_future":
            future = summaries[origin + 1 : origin + output_steps + 1].reshape(-1)
            features[row, cursor : cursor + len(future)] = future
            cursor += len(future)
        if include_surge:
            history = surge[origin - input_steps + 1 : origin + 1]
            features[row, cursor : cursor + input_steps] = history
    return features


def select_alphas_and_fit(
    x_early: np.ndarray,
    y_early: np.ndarray,
    x_tuning: np.ndarray,
    y_tuning: np.ndarray,
    x_all_train: np.ndarray,
    y_all_train: np.ndarray,
    alphas: tuple[float, ...] = ALPHAS,
) -> tuple[StandardScaler, dict[float, Ridge], np.ndarray, np.ndarray]:
    """Choose alpha per lead on 2016, then refit on all 2011-2016 data."""
    selection_scaler = StandardScaler().fit(x_early)
    early_scaled = selection_scaler.transform(x_early)
    tuning_scaled = selection_scaler.transform(x_tuning)
    tuning_rmse = np.empty((len(alphas), y_early.shape[1]), dtype=np.float64)
    for alpha_index, alpha in enumerate(alphas):
        candidate = Ridge(alpha=alpha)
        candidate.fit(early_scaled, y_early)
        predicted = candidate.predict(tuning_scaled)
        tuning_rmse[alpha_index] = np.sqrt(np.mean((predicted - y_tuning) ** 2, axis=0))
    chosen_indices = np.argmin(tuning_rmse, axis=0)
    chosen_alphas = np.asarray(alphas, dtype=float)[chosen_indices]

    final_scaler = StandardScaler().fit(x_all_train)
    all_scaled = final_scaler.transform(x_all_train)
    models: dict[float, Ridge] = {}
    for alpha in np.unique(chosen_alphas):
        leads = np.flatnonzero(chosen_alphas == alpha)
        model = Ridge(alpha=float(alpha))
        model.fit(all_scaled, y_all_train[:, leads])
        models[float(alpha)] = model
    selected_tuning_rmse = tuning_rmse[chosen_indices, np.arange(y_early.shape[1])]
    return final_scaler, models, chosen_alphas, selected_tuning_rmse


def predict_selected_models(
    scaler: StandardScaler,
    models: dict[float, Ridge],
    chosen_alphas: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    scaled = scaler.transform(features)
    predictions = np.empty((len(features), len(chosen_alphas)), dtype=np.float32)
    for alpha, model in models.items():
        leads = np.flatnonzero(chosen_alphas == alpha)
        values = np.asarray(model.predict(scaled), dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        predictions[:, leads] = values
    return predictions


def metric_table(
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    surge: np.ndarray,
    origins: np.ndarray,
) -> pd.DataFrame:
    reference_name = "combined_ridge_past_future"
    rows: list[dict[str, Any]] = []
    for lead_index in range(observed.shape[1]):
        lead = lead_index + 1
        target = observed[:, lead_index]
        prior = np.asarray(surge[origins + lead_index], dtype=float)
        absolute_threshold = float(np.quantile(np.abs(target), 0.95))
        top_mask = np.abs(target) >= absolute_threshold
        rises = target - prior
        positive = rises[rises > 0]
        rise_threshold = float(np.quantile(positive, 0.90)) if len(positive) else np.nan
        rise_mask = rises >= rise_threshold if np.isfinite(rise_threshold) else np.zeros(len(target), dtype=bool)
        reference = predictions[reference_name][:, lead_index]
        reference_mse = float(np.mean((reference - target) ** 2))
        for name in MODEL_NAMES:
            predicted = predictions[name][:, lead_index]
            overall = calculate_metrics(target, predicted)
            top = calculate_metrics(target[top_mask], predicted[top_mask])
            rapid = calculate_metrics(target[rise_mask], predicted[rise_mask])
            model_mse = float(np.mean((predicted - target) ** 2))
            rows.append(
                {
                    "lead_hours": lead,
                    "model": name,
                    **overall,
                    "skill_vs_combined_past_future": (
                        float(1 - model_mse / reference_mse) if reference_mse > 0 else np.nan
                    ),
                    "top5_n": int(top["n"]),
                    "top5_threshold_cm": absolute_threshold * 100,
                    "top5_rmse_cm": top["rmse_cm"],
                    "rapid_rise_n": int(rapid["n"]),
                    "rapid_rise_threshold_cm_per_hour": rise_threshold * 100,
                    "rapid_rise_rmse_cm": rapid["rmse_cm"],
                }
            )
    return pd.DataFrame(rows)


def plot_rmse(metrics: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    history_models = (
        "persistence", "ar_ridge", "combined_ridge_past",
        "combined_ridge_past_future",
    )
    for name in history_models:
        subset = metrics[metrics.model == name]
        axes[0].plot(subset.lead_hours, subset.rmse_cm, linewidth=1.8, label=DISPLAY_NAMES[name])
    for name in ("era5_ridge_past", "era5_ridge_past_future"):
        subset = metrics[metrics.model == name]
        axes[1].plot(subset.lead_hours, subset.rmse_cm, linewidth=1.8, label=DISPLAY_NAMES[name])
    for ax in axes:
        ax.set_ylabel("RMSE (cm)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_title("History-based and combined baselines")
    axes[1].set_title("ERA5-only ablation")
    axes[1].set_xlabel("Direct forecast lead (hours)")
    axes[1].set_xticks([1, 3, 6, 12, 18, 24])
    fig.suptitle("2017 direct multi-step ridge baselines (24-hour output)")
    fig.tight_layout()
    fig.savefig(destination, dpi=400)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---:" for _ in frame.columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare_with_recursive(
    rolling_path: Path,
    times: pd.DatetimeIndex,
    origins: np.ndarray,
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame | None:
    """Compare direct and recursive ridge on exactly matched 2017 origins."""
    if not rolling_path.is_file():
        return None
    rolling = pd.read_csv(rolling_path, parse_dates=["forecast_origin"])
    rows: list[dict[str, Any]] = []
    for lead in PRIMARY_LEADS:
        direct = pd.DataFrame(
            {
                "rolling_origin": times[origins] + pd.Timedelta(hours=1),
                "observed": observed[:, lead - 1],
                "persistence": predictions["persistence"][:, lead - 1],
                "ar_ridge": predictions["ar_ridge"][:, lead - 1],
                "combined_ridge_past": predictions["combined_ridge_past"][:, lead - 1],
                "combined_ridge_past_future": predictions[
                    "combined_ridge_past_future"
                ][:, lead - 1],
            }
        )
        recursive = rolling[rolling.lead_hours == lead][
            ["forecast_origin", "observed_m", "ridge_m"]
        ]
        joined = direct.merge(
            recursive, left_on="rolling_origin", right_on="forecast_origin"
        )
        if not len(joined):
            continue
        if not np.allclose(joined.observed, joined.observed_m, equal_nan=False):
            raise AssertionError("Direct and recursive verification targets do not align")
        row: dict[str, Any] = {"lead_hours": lead, "valid_samples": len(joined)}
        for name in (
            "persistence", "recursive_ridge", "ar_ridge",
            "combined_ridge_past", "combined_ridge_past_future",
        ):
            values = joined.ridge_m if name == "recursive_ridge" else joined[name]
            row[name] = float(
                np.sqrt(np.mean(((np.asarray(values) - joined.observed) * 100) ** 2))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    output: Path,
    selected_rmse: pd.DataFrame,
    metadata: dict[str, Any],
    recursive_comparison: pd.DataFrame | None = None,
) -> None:
    comparison_section = ""
    if recursive_comparison is not None and len(recursive_comparison):
        comparison = recursive_comparison.set_index("lead_hours")
        direct_best = {
            lead: min(
                comparison.loc[lead, "ar_ridge"],
                comparison.loc[lead, "combined_ridge_past"],
                comparison.loc[lead, "combined_ridge_past_future"],
            )
            for lead in PRIMARY_LEADS
        }
        reductions = {
            lead: 100 * (1 - direct_best[lead] / comparison.loc[lead, "recursive_ridge"])
            for lead in PRIMARY_LEADS
        }
        comparison_section = f"""
## 与原递归岭回归的同起报时刻比较

下表只保留两套实验都有效的共同起报时刻，避免因样本不同造成误判。单位为cm。

{markdown_table(recursive_comparison)}

该表用于判断直接输出本身是否优于递归；不能用上面的全样本表直接与旧滚动表比较。
各提前量最优直接岭回归相对递归岭回归的RMSE变化分别为：1小时{reductions[1]:.2f}%、
3小时{reductions[3]:.2f}%、6小时{reductions[6]:.2f}%、12小时{reductions[12]:.2f}%、
24小时{reductions[24]:.2f}%。当前线性直接输出只带来约0—1.2%的小幅变化，
尚不能说明“避免递归”本身会在线性模型上产生大幅改善。
"""
    primary = selected_rmse.set_index("lead_hours")
    future_effect = {
        lead: 100 * (
            1 - primary.loc[lead, "combined_ridge_past_future"]
            / primary.loc[lead, "combined_ridge_past"]
        )
        for lead in PRIMARY_LEADS
    }
    content = f"""# Prickly Bay 2017直接多步线性基线报告

## 实验边界

- 仅加载到2017年末，未加载2018直接多步数据。
- 输入历史为24小时，输出为连续未来24小时。
- 所有24个目标必须存在，且目标序列不得跨越年份边界。
- 岭回归正则系数先用2011—2015拟合、2016选择，再用2011—2016重拟合。
- ERA5统计量为每个变量、每小时的空间均值、标准差、最小值和最大值。
- `past+future` 使用未来24小时ERA5再分析真值，属于已知未来大气强迫的历史回算，不是业务预报。
- 训练样本：{metadata['train_samples']}；2016内部调参样本：{metadata['tuning_samples_2016']}；2017验证样本：{metadata['validation_samples']}。

## 主要提前量RMSE（cm）

{markdown_table(selected_rmse)}

{comparison_section}

## 当前结论

1. Combined-Ridge (past)在1和3小时最低；Combined-Ridge (past+future)仅在6和12小时略低；24小时则是AR-Ridge最低。
2. 在Combined-Ridge中加入未来ERA5后，1、3、6、12、24小时RMSE变化分别为
{future_effect[1]:.2f}%、{future_effect[3]:.2f}%、{future_effect[6]:.2f}%、
{future_effect[12]:.2f}%、{future_effect[24]:.2f}%（正值表示改善）。未来ERA5统计特征只在6小时出现约1.3%的改善，证据并不稳定。
3. ERA5-only线性基线明显较差，说明当前空间均值/标准差/极值不足以单独描述站点海洋状态，也可能存在明显年度分布变化。
4. 下一步仍可训练24维输出的surge_mlp和双分支模型，但必须同时比较AR-Ridge、Combined-Ridge (past)和Combined-Ridge (past+future)，不能只选一个较弱参照。
5. 神经网络第一轮保持普通MSE和现有编码器；暂时不增加复杂结构或强事件加权。

## 模型含义

- Persistence：未来24小时均等于起报时刻增水。
- AR-Ridge：只使用历史24小时增水。
- ERA5-Ridge (past)：只使用历史24小时ERA5统计特征。
- ERA5-Ridge (past+future)：使用历史和未来各24小时ERA5统计特征。
- Combined-Ridge (past)：历史增水加历史ERA5统计特征。
- Combined-Ridge (past+future)：历史增水加历史、未来ERA5统计特征，是本阶段主要线性基线。

## 输出

- `direct_metrics_all_leads.csv`：1—24小时全部指标。
- `direct_rmse_primary_leads.csv`：1、3、6、12、24小时RMSE。
- `selected_alphas.csv`：各模型、各提前量由2016选出的正则系数。
- `validation_predictions.csv`：2017逐起报时刻连续24小时预测。
- `direct_rmse_vs_lead.png`：直接预测RMSE随提前量变化。
- `direct_ridge_models.joblib`：重拟合后的标准化器、模型和配置。
"""
    (output / "2017_direct_multistep_baseline_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.validation_year != 2017:
        raise ValueError("This leakage guard permits the 2017 validation year only")
    if args.input_steps != 24 or args.output_steps != 24:
        raise ValueError("This baseline is fixed to 24-hour history and 24-hour output")
    dataset_path = args.dataset_path or (
        MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    )
    output = args.output_dir or (
        MODULE_ROOT / "outputs" / "experiments" / args.station / "direct_multistep_baselines_2017"
    )
    output.mkdir(parents=True, exist_ok=True)
    rolling_path = args.rolling_predictions or (
        MODULE_ROOT / "outputs" / "experiments" / args.station /
        "rolling_2017_seed42" / "rolling_predictions_selected_leads.csv"
    )

    atmosphere, surge, times = load_prepared(dataset_path, 2011, 2017)
    if (times.year > 2017).any() or times.max().year != 2017:
        raise AssertionError("Data later than 2017 entered direct multi-step development")
    atmosphere_valid = hourly_atmosphere_valid(atmosphere)
    all_train_origins = direct_origins(
        times, atmosphere_valid, surge, set(range(2011, 2017)), args.input_steps, args.output_steps
    )
    early_origins = all_train_origins[times[all_train_origins].year <= 2015]
    tuning_origins = all_train_origins[times[all_train_origins].year == 2016]
    validation_origins = direct_origins(
        times, atmosphere_valid, surge, {2017}, args.input_steps, args.output_steps
    )
    if not len(early_origins) or not len(tuning_origins) or not len(validation_origins):
        raise ValueError("At least one direct multi-step split is empty")
    print(
        f"loaded through {times.max()}; train={len(all_train_origins)} "
        f"tune2016={len(tuning_origins)} validation2017={len(validation_origins)}",
        flush=True,
    )

    summaries = summarise_atmosphere(atmosphere)
    y_all = direct_labels(surge, all_train_origins, args.output_steps)
    y_early = direct_labels(surge, early_origins, args.output_steps)
    y_tuning = direct_labels(surge, tuning_origins, args.output_steps)
    y_validation = direct_labels(surge, validation_origins, args.output_steps)
    configurations = {
        "ar_ridge": (True, "none"),
        "era5_ridge_past": (False, "past"),
        "era5_ridge_past_future": (False, "past_future"),
        "combined_ridge_past": (True, "past"),
        "combined_ridge_past_future": (True, "past_future"),
    }
    predictions: dict[str, np.ndarray] = {
        "persistence": np.repeat(
            np.asarray(surge[validation_origins], dtype=np.float32)[:, None],
            args.output_steps,
            axis=1,
        )
    }
    saved_models: dict[str, Any] = {}
    alpha_rows: list[dict[str, Any]] = []
    for name, (include_surge, era5_mode) in configurations.items():
        print(f"fitting {name}", flush=True)
        x_all = build_feature_matrix(
            summaries, surge, all_train_origins, include_surge, era5_mode,
            args.input_steps, args.output_steps,
        )
        early_mask = times[all_train_origins].year <= 2015
        tuning_mask = times[all_train_origins].year == 2016
        scaler, models, chosen, tuning_rmse = select_alphas_and_fit(
            x_all[early_mask], y_early, x_all[tuning_mask], y_tuning, x_all, y_all
        )
        x_validation = build_feature_matrix(
            summaries, surge, validation_origins, include_surge, era5_mode,
            args.input_steps, args.output_steps,
        )
        predictions[name] = predict_selected_models(scaler, models, chosen, x_validation)
        saved_models[name] = {
            "include_surge": include_surge,
            "era5_mode": era5_mode,
            "scaler": scaler,
            "models_by_alpha": models,
            "selected_alpha_by_lead": chosen,
        }
        alpha_rows.extend(
            {
                "model": name,
                "lead_hours": lead + 1,
                "selected_alpha": chosen[lead],
                "2016_tuning_rmse_cm": tuning_rmse[lead] * 100,
                "feature_count": x_all.shape[1],
            }
            for lead in range(args.output_steps)
        )

    metrics = metric_table(y_validation, predictions, surge, validation_origins)
    metrics.to_csv(output / "direct_metrics_all_leads.csv", index=False)
    selected = metrics[metrics.lead_hours.isin(PRIMARY_LEADS)].pivot(
        index="lead_hours", columns="model", values="rmse_cm"
    ).reset_index()
    selected.insert(1, "valid_samples", len(validation_origins))
    selected = selected[["lead_hours", "valid_samples", *MODEL_NAMES]]
    selected.to_csv(output / "direct_rmse_primary_leads.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(output / "selected_alphas.csv", index=False)

    prediction_data: dict[str, Any] = {
        "forecast_origin": times[validation_origins],
    }
    for lead in range(1, args.output_steps + 1):
        prediction_data[f"valid_time_{lead:02d}h"] = times[validation_origins + lead]
        prediction_data[f"observed_{lead:02d}h_m"] = y_validation[:, lead - 1]
        for name in MODEL_NAMES:
            prediction_data[f"{name}_{lead:02d}h_m"] = predictions[name][:, lead - 1]
    pd.DataFrame(prediction_data).to_csv(output / "validation_predictions.csv", index=False)
    plot_rmse(metrics, output / "direct_rmse_vs_lead.png")
    recursive_comparison = compare_with_recursive(
        rolling_path, times, validation_origins, y_validation, predictions
    )
    if recursive_comparison is not None:
        recursive_comparison.to_csv(
            output / "direct_vs_recursive_common_origins.csv", index=False
        )

    metadata = {
        "experiment": "2017 direct 24-hour baseline development",
        "loaded_time_min": times.min().isoformat(),
        "loaded_time_max": times.max().isoformat(),
        "2018_loaded": False,
        "input_steps": args.input_steps,
        "output_steps": args.output_steps,
        "train_years": [2011, 2016],
        "alpha_selection_year": 2016,
        "validation_year": 2017,
        "train_samples": int(len(all_train_origins)),
        "early_fit_samples_2011_2015": int(len(early_origins)),
        "tuning_samples_2016": int(len(tuning_origins)),
        "validation_samples": int(len(validation_origins)),
        "alpha_candidates": list(ALPHAS),
        "future_era5_definition": "known future ERA5 reanalysis; historical hindcast only",
        "recursive_comparison_path": str(rolling_path) if rolling_path.is_file() else None,
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {"metadata": metadata, "models": saved_models},
        output / "direct_ridge_models.joblib",
        compress=3,
    )
    write_report(output, selected, metadata, recursive_comparison)
    print(selected.to_string(index=False), flush=True)
    print(f"outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
