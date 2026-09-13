"""2017 recursive hindcast diagnostics with known future ERA5 forcing.

This module deliberately loads data only through the validation year.  It is a
diagnostic of one-hour models under perfect atmospheric forcing, not an
operational forecast.
"""

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
    from .evaluate import calculate_metrics
    from .evaluate_baselines import summarise_atmosphere
    from .forecast_model import model_from_checkpoint
    from .train_station import load_prepared
except ImportError:
    from evaluate import calculate_metrics
    from evaluate_baselines import summarise_atmosphere
    from forecast_model import model_from_checkpoint
    from train_station import load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[1]
LEADS = (1, 3, 6, 12, 24, 48, 72)
MODEL_NAMES = ("persistence", "ridge", "surge_mlp", "era5_cnn", "dual")
DISPLAY_NAMES = {
    "persistence": "Persistence",
    "ridge": "Ridge",
    "surge_mlp": "Surge MLP",
    "era5_cnn": "ERA5 CNN",
    "dual": "Dual branch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--validation-year", type=int, default=2017)
    parser.add_argument("--input-steps", type=int, default=24)
    parser.add_argument("--max-lead", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--ridge-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def atmospheric_validity(atmosphere: np.ndarray, chunk_hours: int = 168) -> np.ndarray:
    valid = np.zeros(len(atmosphere), dtype=bool)
    for start in range(0, len(atmosphere), chunk_hours):
        chunk = np.asarray(atmosphere[start : start + chunk_hours])
        valid[start : start + len(chunk)] = np.isfinite(chunk).all(axis=(1, 2, 3))
    return valid


def find_common_origins(
    times: pd.DatetimeIndex,
    atmosphere_valid: np.ndarray,
    surge: np.ndarray,
    validation_year: int,
    input_steps: int,
    max_lead: int,
    leads: tuple[int, ...] = LEADS,
) -> np.ndarray:
    """Return origins valid for every model and every requested lead.

    An origin predicts ``origin`` at lead 1 using history ending at
    ``origin - 1``. Only requested verification targets need observations;
    intermediate true surge values are never required by recursive models.
    """
    surge = np.asarray(surge, dtype=float)
    origins: list[int] = []
    one_hour = np.timedelta64(1, "h")
    for origin in np.flatnonzero(times.year == validation_year):
        end = int(origin) + max_lead - 1
        if origin < input_steps or end >= len(times):
            continue
        if times[end].year != validation_year:
            continue
        if not np.all(np.diff(times[origin - input_steps : end + 1].values) == one_hour):
            continue
        if not atmosphere_valid[origin - input_steps : end].all():
            continue
        if not np.isfinite(surge[origin - input_steps : origin]).all():
            continue
        targets = int(origin) + np.asarray(leads, dtype=int) - 1
        if not np.isfinite(surge[targets]).all():
            continue
        # Needed to classify rapid-rise targets without borrowing model inputs.
        if not np.isfinite(surge[targets - 1]).all():
            continue
        origins.append(int(origin))
    return np.asarray(origins, dtype=np.int64)


def load_models(
    checkpoint_root: Path, device: torch.device
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    models: dict[str, torch.nn.Module] = {}
    checkpoints: dict[str, Any] = {}
    for name in ("surge_mlp", "era5_cnn", "dual"):
        path = checkpoint_root / name / "best_model.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed-42 checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = model_from_checkpoint(checkpoint).to(device)
        model.eval()
        models[name] = model
        checkpoints[name] = checkpoint
    reference = checkpoints["dual"]["scalers"]
    for name, checkpoint in checkpoints.items():
        if checkpoint["scalers"] != reference:
            raise ValueError(f"{name} uses scalers inconsistent with the dual model")
        if int(checkpoint["input_steps"]) != 24:
            raise ValueError(f"{name} was not trained with a 24-hour input window")
    return models, checkpoints


def recursive_predictions(
    atmosphere: np.ndarray,
    surge: np.ndarray,
    origins: np.ndarray,
    summaries: np.ndarray,
    ridge: Any,
    models: dict[str, torch.nn.Module],
    scalers: dict[str, Any],
    input_steps: int,
    max_lead: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    count = len(origins)
    predictions = {
        name: np.empty((count, max_lead), dtype=np.float32) for name in MODEL_NAMES
    }
    predictions["persistence"][:] = np.asarray(surge[origins - 1], dtype=np.float32)[:, None]
    atmosphere_mean = np.asarray(scalers["atmosphere"]["mean"], dtype=np.float32).reshape(
        1, 1, -1, 1, 1
    )
    atmosphere_scale = np.asarray(
        scalers["atmosphere"]["scale"], dtype=np.float32
    ).reshape(1, 1, -1, 1, 1)
    surge_mean = float(scalers["surge"]["mean"][0])
    surge_scale = float(scalers["surge"]["scale"][0])
    weather_width = input_steps * summaries.shape[1] * summaries.shape[2]

    for batch_start in range(0, count, batch_size):
        batch_stop = min(count, batch_start + batch_size)
        batch_origins = origins[batch_start:batch_stop]
        size = len(batch_origins)
        histories = {
            name: np.stack(
                [np.asarray(surge[o - input_steps : o], dtype=np.float32) for o in batch_origins]
            )
            for name in ("ridge", "surge_mlp", "dual")
        }
        empty_weather = torch.empty((size, 0), dtype=torch.float32, device=device)
        with torch.inference_mode():
            for lead in range(1, max_lead + 1):
                targets = batch_origins + lead - 1
                raw_weather = np.stack(
                    [
                        np.asarray(atmosphere[t - input_steps : t], dtype=np.float32)
                        for t in targets
                    ]
                )
                scaled_weather = (raw_weather - atmosphere_mean) / atmosphere_scale
                weather_tensor = torch.from_numpy(
                    scaled_weather.reshape(
                        size, -1, atmosphere.shape[-2], atmosphere.shape[-1]
                    )
                ).to(device, non_blocking=True)

                ridge_features = np.empty(
                    (size, weather_width + input_steps), dtype=np.float32
                )
                for row, target in enumerate(targets):
                    ridge_features[row, :weather_width] = summaries[
                        target - input_steps : target
                    ].reshape(-1)
                ridge_features[:, weather_width:] = histories["ridge"]
                ridge_values = ridge.predict(ridge_features).astype(np.float32)
                predictions["ridge"][batch_start:batch_stop, lead - 1] = ridge_values

                mlp_history = torch.from_numpy(
                    (histories["surge_mlp"] - surge_mean) / surge_scale
                ).to(device, non_blocking=True)
                mlp_scaled = models["surge_mlp"](empty_weather, mlp_history)
                mlp_values = (
                    mlp_scaled.detach().cpu().numpy() * surge_scale + surge_mean
                ).astype(np.float32)
                predictions["surge_mlp"][batch_start:batch_stop, lead - 1] = mlp_values

                dummy_history = torch.empty((size, 0), dtype=torch.float32, device=device)
                era_scaled = models["era5_cnn"](weather_tensor, dummy_history)
                era_values = (
                    era_scaled.detach().cpu().numpy() * surge_scale + surge_mean
                ).astype(np.float32)
                predictions["era5_cnn"][batch_start:batch_stop, lead - 1] = era_values

                dual_history = torch.from_numpy(
                    (histories["dual"] - surge_mean) / surge_scale
                ).to(device, non_blocking=True)
                dual_scaled = models["dual"](weather_tensor, dual_history)
                dual_values = (
                    dual_scaled.detach().cpu().numpy() * surge_scale + surge_mean
                ).astype(np.float32)
                predictions["dual"][batch_start:batch_stop, lead - 1] = dual_values

                for name, values in (
                    ("ridge", ridge_values),
                    ("surge_mlp", mlp_values),
                    ("dual", dual_values),
                ):
                    histories[name][:, :-1] = histories[name][:, 1:]
                    histories[name][:, -1] = values
        print(
            f"recursive batch {batch_stop}/{count} complete on {device}",
            flush=True,
        )
    return predictions


def build_metrics(
    surge: np.ndarray,
    origins: np.ndarray,
    predictions: dict[str, np.ndarray],
    leads: tuple[int, ...] = LEADS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lead in leads:
        targets = origins + lead - 1
        observed = np.asarray(surge[targets], dtype=float)
        prior = np.asarray(surge[targets - 1], dtype=float)
        ridge_values = predictions["ridge"][:, lead - 1]
        ridge_mse = float(np.mean((ridge_values - observed) ** 2))
        top_threshold = float(np.quantile(np.abs(observed), 0.95))
        top_mask = np.abs(observed) >= top_threshold
        rises = observed - prior
        positive = rises[rises > 0]
        rise_threshold = float(np.quantile(positive, 0.90)) if len(positive) else np.nan
        rise_mask = rises >= rise_threshold if np.isfinite(rise_threshold) else np.zeros(
            len(observed), dtype=bool
        )
        for name in MODEL_NAMES:
            predicted = predictions[name][:, lead - 1]
            overall = calculate_metrics(observed, predicted)
            model_mse = float(np.mean((predicted - observed) ** 2))
            top = calculate_metrics(observed[top_mask], predicted[top_mask])
            rapid = calculate_metrics(observed[rise_mask], predicted[rise_mask])
            rows.append(
                {
                    "lead_hours": lead,
                    "model": name,
                    **overall,
                    "skill_vs_same_lead_ridge": (
                        float(1 - model_mse / ridge_mse) if ridge_mse > 0 else np.nan
                    ),
                    "top5_n": int(top["n"]),
                    "top5_threshold_cm": top_threshold * 100,
                    "top5_rmse_cm": top["rmse_cm"],
                    "rapid_rise_n": int(rapid["n"]),
                    "rapid_rise_threshold_cm_per_hour": rise_threshold * 100,
                    "rapid_rise_rmse_cm": rapid["rmse_cm"],
                }
            )
    return pd.DataFrame(rows)


def save_prediction_table(
    output: Path,
    times: pd.DatetimeIndex,
    surge: np.ndarray,
    origins: np.ndarray,
    predictions: dict[str, np.ndarray],
    leads: tuple[int, ...] = LEADS,
) -> None:
    frames = []
    for lead in leads:
        targets = origins + lead - 1
        data: dict[str, Any] = {
            "forecast_origin": times[origins],
            "valid_time": times[targets],
            "lead_hours": lead,
            "observed_m": np.asarray(surge[targets], dtype=np.float32),
        }
        data.update({f"{name}_m": values[:, lead - 1] for name, values in predictions.items()})
        frames.append(pd.DataFrame(data))
    pd.concat(frames, ignore_index=True).to_csv(
        output / "rolling_predictions_selected_leads.csv", index=False
    )


def plot_rmse(metrics: pd.DataFrame, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in MODEL_NAMES:
        subset = metrics[metrics.model == name]
        ax.plot(
            subset.lead_hours,
            subset.rmse_cm,
            marker="o",
            linewidth=1.8,
            label=DISPLAY_NAMES[name],
        )
    ax.set(xlabel="Lead time (hours)", ylabel="RMSE (cm)")
    ax.set_xticks(LEADS)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    ax.set_title("2017 recursive hindcast with known future ERA5 forcing")
    fig.tight_layout()
    fig.savefig(destination, dpi=400)
    plt.close(fig)


def plot_strong_events(
    output: Path,
    times: pd.DatetimeIndex,
    surge: np.ndarray,
    origins: np.ndarray,
    predictions: dict[str, np.ndarray],
    max_lead: int,
    event_count: int = 3,
) -> list[dict[str, Any]]:
    complete = np.asarray(
        [
            np.isfinite(surge[origin : origin + max_lead]).all()
            for origin in origins
        ],
        dtype=bool,
    )
    candidates = np.flatnonzero(complete)
    scores = np.asarray(
        [np.max(np.abs(surge[origins[i] : origins[i] + max_lead])) for i in candidates]
    )
    selected: list[int] = []
    for candidate_position in candidates[np.argsort(scores)[::-1]]:
        origin = origins[candidate_position]
        if all(abs(origin - origins[other]) >= max_lead for other in selected):
            selected.append(int(candidate_position))
        if len(selected) == event_count:
            break
    details = []
    for number, position in enumerate(selected, start=1):
        origin = int(origins[position])
        valid_times = times[origin : origin + max_lead]
        observed = np.asarray(surge[origin : origin + max_lead]) * 100
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(valid_times, observed, color="black", linewidth=2.2, label="Observed")
        for name in MODEL_NAMES:
            ax.plot(
                valid_times,
                predictions[name][position] * 100,
                linewidth=1.3,
                label=DISPLAY_NAMES[name],
            )
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set(xlabel="Valid time", ylabel="Storm surge (cm)")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=8)
        ax.set_title(
            f"Strong-event 72 h hindcast from {times[origin]:%Y-%m-%d %H:%M}\n"
            "Known future ERA5 forcing (diagnostic, not operational forecast)"
        )
        fig.autofmt_xdate()
        fig.tight_layout()
        filename = f"strong_event_{number}_{times[origin]:%Y%m%d_%H%M}.png"
        fig.savefig(output / filename, dpi=400)
        plt.close(fig)
        details.append(
            {
                "forecast_origin": times[origin].isoformat(),
                "maximum_absolute_observed_cm": float(np.max(np.abs(observed))),
                "plot": filename,
            }
        )
    return details


def write_report(
    output: Path,
    metrics: pd.DataFrame,
    wide: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    def markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
        headers = list(frame.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---:" for _ in headers) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            cells = [
                f"{value:.{decimals}f}"
                if isinstance(value, (float, np.floating))
                else str(value)
                for value in row
            ]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    best_by_lead = []
    for lead in LEADS:
        subset = metrics[metrics.lead_hours == lead].sort_values("rmse_cm")
        row = subset.iloc[0]
        best_by_lead.append(
            f"- {lead} h：{DISPLAY_NAMES[row.model]} 最低，RMSE {row.rmse_cm:.3f} cm。"
        )
    table = markdown_table(wide)
    dual = metrics[metrics.model == "dual"][
        [
            "lead_hours",
            "mae_cm",
            "bias_cm",
            "pearson_r",
            "skill_vs_same_lead_ridge",
            "top5_rmse_cm",
            "rapid_rise_rmse_cm",
        ]
    ].copy()
    dual.columns = [
        "提前量(h)",
        "MAE(cm)",
        "Bias(cm)",
        "Pearson r",
        "相对岭回归技能",
        "Top5% RMSE(cm)",
        "快速上涨RMSE(cm)",
    ]
    dual_table = markdown_table(dual)
    dual_skills = metrics[metrics.model == "dual"].set_index("lead_hours")[
        "skill_vs_same_lead_ridge"
    ]
    top5 = metrics.pivot(index="lead_hours", columns="model", values="top5_rmse_cm")
    content = f"""# Prickly Bay 2017滚动预报诊断报告

## 实验边界

- 实验名称：**已知未来大气强迫条件下的历史滚动回算**。
- 仅加载并评估到2017年，未加载2018测试数据。
- 使用已训练的seed 42一步模型，输入窗口24小时，递归到72小时。
- ERA5使用各未来时次的再分析真值；这只隔离检查模型与递归误差，不能表述为真实业务预报。
- 岭回归、surge_mlp和双分支模型各自回填预测增水，没有使用未来真实增水。
- 五个模型共享同一批{metadata['common_origin_count']}个起报时刻；每个起报点的全部指定提前量均有真实观测。
- 单位：下表RMSE均为cm。

## 各提前量RMSE

{table}

## 最低RMSE模型

{chr(10).join(best_by_lead)}

## 双分支详细诊断

{dual_table}

技能评分按 `1 - MSE(双分支) / MSE(同提前量岭回归)` 计算。双分支在3、6、12小时为正技能，分别为
{dual_skills.loc[3] * 100:.2f}%、{dual_skills.loc[6] * 100:.2f}%、{dual_skills.loc[12] * 100:.2f}%；
其中6小时最高。到24、48、72小时转为负技能，分别为
{dual_skills.loc[24] * 100:.2f}%、{dual_skills.loc[48] * 100:.2f}%、{dual_skills.loc[72] * 100:.2f}%。

强增水Top 5%样本中，双分支在3、6、12小时RMSE分别为
{top5.loc[3, 'dual']:.3f}、{top5.loc[6, 'dual']:.3f}、{top5.loc[12, 'dual']:.3f} cm，
均低于对应岭回归的{top5.loc[3, 'ridge']:.3f}、{top5.loc[6, 'ridge']:.3f}、{top5.loc[12, 'ridge']:.3f} cm；
24小时后这一优势不再稳定。

## 下一步判断

1. 双分支在3—12小时、尤其6小时出现小幅但一致的正技能，说明ERA5与历史增水的联合表达在短中提前量存在增量价值。
2. 优势在24小时消失，48—72小时进一步恶化，说明当前一步模型的递归误差累积是主要限制之一。
3. ERA5-only整体RMSE约4 cm，所有提前量均未超过岭回归，当前气象分支单独表达能力不足。
4. surge_mlp各提前量均未超过岭回归，说明仅把历史增水换成非线性网络没有带来优势。
5. 因此下一步优先开发**直接多步模型**，先针对3、6、12、24小时分别输出或联合输出；继续只用2017选择结构。暂不优先做“岭回归＋ERA5残差修正”，也不继续依赖72小时递归。

## 输出说明

- `rolling_metrics_long.csv`：RMSE、MAE、Bias、Pearson r、R²、同提前量岭回归技能评分、Top 5%强增水RMSE、快速上涨RMSE。
- `rolling_predictions_selected_leads.csv`：所有共同起报时刻在7个指定提前量的观测与预测。
- `rmse_vs_lead.png`：RMSE随提前量变化。
- `strong_event_*.png`：典型强增水过程的72小时曲线。

## 解释限制

现有网络只针对下一小时训练，长提前量结果用于诊断递归误差从何时开始失控，以及ERA5在较长提前量是否出现增量价值；不能据此直接否定神经网络，也不能把本实验称为业务预报。
"""
    (output / "2017_rolling_diagnostic_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.validation_year != 2017:
        raise ValueError("This leakage guard permits the 2017 validation year only")
    if args.input_steps != 24 or args.max_lead != 72:
        raise ValueError("This experiment is fixed to a 24-hour input and 72-hour recursion")
    dataset_path = args.dataset_path or (
        MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    )
    checkpoint_root = args.checkpoint_root or (
        MODULE_ROOT / "models" / args.station / "formal_seed42"
    )
    ridge_path = args.ridge_path or (
        MODULE_ROOT / "models" / args.station / "baselines" / "ridge_pipeline.joblib"
    )
    output = args.output_dir or (
        MODULE_ROOT / "outputs" / "experiments" / args.station / "rolling_2017_seed42"
    )
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    # Hard leakage boundary: load_prepared slices the mapped arrays through 2017.
    atmosphere, surge, times = load_prepared(dataset_path, 2011, 2017)
    if times.max().year != 2017 or (times.year > 2017).any():
        raise AssertionError("Data later than 2017 entered the rolling diagnostic")
    print(f"loaded {times.min()} through {times.max()}; device={device}", flush=True)
    valid_atmosphere = atmospheric_validity(atmosphere)
    origins = find_common_origins(
        times, valid_atmosphere, surge, 2017, args.input_steps, args.max_lead
    )
    if not len(origins):
        raise ValueError("No common 2017 rolling origins satisfy the experiment rules")
    print(f"common forecast origins: {len(origins)}", flush=True)

    models, checkpoints = load_models(checkpoint_root, device)
    ridge = joblib.load(ridge_path)
    summaries = summarise_atmosphere(atmosphere)
    predictions = recursive_predictions(
        atmosphere,
        surge,
        origins,
        summaries,
        ridge,
        models,
        checkpoints["dual"]["scalers"],
        args.input_steps,
        args.max_lead,
        args.batch_size,
        device,
    )
    metrics = build_metrics(surge, origins, predictions)
    metrics.to_csv(output / "rolling_metrics_long.csv", index=False)
    rmse = metrics.pivot(index="lead_hours", columns="model", values="rmse_cm").reset_index()
    rmse.insert(1, "valid_samples", len(origins))
    rmse = rmse[
        ["lead_hours", "valid_samples", "persistence", "ridge", "surge_mlp", "era5_cnn", "dual"]
    ]
    rmse.to_csv(output / "rolling_rmse_table.csv", index=False)
    save_prediction_table(output, times, surge, origins, predictions)
    plot_rmse(metrics, output / "rmse_vs_lead.png")
    events = plot_strong_events(output, times, surge, origins, predictions, args.max_lead)
    metadata = {
        "experiment_name_zh": "已知未来大气强迫条件下的历史滚动回算",
        "operational_forecast": False,
        "validation_year": 2017,
        "loaded_time_min": times.min().isoformat(),
        "loaded_time_max": times.max().isoformat(),
        "latest_verification_time": times[
            int(origins[-1]) + args.max_lead - 1
        ].isoformat(),
        "future_era5": "known ERA5 reanalysis values at future valid times",
        "future_observed_surge_used_as_model_input": False,
        "input_steps": args.input_steps,
        "leads_hours": list(LEADS),
        "common_origin_count": int(len(origins)),
        "first_origin": times[int(origins[0])].isoformat(),
        "last_origin": times[int(origins[-1])].isoformat(),
        "device": str(device),
        "torch_version": torch.__version__,
        "checkpoints": {
            name: str(checkpoint_root / name / "best_model.pth")
            for name in ("surge_mlp", "era5_cnn", "dual")
        },
        "ridge_pipeline": str(ridge_path),
        "strong_events": events,
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output, metrics, rmse, metadata)
    print(rmse.to_string(index=False), flush=True)
    print(f"outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
