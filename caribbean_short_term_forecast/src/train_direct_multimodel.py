"""Train first-round direct 24-hour models and evaluate only on 2017."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .dataset_builder import fit_scalers
    from .direct_forecast_models import (
        ArrayDirectDataset,
        DirectDualCNN,
        DirectGRU,
        DirectMLP,
        GridDirectDataset,
    )
    from .direct_multistep_baselines import (
        PRIMARY_LEADS,
        build_feature_matrix,
        direct_labels,
        direct_origins,
        hourly_atmosphere_valid,
        summarise_atmosphere,
    )
    from .evaluate import calculate_metrics
    from .train_station import load_prepared
except ImportError:
    from dataset_builder import fit_scalers
    from direct_forecast_models import (
        ArrayDirectDataset,
        DirectDualCNN,
        DirectGRU,
        DirectMLP,
        GridDirectDataset,
    )
    from direct_multistep_baselines import (
        PRIMARY_LEADS,
        build_feature_matrix,
        direct_labels,
        direct_origins,
        hourly_atmosphere_valid,
        summarise_atmosphere,
    )
    from evaluate import calculate_metrics
    from train_station import load_prepared


MODULE_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = (
    "persistence",
    "combined_ridge",
    "xgboost",
    "mlp",
    "gru",
    "dual_cnn_past",
    "dual_cnn_future",
)
DISPLAY_NAMES = {
    "persistence": "Persistence",
    "combined_ridge": "Combined-Ridge",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "gru": "GRU",
    "dual_cnn_past": "Dual-CNN-Past",
    "dual_cnn_future": "Dual-CNN-Future",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="prickly_bay")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grid-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_gru_sequences(
    summaries: np.ndarray,
    surge: np.ndarray,
    origins: np.ndarray,
    weather_scaler: StandardScaler,
    surge_mean: float,
    surge_scale: float,
    input_steps: int = 24,
    output_steps: int = 24,
) -> np.ndarray:
    sequences = np.zeros(
        (len(origins), input_steps + output_steps, 14), dtype=np.float32
    )
    for row, origin in enumerate(origins):
        weather = summaries[
            origin - input_steps + 1 : origin + output_steps + 1
        ].reshape(input_steps + output_steps, -1)
        sequences[row, :, :12] = weather_scaler.transform(weather)
        history = np.asarray(
            surge[origin - input_steps + 1 : origin + 1], dtype=np.float32
        )
        sequences[row, :input_steps, 12] = (history - surge_mean) / surge_scale
        sequences[row, :input_steps, 13] = 1.0
    return sequences


def fit_gru_weather_scaler(
    summaries: np.ndarray,
    origins: np.ndarray,
    input_steps: int = 24,
    output_steps: int = 24,
) -> StandardScaler:
    values = np.concatenate(
        [
            summaries[o - input_steps + 1 : o + output_steps + 1].reshape(
                input_steps + output_steps, -1
            )
            for o in origins
        ],
        axis=0,
    )
    return StandardScaler().fit(values)


def _forward_batch(
    model: nn.Module,
    batch: tuple[torch.Tensor, ...] | list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 2:
        features, target = batch
        return model(features.to(device, non_blocking=True)), target.to(device, non_blocking=True)
    atmosphere, history, target = batch
    return (
        model(
            atmosphere.to(device, non_blocking=True),
            history.to(device, non_blocking=True),
        ),
        target.to(device, non_blocking=True),
    )


def train_neural_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    output: Path,
    device: torch.device,
    surge_mean: float,
    surge_scale: float,
    epochs: int,
    patience: int,
    seed: int,
    extra_checkpoint: dict[str, Any],
    reuse_checkpoint: bool,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    model_dir = output / name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best_model.pth"
    primary_indices = torch.tensor([lead - 1 for lead in PRIMARY_LEADS], device=device)
    if not (reuse_checkpoint and checkpoint_path.is_file()):
        set_seed(seed)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()
        amp_enabled = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        best_score = float("inf")
        stale = 0
        history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            model.train()
            train_total = 0.0
            train_count = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    predicted, target = _forward_batch(model, batch, device)
                    loss = criterion(predicted, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_total += float(loss.detach()) * len(target)
                train_count += len(target)
            model.eval()
            validation_total = 0.0
            validation_primary = 0.0
            validation_count = 0
            with torch.inference_mode():
                for batch in validation_loader:
                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        predicted, target = _forward_batch(model, batch, device)
                        full_loss = criterion(predicted, target)
                        primary_loss = criterion(
                            torch.index_select(predicted, 1, primary_indices),
                            torch.index_select(target, 1, primary_indices),
                        )
                    validation_total += float(full_loss) * len(target)
                    validation_primary += float(primary_loss) * len(target)
                    validation_count += len(target)
            record = {
                "epoch": epoch,
                "train_scaled_mse": train_total / train_count,
                "validation_scaled_mse": validation_total / validation_count,
                "validation_primary_scaled_mse": validation_primary / validation_count,
                "validation_primary_rmse_cm": float(
                    np.sqrt(validation_primary / validation_count) * surge_scale * 100
                ),
            }
            history.append(record)
            print(
                f"{name} epoch={epoch} primary_rmse_cm={record['validation_primary_rmse_cm']:.4f}",
                flush=True,
            )
            score = record["validation_primary_scaled_mse"]
            if score < best_score - 1e-8:
                best_score = score
                stale = 0
                torch.save(
                    {
                        **model.architecture_config(),
                        "model_state_dict": model.state_dict(),
                        "surge_mean": surge_mean,
                        "surge_scale": surge_scale,
                        "seed": seed,
                        "selection_metric": "mean 2017 scaled MSE at 1/3/6/12/24 h",
                        **extra_checkpoint,
                    },
                    checkpoint_path,
                )
            else:
                stale += 1
                if stale >= patience:
                    print(f"{name} early stopping at epoch {epoch}", flush=True)
                    break
        pd.DataFrame(history).to_csv(model_dir / "loss_history.csv", index=False)
    else:
        history = []
        print(f"{name}: reusing {checkpoint_path}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    scaled_predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in validation_loader:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                predicted, _ = _forward_batch(model, batch, device)
            scaled_predictions.append(predicted.float().cpu().numpy())
    scaled = np.concatenate(scaled_predictions, axis=0)
    return scaled * surge_scale + surge_mean, history


def calculate_all_metrics(
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    surge: np.ndarray,
    origins: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference = predictions["combined_ridge"]
    for lead_index in range(observed.shape[1]):
        lead = lead_index + 1
        target = observed[:, lead_index]
        prior = np.asarray(surge[origins + lead_index], dtype=float)
        top_threshold = float(np.quantile(np.abs(target), 0.95))
        top_mask = np.abs(target) >= top_threshold
        rises = target - prior
        positive = rises[rises > 0]
        rise_threshold = float(np.quantile(positive, 0.90)) if len(positive) else np.nan
        rapid_mask = rises >= rise_threshold if np.isfinite(rise_threshold) else np.zeros(len(target), bool)
        reference_mse = float(np.mean((reference[:, lead_index] - target) ** 2))
        for name in MODEL_NAMES:
            predicted = predictions[name][:, lead_index]
            overall = calculate_metrics(target, predicted)
            top = calculate_metrics(target[top_mask], predicted[top_mask])
            rapid = calculate_metrics(target[rapid_mask], predicted[rapid_mask])
            mse = float(np.mean((predicted - target) ** 2))
            rows.append(
                {
                    "lead_hours": lead,
                    "model": name,
                    **overall,
                    "skill_vs_combined_ridge": 1 - mse / reference_mse,
                    "top5_n": int(top["n"]),
                    "top5_threshold_cm": top_threshold * 100,
                    "top5_rmse_cm": top["rmse_cm"],
                    "rapid_rise_n": int(rapid["n"]),
                    "rapid_rise_threshold_cm_per_hour": rise_threshold * 100,
                    "rapid_rise_rmse_cm": rapid["rmse_cm"],
                }
            )
    return pd.DataFrame(rows)


def plot_rmse(metrics: pd.DataFrame, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for name in MODEL_NAMES:
        subset = metrics[metrics.model == name]
        ax.plot(subset.lead_hours, subset.rmse_cm, linewidth=1.8, label=DISPLAY_NAMES[name])
    ax.set(xlabel="Direct forecast lead (hours)", ylabel="RMSE (cm)")
    ax.set_xticks([1, 3, 6, 12, 18, 24])
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Prickly Bay 2017 direct 24-hour model comparison")
    fig.tight_layout()
    fig.savefig(destination, dpi=400)
    plt.close(fig)


def plot_events(
    output: Path,
    times: pd.DatetimeIndex,
    origins: np.ndarray,
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    event_count: int = 3,
) -> list[dict[str, Any]]:
    scores = np.max(np.abs(observed), axis=1)
    selected: list[int] = []
    for candidate in np.argsort(scores)[::-1]:
        if all(abs(int(origins[candidate]) - int(origins[other])) >= 72 for other in selected):
            selected.append(int(candidate))
        if len(selected) == event_count:
            break
    details = []
    for number, row in enumerate(selected, start=1):
        valid_times = times[origins[row] + np.arange(1, 25)]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(valid_times, observed[row] * 100, color="black", linewidth=2.2, label="Observed")
        for name in MODEL_NAMES:
            ax.plot(valid_times, predictions[name][row] * 100, linewidth=1.2, label=DISPLAY_NAMES[name])
        ax.set(xlabel="Valid time", ylabel="Storm surge (cm)")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=7)
        ax.set_title(f"Direct 24-hour strong-event forecast from {times[origins[row]]:%Y-%m-%d %H:%M}")
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
    table: pd.DataFrame,
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
) -> None:
    primary = metrics[metrics.lead_hours.isin(PRIMARY_LEADS)]
    skill = primary.pivot(
        index="lead_hours", columns="model", values="skill_vs_combined_ridge"
    )
    top5 = primary.pivot(index="lead_hours", columns="model", values="top5_rmse_cm")
    best_rows = primary.loc[primary.groupby("lead_hours").rmse_cm.idxmin()]
    best_lines = "\n".join(
        f"- {int(row.lead_hours)}小时：{DISPLAY_NAMES[row.model]}，RMSE {row.rmse_cm:.3f} cm。"
        for row in best_rows.itertuples()
    )
    content = f"""# Prickly Bay 2017直接24小时多模型比较

## 实验设置

- 训练：2011—2016；模型选择与比较：2017；2018未加载。
- 统一输入历史24小时，统一输出连续未来24小时，重点评价1、3、6、12、24小时。
- 未来ERA5均为再分析真值，属于已知未来大气强迫历史回算，并非业务预报。
- 神经网络统一seed 42、普通MSE和相同早停指标。
- XGBoost使用24个独立模型分别预测1—24小时，并在CUDA上训练。
- 训练样本{metadata['train_samples']}，2017共同验证样本{metadata['validation_samples']}。

## 重点提前量RMSE（cm）

{markdown_table(table)}

## 各重点提前量最优模型

{best_lines}

## 主要发现

1. Combined-Ridge在1、3、6小时最低，说明短提前量仍主要受增水自相关控制。
2. XGBoost在12小时和24小时最低：相对Combined-Ridge的MSE技能分别为
{skill.loc[12, 'xgboost'] * 100:.2f}%和{skill.loc[24, 'xgboost'] * 100:.2f}%，
对应RMSE分别由{table.set_index('lead_hours').loc[12, 'combined_ridge']:.3f}降至
{table.set_index('lead_hours').loc[12, 'xgboost']:.3f} cm、由
{table.set_index('lead_hours').loc[24, 'combined_ridge']:.3f}降至
{table.set_index('lead_hours').loc[24, 'xgboost']:.3f} cm。
3. Dual-CNN-Past在12和24小时相对Combined-Ridge的MSE技能为
{skill.loc[12, 'dual_cnn_past'] * 100:.2f}%和{skill.loc[24, 'dual_cnn_past'] * 100:.2f}%，
说明原始空间场在中长提前量可能有价值，但总体仍未超过XGBoost。
4. Dual-CNN-Future在1、3、6、12、24小时均未超过Dual-CNN-Past；第一轮没有证据表明把未来ERA5原始网格直接拼接为通道能够改善结果。
5. MLP和GRU第一版在所有重点提前量均未超过Combined-Ridge，不进入下一轮优先候选。
6. Top 5%强增水中，XGBoost在6、12、24小时RMSE分别为
{top5.loc[6, 'xgboost']:.3f}、{top5.loc[12, 'xgboost']:.3f}、{top5.loc[24, 'xgboost']:.3f} cm，
均明显低于Combined-Ridge的{top5.loc[6, 'combined_ridge']:.3f}、
{top5.loc[12, 'combined_ridge']:.3f}、{top5.loc[24, 'combined_ridge']:.3f} cm。

## 下一阶段候选

- 传统机器学习候选：XGBoost。
- 空间场候选：Dual-CNN-Past。
- 短提前量强参照：Combined-Ridge。
- 下一步可只给XGBoost补充一步滚动24小时实验；现有Ridge和Dual-CNN滚动结果继续复用。
- 本轮仍属于单seed模型开发。神经网络checkpoint使用2017选择，因此2017指标用于筛选，不能作为最终无偏泛化误差。

## 模型说明

- Combined-Ridge、XGBoost和MLP使用历史增水及过去＋未来ERA5统计特征。
- GRU使用48小时统计序列；未来24小时没有增水输入，并通过已知增水掩码与历史段区分。
- Dual-CNN-Past只使用过去24小时ERA5原始40×40网格。
- Dual-CNN-Future使用过去和未来各24小时ERA5原始40×40网格。

## 评价文件

- `metrics_all_leads.csv`：1—24小时RMSE、MAE、Bias、r、R²、技能、Top 5%和快速上涨指标。
- `rmse_primary_leads.csv`：重点提前量统一表。
- `validation_predictions.csv`：全部2017起报时刻的连续24小时观测和预测。
- `rmse_vs_lead.png`：1—24小时RMSE曲线。
- `strong_event_*.png`：典型强增水过程。
"""
    (output / "2017_direct_multimodel_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dataset_path = args.dataset_path or (
        MODULE_ROOT / "outputs" / "processed" / args.station / "aligned_dataset"
    )
    baseline_path = args.baseline_predictions or (
        MODULE_ROOT / "outputs" / "experiments" / args.station /
        "direct_multistep_baselines_2017" / "validation_predictions.csv"
    )
    output = args.output_dir or (
        MODULE_ROOT / "outputs" / "experiments" / args.station /
        "direct_multimodel_2017_seed42"
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
        raise AssertionError("2018 entered direct model development")
    atmosphere_valid = hourly_atmosphere_valid(atmosphere)
    train_origins = direct_origins(times, atmosphere_valid, surge, set(range(2011, 2017)))
    validation_origins = direct_origins(times, atmosphere_valid, surge, {2017})
    train_labels = direct_labels(surge, train_origins)
    validation_labels = direct_labels(surge, validation_origins)
    print(
        f"loaded through {times.max()}; train={len(train_origins)} "
        f"validation={len(validation_origins)} device={device}", flush=True,
    )

    baseline = pd.read_csv(baseline_path, parse_dates=["forecast_origin"])
    expected_origins = pd.DatetimeIndex(times[validation_origins])
    baseline = baseline.set_index("forecast_origin").reindex(expected_origins)
    if baseline.isna().any().any():
        raise ValueError("Direct ridge baseline does not cover all common origins")
    predictions: dict[str, np.ndarray] = {
        "persistence": np.column_stack(
            [baseline[f"persistence_{lead:02d}h_m"] for lead in range(1, 25)]
        ).astype(np.float32),
        "combined_ridge": np.column_stack(
            [baseline[f"combined_ridge_past_future_{lead:02d}h_m"] for lead in range(1, 25)]
        ).astype(np.float32),
    }
    if not np.allclose(
        validation_labels,
        np.column_stack([baseline[f"observed_{lead:02d}h_m"] for lead in range(1, 25)]),
    ):
        raise AssertionError("Baseline and neural labels do not align")

    summaries = summarise_atmosphere(atmosphere)
    combined_train = build_feature_matrix(
        summaries, surge, train_origins, True, "past_future"
    )
    combined_validation = build_feature_matrix(
        summaries, surge, validation_origins, True, "past_future"
    )

    tree_path = output / "xgboost_models.joblib"
    if args.reuse_checkpoints and tree_path.is_file():
        xgboost_models = joblib.load(tree_path)
    else:
        from xgboost import XGBRegressor

        print("fitting 24 direct XGBoost models", flush=True)
        xgboost_models = []
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
                random_state=args.seed,
            )
            model.fit(combined_train, train_labels[:, lead])
            xgboost_models.append(model)
            print(f"xgboost lead={lead + 1}/24 complete", flush=True)
        joblib.dump(xgboost_models, tree_path, compress=3)
    predictions["xgboost"] = np.column_stack(
        [model.predict(combined_validation) for model in xgboost_models]
    ).astype(np.float32)

    training_positions = np.flatnonzero((times.year >= 2011) & (times.year <= 2016))
    raw_scalers = fit_scalers(
        atmosphere, surge, int(training_positions[-1]) + 1, int(training_positions[0])
    )
    surge_mean = float(raw_scalers["surge"].mean.reshape(-1)[0])
    surge_scale = float(raw_scalers["surge"].scale.reshape(-1)[0])
    scaled_train_labels = (train_labels - surge_mean) / surge_scale
    scaled_validation_labels = (validation_labels - surge_mean) / surge_scale

    feature_scaler = StandardScaler().fit(combined_train)
    mlp_train = feature_scaler.transform(combined_train).astype(np.float32)
    mlp_validation = feature_scaler.transform(combined_validation).astype(np.float32)
    train_loader = DataLoader(
        ArrayDirectDataset(mlp_train, scaled_train_labels),
        batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        ArrayDirectDataset(mlp_validation, scaled_validation_labels),
        batch_size=args.batch_size, pin_memory=device.type == "cuda",
    )
    predictions["mlp"], _ = train_neural_model(
        "mlp", DirectMLP(combined_train.shape[1]), train_loader, validation_loader,
        output, device, surge_mean, surge_scale, args.epochs, args.patience,
        args.seed, {"feature_scaler": feature_scaler, "feature_definition": "combined past+future ERA5 statistics"},
        args.reuse_checkpoints,
    )
    del mlp_train, mlp_validation, train_loader, validation_loader

    print("building GRU sequences", flush=True)
    gru_weather_scaler = fit_gru_weather_scaler(summaries, train_origins)
    gru_train = build_gru_sequences(
        summaries, surge, train_origins, gru_weather_scaler, surge_mean, surge_scale
    )
    gru_validation = build_gru_sequences(
        summaries, surge, validation_origins, gru_weather_scaler, surge_mean, surge_scale
    )
    train_loader = DataLoader(
        ArrayDirectDataset(gru_train, scaled_train_labels),
        batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        ArrayDirectDataset(gru_validation, scaled_validation_labels),
        batch_size=args.batch_size, pin_memory=device.type == "cuda",
    )
    predictions["gru"], _ = train_neural_model(
        "gru", DirectGRU(), train_loader, validation_loader, output, device,
        surge_mean, surge_scale, args.epochs, args.patience, args.seed,
        {"weather_scaler": gru_weather_scaler, "sequence_definition": "48 hours; past+future ERA5 stats, past surge and known mask"},
        args.reuse_checkpoints,
    )
    del gru_train, gru_validation, train_loader, validation_loader, combined_train, combined_validation

    atmosphere_mean = raw_scalers["atmosphere"].mean.reshape(-1)
    atmosphere_scale = raw_scalers["atmosphere"].scale.reshape(-1)
    for name, include_future in (("dual_cnn_past", False), ("dual_cnn_future", True)):
        atmosphere_steps = 48 if include_future else 24
        train_dataset = GridDirectDataset(
            atmosphere, surge, train_origins, train_labels,
            atmosphere_mean, atmosphere_scale, surge_mean, surge_scale,
            include_future,
        )
        validation_dataset = GridDirectDataset(
            atmosphere, surge, validation_origins, validation_labels,
            atmosphere_mean, atmosphere_scale, surge_mean, surge_scale,
            include_future,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.grid_batch_size, shuffle=True,
            pin_memory=device.type == "cuda", num_workers=0,
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=args.grid_batch_size,
            pin_memory=device.type == "cuda", num_workers=0,
        )
        predictions[name], _ = train_neural_model(
            name, DirectDualCNN(atmosphere_steps), train_loader, validation_loader,
            output, device, surge_mean, surge_scale, args.epochs, args.patience,
            args.seed,
            {
                "atmosphere_scaler": raw_scalers["atmosphere"].state_dict(),
                "forcing": "past+future ERA5 raw grid" if include_future else "past ERA5 raw grid",
            },
            args.reuse_checkpoints,
        )
        del train_dataset, validation_dataset, train_loader, validation_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = calculate_all_metrics(
        validation_labels, predictions, surge, validation_origins
    )
    metrics.to_csv(output / "metrics_all_leads.csv", index=False)
    table = metrics[metrics.lead_hours.isin(PRIMARY_LEADS)].pivot(
        index="lead_hours", columns="model", values="rmse_cm"
    ).reset_index()
    table.insert(1, "valid_samples", len(validation_origins))
    table = table[["lead_hours", "valid_samples", *MODEL_NAMES]]
    table.to_csv(output / "rmse_primary_leads.csv", index=False)

    prediction_data: dict[str, Any] = {"forecast_origin": expected_origins}
    for lead in range(1, 25):
        prediction_data[f"valid_time_{lead:02d}h"] = times[validation_origins + lead]
        prediction_data[f"observed_{lead:02d}h_m"] = validation_labels[:, lead - 1]
        for name in MODEL_NAMES:
            prediction_data[f"{name}_{lead:02d}h_m"] = predictions[name][:, lead - 1]
    pd.DataFrame(prediction_data).to_csv(output / "validation_predictions.csv", index=False)
    plot_rmse(metrics, output / "rmse_vs_lead.png")
    events = plot_events(output, times, validation_origins, validation_labels, predictions)
    metadata = {
        "experiment": "direct 24-hour multimodel first round",
        "loaded_time_max": times.max().isoformat(),
        "2018_loaded": False,
        "train_years": [2011, 2016],
        "validation_year": 2017,
        "train_samples": int(len(train_origins)),
        "validation_samples": int(len(validation_origins)),
        "seed": args.seed,
        "loss": "ordinary MSE across all 24 outputs",
        "checkpoint_selection": "mean 2017 MSE at 1/3/6/12/24 h",
        "tree_model": "24 independent XGBoost 3.2.0 models",
        "device": str(device),
        "events": events,
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output, table, metadata, metrics)
    print(table.to_string(index=False), flush=True)
    print(f"outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
