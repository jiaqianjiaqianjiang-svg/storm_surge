from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent / "recovered_caribbean_report_20260812"
OUT.mkdir(parents=True, exist_ok=True)

LEADS = np.array([1, 3, 6, 12, 24])

DIRECT_ROLLING = {
    "Persistence": [1.031, 2.103, 2.864, 2.902, 2.718],
    "Ridge direct": [0.910, 1.673, 2.099, 2.351, 2.694],
    "Ridge rolling": [0.898, 1.668, 2.122, 2.348, 2.595],
    "XGBoost direct": [0.967, 1.760, 2.116, 2.229, 2.411],
    "XGBoost rolling": [0.978, 1.786, 2.249, 2.442, 2.597],
    "Dual-CNN direct": [1.217, 1.786, 2.112, 2.283, 2.502],
    "Dual-CNN rolling": [0.898, 1.646, 2.059, 2.288, 2.597],
}

ABLATION = {
    "XGB-Surge": [0.963, 1.810, 2.251, 2.390, 2.501],
    "XGB-Past-ERA5": [0.983, 1.782, 2.139, 2.292, 2.484],
    "XGB-Future-ERA5": [0.967, 1.760, 2.116, 2.229, 2.411],
    "CNN-Surge": [1.054, 1.758, 2.180, 2.383, 2.552],
    "Dual-CNN-Past": [1.217, 1.786, 2.112, 2.283, 2.502],
    "Dual-CNN-Future": [1.291, 1.910, 2.132, 2.376, 2.545],
}

TOP5_IMPROVEMENT = [12.92, 18.62, 17.20, 16.58]


def save_table(values: dict[str, list[float]], name: str) -> None:
    frame = pd.DataFrame(values, index=LEADS)
    frame.index.name = "lead_hours"
    frame.to_csv(OUT / name, encoding="utf-8-sig")


def plot_direct_rolling() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.3), constrained_layout=True)
    colors = {"Ridge": "#4c78a8", "XGBoost": "#f58518", "Dual-CNN": "#54a24b"}
    for name, values in DIRECT_ROLLING.items():
        if name == "Persistence":
            ax.plot(LEADS, values, color="#888888", ls=":", lw=2.2, marker="o", label=name)
            continue
        family = name.split()[0]
        style = "-" if "direct" in name else "--"
        marker = "o" if "direct" in name else "s"
        ax.plot(LEADS, values, color=colors[family], ls=style, lw=2.3, marker=marker, label=name)
    ax.set_title("Prickly Bay 2017: Direct vs Rolling 24-hour Forecast")
    ax.set_xlabel("Lead time (hour)")
    ax.set_ylabel("RMSE (cm)")
    ax.set_xticks(LEADS)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    ax.annotate("Dual-CNN rolling best at 1–6 h", xy=(6, 2.059), xytext=(7.5, 1.25),
                arrowprops={"arrowstyle": "->", "color": "#333333"})
    ax.annotate("XGBoost direct best at 12–24 h", xy=(24, 2.411), xytext=(13.3, 2.05),
                arrowprops={"arrowstyle": "->", "color": "#333333"})
    fig.savefig(OUT / "rmse_direct_vs_rolling_recovered.png", dpi=220)
    plt.close(fig)


def plot_ablation() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), sharey=True, constrained_layout=True)
    palettes = ["#9ecae9", "#4292c6", "#084594"], ["#a1d99b", "#41ab5d", "#006d2c"]
    groups = [list(ABLATION)[:3], list(ABLATION)[3:]]
    titles = ["XGBoost ablation", "Matched CNN ablation"]
    for ax, group, colors, title in zip(axes, groups, palettes, titles):
        for name, color in zip(group, colors):
            ax.plot(LEADS, ABLATION[name], marker="o", lw=2.4, color=color, label=name)
        ax.set_title(title)
        ax.set_xlabel("Lead time (hour)")
        ax.set_xticks(LEADS)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("RMSE (cm)")
    fig.suptitle("Prickly Bay 2017: Minimal ERA5 Ablation")
    fig.savefig(OUT / "rmse_era5_ablation_recovered.png", dpi=220)
    plt.close(fig)


def plot_top5() -> None:
    leads = np.array([3, 6, 12, 24])
    fig, ax = plt.subplots(figsize=(8.8, 5.3), constrained_layout=True)
    bars = ax.bar(leads.astype(str), TOP5_IMPROVEMENT, color=["#9ecae1", "#2171b5", "#4292c6", "#6baed6"])
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in TOP5_IMPROVEMENT], padding=4)
    ax.set_ylim(0, 21)
    ax.set_title("XGB-Future-ERA5 vs XGB-Surge: Top 5% Surge RMSE Improvement")
    ax.set_xlabel("Lead time (hour)")
    ax.set_ylabel("RMSE improvement (%)")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "top5_era5_improvement_recovered.png", dpi=220)
    plt.close(fig)


save_table(DIRECT_ROLLING, "direct_vs_rolling_selected_leads.csv")
save_table(ABLATION, "era5_ablation_selected_leads.csv")
pd.DataFrame({"lead_hours": [3, 6, 12, 24], "rmse_improvement_percent": TOP5_IMPROVEMENT}).to_csv(
    OUT / "top5_era5_improvement.csv", index=False, encoding="utf-8-sig"
)
plot_direct_rolling()
plot_ablation()
plot_top5()
print(OUT)
