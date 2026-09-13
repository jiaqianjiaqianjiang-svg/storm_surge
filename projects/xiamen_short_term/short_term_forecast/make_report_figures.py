"""Create presentation figures for the short-term CNN forecast discussion."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
SURGE_CSV = HERE / "results" / "xiamen_1985_hourly_surge.csv"
OUTPUT_DIR = HERE / "figures_for_report"
PROJECT_ROOT = HERE.parent
ERA_ROOT = PROJECT_ROOT / "data" / "ERA20C_1985"
SITE_LAT = 24.45
SITE_LON = 118.067


COLORS = {
    "u10": "#2D6A9F",
    "v10": "#2E8B57",
    "slp": "#D98C2B",
    "cnn": "#5C4B8A",
    "output": "#B33A3A",
    "ink": "#263238",
}


def rounded_box(ax, x, y, width, height, text, facecolor, fontsize=11, edgecolor="none"):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, color="white")
    return box


def arrow(ax, start, end, text=None, y_offset=0.16):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color=COLORS["ink"])
    ax.add_patch(patch)
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + y_offset, text,
                ha="center", va="bottom", fontsize=9, color=COLORS["ink"])


def weather_stack(ax, x, y, label, forecast=False):
    size = 0.78
    offsets = ((0.16, 0.16, COLORS["slp"]), (0.08, 0.08, COLORS["v10"]), (0, 0, COLORS["u10"]))
    for dx, dy, color in offsets:
        ax.add_patch(Rectangle((x + dx, y + dy), size, size, facecolor=color, edgecolor="white", linewidth=1.2))
    ax.text(x + 0.48, y + 0.43, "U10", ha="center", va="center", color="white", fontsize=10, weight="bold")
    ax.text(x + 0.5, y - 0.18, label, ha="center", va="top", fontsize=10, color=COLORS["ink"])
    ax.text(x + 0.5, y - 0.43, "3 variables, 40 x 40", ha="center", va="top", fontsize=8, color="#546E7A")
    if forecast:
        ax.text(x + 0.5, y + 1.12, "new NWP grid", ha="center", fontsize=8, color=COLORS["output"], weight="bold")


def make_workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    ax.text(7, 7.25, "Hybrid short-term forecast: meteorological CNN + autoregressive surge state",
            ha="center", va="center", fontsize=18, weight="bold", color=COLORS["ink"])

    ax.text(0.35, 6.4, "Forecast cycle k", fontsize=12, weight="bold", color=COLORS["ink"])
    weather_stack(ax, 0.7, 5.0, "X(k-t+1)")
    ax.text(2.15, 5.4, "...", fontsize=22, color="#78909C")
    weather_stack(ax, 2.75, 5.0, "X(k)")
    rounded_box(ax, 4.55, 4.93, 1.75, 1.05, "CNN weather\nencoder", COLORS["cnn"], fontsize=11)
    rounded_box(ax, 7.15, 4.93, 1.65, 1.05, "Feature fusion", "#607D8B", fontsize=11)
    rounded_box(ax, 9.7, 4.93, 1.7, 1.05, "Surge forecast\ny_hat(k+1)", COLORS["output"], fontsize=11)
    arrow(ax, (3.7, 5.45), (4.5, 5.45))
    arrow(ax, (6.35, 5.45), (7.1, 5.45))
    arrow(ax, (8.85, 5.45), (9.65, 5.45))

    rounded_box(ax, 4.55, 3.65, 1.75, 0.78, "Surge history\ny(k-t+1 ... k)", "#2E8B57", fontsize=10)
    arrow(ax, (6.35, 4.04), (7.35, 4.9), "autoregressive state", y_offset=0.02)

    ax.text(0.35, 2.75, "Forecast cycle k+1", fontsize=12, weight="bold", color=COLORS["ink"])
    weather_stack(ax, 0.7, 1.35, "X(k-t+2)")
    ax.text(2.15, 1.75, "...", fontsize=22, color="#78909C")
    weather_stack(ax, 2.75, 1.35, "X_NWP(k+1)", forecast=True)
    rounded_box(ax, 4.55, 1.28, 1.75, 1.05, "CNN weather\nencoder", COLORS["cnn"], fontsize=11)
    rounded_box(ax, 7.15, 1.28, 1.65, 1.05, "Feature fusion", "#607D8B", fontsize=11)
    rounded_box(ax, 9.7, 1.28, 1.7, 1.05, "Surge forecast\ny_hat(k+2)", COLORS["output"], fontsize=11)
    arrow(ax, (3.7, 1.8), (4.5, 1.8))
    arrow(ax, (6.35, 1.8), (7.1, 1.8))
    arrow(ax, (8.85, 1.8), (9.65, 1.8))

    rounded_box(ax, 4.55, 0.08, 1.75, 0.78, "Updated surge history\n[..., y(k), y_hat(k+1)]", "#2E8B57", fontsize=9)
    arrow(ax, (6.35, 0.47), (7.35, 1.25), "append predicted surge", y_offset=0.02)
    feedback = FancyArrowPatch(
        (10.55, 4.9), (5.5, 0.9),
        connectionstyle="arc3,rad=-0.28", arrowstyle="-|>", mutation_scale=16,
        linewidth=2.0, linestyle="--", color=COLORS["output"],
    )
    ax.add_patch(feedback)
    ax.text(10.9, 3.4, "feedback y_hat(k+1)", color=COLORS["output"], fontsize=10, weight="bold")

    ax.text(12.0, 5.65, "Meteorological inputs:\nU10, V10, SLP only",
            ha="left", va="center", fontsize=10, color=COLORS["ink"], weight="bold")
    ax.text(12.0, 1.8, "Both windows roll:\nweather grids + surge state",
            ha="left", va="center", fontsize=10, color=COLORS["ink"], weight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cnn_rolling_workflow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_surge() -> pd.DataFrame:
    frame = pd.read_csv(SURGE_CSV, index_col=0, parse_dates=True)
    frame.index.name = "time"
    return frame


def make_tide_separation_figure(frame: pd.DataFrame) -> None:
    peak_time = frame["storm_surge_m"].idxmax()
    event = frame.loc[peak_time - pd.Timedelta(hours=72): peak_time + pd.Timedelta(hours=72)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=(1.15, 1))
    axes[0].plot(event.index, event["sea_level_m"], color="#1F77B4", linewidth=1.8, label="Observed sea level")
    axes[0].plot(event.index, event["predicted_tide_m"], color="#D98C2B", linewidth=1.5, label="UTide prediction")
    axes[0].set_ylabel("Sea level (m)")
    axes[0].set_title("Xiamen 1985: tide separation around the maximum surge event", weight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.22)

    surge_cm = event["storm_surge_m"] * 100
    axes[1].plot(event.index, surge_cm, color="#B33A3A", linewidth=1.8, label="Storm surge residual")
    axes[1].fill_between(event.index, 0, surge_cm, where=surge_cm >= 0, color="#E57373", alpha=0.3)
    axes[1].fill_between(event.index, 0, surge_cm, where=surge_cm < 0, color="#64B5F6", alpha=0.25)
    peak_cm = float(frame.loc[peak_time, "storm_surge_m"] * 100)
    axes[1].scatter([peak_time], [peak_cm], color="#7F0000", s=55, zorder=5)
    axes[1].annotate(f"peak = {peak_cm:.1f} cm\n{peak_time:%Y-%m-%d %H:%M}",
                     xy=(peak_time, peak_cm), xytext=(18, -38), textcoords="offset points",
                     arrowprops={"arrowstyle": "->", "color": "#7F0000"}, fontsize=9)
    axes[1].axhline(0, color="#455A64", linewidth=0.8)
    axes[1].set_ylabel("Storm surge (cm)")
    axes[1].set_xlabel("Time")
    axes[1].grid(alpha=0.22)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "xiamen_1985_tide_separation_event.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_annual_overview(frame: pd.DataFrame) -> None:
    daily_max_cm = frame["storm_surge_m"].resample("D").max() * 100
    daily_min_cm = frame["storm_surge_m"].resample("D").min() * 100
    selected_times = []
    for time in daily_max_cm.sort_values(ascending=False).index:
        if all(abs((time - chosen).days) >= 10 for chosen in selected_times):
            selected_times.append(time)
        if len(selected_times) == 5:
            break
    top = daily_max_cm.loc[selected_times].sort_index()

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.fill_between(daily_max_cm.index, daily_min_cm, daily_max_cm, color="#90CAF9", alpha=0.35,
                    label="Daily min-max range")
    ax.plot(daily_max_cm.index, daily_max_cm, color="#B33A3A", linewidth=1.15, label="Daily maximum surge")
    ax.scatter(top.index, top.values, color="#7F0000", s=30, zorder=4, label="Top five days")
    for time, value in top.items():
        ax.annotate(f"{time:%m-%d}\n{value:.0f} cm", (time, value), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(0, color="#455A64", linewidth=0.8)
    ax.set(title="Xiamen 1985 hourly storm-surge residual: annual overview",
           xlabel="Month", ylabel="Storm surge (cm)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "xiamen_1985_surge_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_weather_window(variable_folder: str, variable_name: str, times: pd.DatetimeIndex) -> xr.DataArray:
    path = next((ERA_ROOT / variable_folder).glob("*.grb"))
    dataset = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    data = dataset[variable_name].sel(time=times)
    data = data.sel(
        latitude=slice(SITE_LAT + 5.0, SITE_LAT - 5.0),
        longitude=slice(SITE_LON - 5.0, SITE_LON + 5.0),
    ).sortby("latitude")
    target_lat = np.linspace(float(data.latitude.min()), float(data.latitude.max()), 40)
    target_lon = np.linspace(float(data.longitude.min()), float(data.longitude.max()), 40)
    return data.interp(latitude=target_lat, longitude=target_lon).load()


def make_actual_weather_window(frame: pd.DataFrame) -> None:
    peak_time = frame["storm_surge_m"].idxmax().floor("3h")
    times = pd.date_range(peak_time - pd.Timedelta(hours=9), peak_time, freq="3h")
    variables = [
        ("10U", "u10", "U10 (m/s)", "RdBu_r"),
        ("10V", "v10", "V10 (m/s)", "RdBu_r"),
        ("SLP", "msl", "SLP (hPa)", "coolwarm"),
    ]
    arrays = []
    for folder, name, label, cmap in variables:
        data = load_weather_window(folder, name, times)
        if name == "msl":
            data = data / 100.0
        arrays.append((data, label, cmap))

    fig, axes = plt.subplots(3, 4, figsize=(13.5, 9), constrained_layout=True)
    for row, (data, label, cmap) in enumerate(arrays):
        values = data.values
        if row < 2:
            bound = float(np.nanmax(np.abs(values)))
            vmin, vmax = -bound, bound
        else:
            vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        image = None
        for col, time in enumerate(times):
            ax = axes[row, col]
            image = ax.imshow(
                values[col], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[float(data.longitude.min()), float(data.longitude.max()),
                        float(data.latitude.min()), float(data.latitude.max())],
                aspect="auto",
            )
            ax.scatter(SITE_LON, SITE_LAT, marker="*", s=55, color="black", edgecolor="white", linewidth=0.5)
            if row == 0:
                ax.set_title(pd.Timestamp(time).strftime("%Y-%m-%d\n%H:%M UTC"), fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{label}\nLatitude")
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel("Longitude")
            else:
                ax.set_xticklabels([])
            ax.grid(False)
        fig.colorbar(image, ax=axes[row, :], shrink=0.78, pad=0.015, label=label)

    fig.suptitle(
        "Example rolling meteorological input window near the 1985 maximum surge\n"
        "Four consecutive 3-hour ERA20C steps, each processed to 40 x 40; star = Xiamen",
        fontsize=15, weight="bold",
    )
    fig.savefig(OUTPUT_DIR / "actual_u10_v10_slp_window.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_surge_autocorrelation(frame: pd.DataFrame) -> None:
    surge = frame["storm_surge_m"]
    lags = np.arange(1, 73)
    correlations = np.asarray([surge.autocorr(lag=int(lag)) for lag in lags])

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(lags, correlations, color="#2D6A9F", linewidth=2)
    ax.fill_between(lags, 0, correlations, color="#90CAF9", alpha=0.35)
    candidate_lags = (6, 12, 24, 48)
    for lag in candidate_lags:
        value = correlations[lag - 1]
        ax.scatter([lag], [value], color="#B33A3A", s=42, zorder=4)
        ax.annotate(f"t={lag}h\nr={value:.2f}", (lag, value), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(0, color="#455A64", linewidth=0.8)
    ax.set(
        title="Autocorrelation of hourly Xiamen storm-surge residuals (1985)",
        xlabel="Lag (hour)", ylabel="Autocorrelation",
        xlim=(1, 72), ylim=(min(-0.1, float(correlations.min()) - 0.05), 1.02),
    )
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "surge_history_autocorrelation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_surge()
    make_workflow_figure()
    make_tide_separation_figure(frame)
    make_annual_overview(frame)
    make_actual_weather_window(frame)
    make_surge_autocorrelation(frame)
    print(f"Saved report figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
