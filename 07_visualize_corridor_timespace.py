from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\Sabin\Streetlight-Data-Processing")
RESULTS_DIR = PROJECT_ROOT / "MOE for selected links" / "Results"
INPUT_FILE = RESULTS_DIR / "corridor_moe_5min.csv"
PLOT_DIR = RESULTS_DIR / "Corridor Time-Space Visualizations"

FIGURES = [
    ("avg_travel_time_s", "Average Travel Time (s)", "01_average_travel_time_timespace.png", "robust"),
    ("slow_movement_count", "Slow Movement Count", "02_slow_movement_count_timespace.png", "robust"),
    ("slow_movement_pct", "Slow Movement Percentage (%)", "03_slow_movement_percentage_timespace.png", "percent"),
    ("DSH", "Degree of Speed Harmonization (mph)", "04_DSH_timespace.png", "robust"),
    ("speed_reduction_pct", "Speed Reduction Percentage (%)", "05_speed_reduction_percentage_timespace.png", "percent_clip"),
]


def build_grid(df, metric):
    links = (
        df[["corridor_order", "link_key", "distance_start_ft", "distance_end_ft"]]
        .drop_duplicates("link_key")
        .sort_values("corridor_order")
        .reset_index(drop=True)
    )
    times = pd.DatetimeIndex(sorted(pd.to_datetime(df["time_bin"], errors="coerce").dropna().unique()))
    pivot = (
        df.pivot_table(index="link_key", columns="time_bin", values=metric, aggfunc="mean", observed=False)
        .reindex(index=links["link_key"], columns=times)
    )

    y_edges = np.concatenate([[float(links["distance_start_ft"].iloc[0])], links["distance_end_ft"].to_numpy(dtype=float)])
    step = pd.Timedelta(minutes=5) if len(times) < 2 else pd.Series(times).diff().dropna().median()
    x_times = list(times) + [times[-1] + step]
    x_edges = mdates.date2num(pd.to_datetime(x_times))
    return links, x_edges, y_edges, pivot.to_numpy(dtype=float)


def limits(values, mode):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0, values
    if mode == "percent":
        return 0.0, 100.0, values
    if mode == "percent_clip":
        return 0.0, 100.0, np.clip(values, 0.0, 100.0)
    positive = finite[finite >= 0]
    vmax = np.nanpercentile(positive, 95) if positive.size else np.nanmax(finite)
    return 0.0, (vmax if np.isfinite(vmax) and vmax > 0 else 1.0), values


def plot_metric(df, metric, label, filename, mode):
    links, x_edges, y_edges, values = build_grid(df, metric)
    vmin, vmax, values = limits(values, mode)
    masked = np.ma.masked_invalid(values)

    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("lightgray")

    fig, ax = plt.subplots(figsize=(14, 8))
    mesh = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    ax.set_title(label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Corridor Distance (ft)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    for distance in links["distance_end_ft"].iloc[:-1]:
        ax.axhline(distance, linewidth=0.25, alpha=0.25)

    colorbar = fig.colorbar(mesh, ax=ax, pad=0.025)
    colorbar.set_label(label)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / filename}")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE, dtype={"link_key": "string"})
    df["time_bin"] = pd.to_datetime(df["time_bin"], errors="coerce")
    for metric, label, filename, mode in FIGURES:
        plot_metric(df, metric, label, filename, mode)


if __name__ == "__main__":
    main()
