from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\Sabin\Streetlight-Data-Processing")
RESULTS_DIR = PROJECT_ROOT / "MOE for selected links" / "Results"
INPUT_FILE = RESULTS_DIR / "link_moe_5min.csv"
PLOT_DIR = RESULTS_DIR / "MOE Heatmaps"

TOP_N_LINKS = 30

METRICS = [
    ("avg_travel_time_s", "Average Travel Time (s)", "01_average_travel_time_heatmap.png"),
    ("slow_movement_count", "Slow Movement Count", "02_slow_movement_count_heatmap.png"),
    ("slow_movement_pct", "Slow Movement Percentage (%)", "03_slow_movement_percentage_heatmap.png"),
    ("DSH", "Degree of Speed Harmonization (mph)", "04_DSH_heatmap.png"),
]


def select_top_links(df):
    return (
        df.groupby("link_key", observed=True)["waypoint_count"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_N_LINKS)
        .index.tolist()
    )


def plot_heatmap(df, links, metric, label, filename):
    subset = df[df["link_key"].isin(links)].copy()
    subset["link_key"] = pd.Categorical(subset["link_key"], categories=links, ordered=True)

    matrix = (
        subset.pivot_table(
            index="link_key",
            columns="time_bin",
            values=metric,
            aggfunc="mean",
            observed=False,
        )
        .reindex(links)
    )

    values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("lightgray")

    fig, ax = plt.subplots(figsize=(14, 10))
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(label)
    ax.set_xlabel("5-minute time bin")
    ax.set_ylabel("Link key")

    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index.astype(str))
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(
        [pd.Timestamp(value).strftime("%H:%M") for value in matrix.columns],
        rotation=45,
        ha="right",
    )

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(label)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE, dtype={"link_key": "string"})
    df["time_bin"] = pd.to_datetime(df["time_bin"], errors="coerce")
    links = select_top_links(df)

    for metric, label, filename in METRICS:
        plot_heatmap(df, links, metric, label, filename)
        print(f"Saved: {PLOT_DIR / filename}")


if __name__ == "__main__":
    main()
