from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Paths and settings
MOE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MOE_DIR / "Results"

HOURLY_FILE = RESULTS_DIR / "link_moe_hourly.csv"
FIVE_MIN_FILE = RESULTS_DIR / "link_moe_5min.csv"

FIGURE_DIR = RESULTS_DIR / "Visualizations"
VISUALIZED_LINKS_FILE = FIGURE_DIR / "visualized_links.csv"

TOP_LINKS = 30
FIGURE_DPI = 300


# Load hourly and 5-minute MOE results
def load_results():

    hourly = pd.read_csv(
        HOURLY_FILE,
        dtype={"link_key": "string"},
    )

    five_min = pd.read_csv(
        FIVE_MIN_FILE,
        dtype={"link_key": "string"},
        parse_dates=["time_bin"],
    )

    return hourly, five_min


# Select links using CV waypoint availability only
def select_visualization_links(hourly):

    selected = (
        hourly.sort_values(
            "waypoint_count",
            ascending=False,
        )
        .head(TOP_LINKS)
        .copy()
    )

    return selected


# Prepare one link-by-time matrix for a selected MOE
def prepare_heatmap_data(
    five_min,
    link_order,
    metric,
):

    data = five_min[
        five_min["link_key"].isin(link_order)
    ].copy()

    data["time_label"] = (
        data["time_bin"]
        .dt.strftime("%H:%M")
    )

    matrix = data.pivot(
        index="link_key",
        columns="time_label",
        values=metric,
    )

    matrix = matrix.reindex(
        link_order
    )

    matrix = matrix.reindex(
        sorted(matrix.columns),
        axis=1,
    )

    return matrix


# Plot and save one MOE heatmap
def create_heatmap(
    matrix,
    title,
    colorbar_label,
    output_file,
):

    fig_height = max(
        8,
        len(matrix) * 0.30,
    )

    fig, ax = plt.subplots(
        figsize=(12, fig_height)
    )

    values = matrix.to_numpy(
        dtype=float
    )

    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title(
        title,
        fontsize=14,
        pad=12,
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Link Key"
    )

    ax.set_xticks(
        np.arange(len(matrix.columns))
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(len(matrix.index))
    )

    ax.set_yticklabels(
        matrix.index
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.02,
    )

    colorbar.set_label(
        colorbar_label
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# Create all time-dependent MOE heatmaps
def create_all_heatmaps(
    five_min,
    selected_links,
):

    link_order = (
        selected_links["link_key"]
        .tolist()
    )

    metrics = [
        (
            "avg_travel_time_s",
            "Average Link Travel Time",
            "Average Travel Time (s)",
            "01_average_travel_time_heatmap.png",
        ),
        (
            "slow_movement_count",
            "Slow Movements",
            "Slow Movement Count",
            "02_slow_movement_count_heatmap.png",
        ),
        (
            "slow_movement_pct",
            "Slow Movement Percentage",
            "Slow Movement Percentage (%)",
            "03_slow_movement_percentage_heatmap.png",
        ),
        (
            "DSH",
            "Degree of Speed Harmonization",
            "DSH",
            "04_DSH_heatmap.png",
        ),
    ]

    for (
        metric,
        title,
        colorbar_label,
        filename,
    ) in metrics:

        matrix = prepare_heatmap_data(
            five_min,
            link_order,
            metric,
        )

        create_heatmap(
            matrix,
            title,
            colorbar_label,
            FIGURE_DIR / filename,
        )

        print(
            f"Created: {filename}"
        )


def main():

    # Step 1: Create visualization output folder
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Step 2: Load hourly and 5-minute MOE results
    hourly, five_min = load_results()

    # Step 3: Select links using CV coverage only
    selected_links = (
        select_visualization_links(
            hourly
        )
    )

    # Step 4: Save the links used in the figures
    selected_links.to_csv(
        VISUALIZED_LINKS_FILE,
        index=False,
    )

    print("MOE VISUALIZATION")
    print("=" * 70)
    print(
        f"Links visualized: {len(selected_links):,}"
    )
    print(
        "Selection basis: highest waypoint count only"
    )
    print()

    # Step 5: Create four time-dependent heatmaps
    create_all_heatmaps(
        five_min,
        selected_links,
    )

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print(
        f"Output folder: {FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()