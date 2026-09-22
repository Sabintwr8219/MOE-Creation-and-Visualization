from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


# Paths and settings
RESULTS_DIR = Path(__file__).resolve().parent / "Results"

CORRIDOR_FILE = RESULTS_DIR / "corridor_links.csv"
MOE_FILE = RESULTS_DIR / "corridor_moe_5min.csv"

FIGURE_DIR = (
    RESULTS_DIR
    / "Corridor Time-Space Visualizations"
)

TIME_BIN_MINUTES = 5
BINS_PER_HOUR = 60 // TIME_BIN_MINUTES

FIGURE_DPI = 300
ROBUST_PERCENTILE = 95

FIGURE_SIZE = (12, 8)


# Load corridor and MOE outputs
def load_results():

    corridor = pd.read_csv(
        CORRIDOR_FILE,
        dtype={
            "link_key": "string",
        },
    )

    moe = pd.read_csv(
        MOE_FILE,
        dtype={
            "link_key": "string",
        },
        parse_dates=[
            "time_bin"
        ],
    )

    corridor = corridor.sort_values(
        "corridor_order"
    ).reset_index(drop=True)

    return corridor, moe


# Build a complete link-by-time matrix
def prepare_matrix(
    corridor,
    moe,
    metric,
):

    link_order = (
        corridor["link_key"]
        .tolist()
    )

    start_time = (
        moe["time_bin"]
        .min()
        .floor("h")
    )

    time_bins = pd.date_range(
        start=start_time,
        periods=BINS_PER_HOUR,
        freq=f"{TIME_BIN_MINUTES}min",
    )

    matrix = (
        moe.pivot(
            index="link_key",
            columns="time_bin",
            values=metric,
        )
        .reindex(
            index=link_order,
            columns=time_bins,
        )
    )

    return matrix, time_bins


# Calculate a robust upper display limit
def calculate_display_max(
    values,
    fixed_max=None,
):

    if fixed_max is not None:
        return fixed_max

    finite = values[
        np.isfinite(values)
    ]

    finite = finite[
        finite >= 0
    ]

    if len(finite) == 0:
        return 1.0

    display_max = np.nanpercentile(
        finite,
        ROBUST_PERCENTILE,
    )

    if display_max <= 0:
        return 1.0

    return display_max


# Draw one clean distance-based time-space diagram
def create_timespace_plot(
    corridor,
    moe,
    metric,
    title,
    colorbar_label,
    filename,
    fixed_max=None,
):

    matrix, time_bins = prepare_matrix(
        corridor,
        moe,
        metric,
    )

    values = matrix.to_numpy(
        dtype=float
    )

    masked_values = np.ma.masked_invalid(
        values
    )

    # True physical link boundaries
    y_edges = np.concatenate(
        [
            [
                corridor[
                    "distance_start_ft"
                ].iloc[0]
            ],
            corridor[
                "distance_end_ft"
            ].to_numpy(),
        ]
    )

    # Five-minute time boundaries
    x_edges = (
        np.arange(
            len(time_bins) + 1
        )
        * TIME_BIN_MINUTES
    )

    display_max = calculate_display_max(
        values,
        fixed_max=fixed_max,
    )

    cmap = (
        plt.get_cmap(
            "RdYlGn_r"
        )
        .copy()
    )

    # Missing data
    cmap.set_bad(
        "lightgray"
    )

    norm = Normalize(
        vmin=0,
        vmax=display_max,
        clip=True,
    )

    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE
    )

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        masked_values,
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )

    # Very subtle physical link boundaries
    for boundary in y_edges[1:-1]:

        ax.axhline(
            boundary,
            linewidth=0.25,
            color="white",
            alpha=0.18,
        )

    # Time labels
    x_centers = (
        x_edges[:-1]
        + TIME_BIN_MINUTES / 2
    )

    ax.set_xticks(
        x_centers
    )

    ax.set_xticklabels(
        [
            timestamp.strftime(
                "%H:%M"
            )
            for timestamp in time_bins
        ],
        rotation=45,
        ha="right",
        fontsize=9,
    )

    # Keep distance axis readable
    ax.yaxis.set_major_locator(
        MaxNLocator(
            nbins=6
        )
    )

    ax.tick_params(
        axis="y",
        labelsize=9,
    )

    ax.set_xlim(
        x_edges[0],
        x_edges[-1],
    )

    ax.set_ylim(
        y_edges[0],
        y_edges[-1],
    )

    ax.set_xlabel(
        "Time",
        fontsize=11,
    )

    ax.set_ylabel(
        "Cumulative Distance Along Corridor (ft)",
        fontsize=11,
    )

    ax.set_title(
        title,
        fontsize=14,
        pad=14,
    )

    # Clean frame
    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    # Reserve a clean area for the colorbar
    fig.subplots_adjust(
        left=0.10,
        right=0.86,
        bottom=0.14,
        top=0.91,
    )

    colorbar_axis = fig.add_axes(
        [
            0.885,
            0.18,
            0.018,
            0.64,
        ]
    )

    colorbar = fig.colorbar(
        mesh,
        cax=colorbar_axis,
    )

    colorbar.set_label(
        colorbar_label,
        fontsize=10,
        labelpad=10,
    )

    colorbar.ax.tick_params(
        labelsize=9,
    )

    fig.savefig(
        FIGURE_DIR / filename,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# Create all MOE time-space diagrams
def create_all_figures(
    corridor,
    moe,
):

    figures = [
        (
            "avg_travel_time_s",
            "Average Travel Time",
            "Travel Time (s)",
            "01_average_travel_time_timespace.png",
            None,
        ),
        (
            "slow_movement_count",
            "Slow Movement Count",
            "Slow Movements",
            "02_slow_movement_count_timespace.png",
            None,
        ),
        (
            "slow_movement_pct",
            "Slow Movement Percentage",
            "Slow Movement (%)",
            "03_slow_movement_percentage_timespace.png",
            100,
        ),
        (
            "DSH",
            "Degree of Speed Harmonization",
            "DSH",
            "04_DSH_timespace.png",
            None,
        ),
        (
            "speed_reduction_pct",
            "Speed Reduction Percentage",
            "Speed Reduction (%)",
            "05_speed_reduction_percentage_timespace.png",
            100,
        ),
    ]

    for (
        metric,
        title,
        colorbar_label,
        filename,
        fixed_max,
    ) in figures:

        create_timespace_plot(
            corridor,
            moe,
            metric,
            title,
            colorbar_label,
            filename,
            fixed_max=fixed_max,
        )

        print(
            f"Created: {filename}"
        )


def main():

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Step 1: Load corridor and MOE data
    corridor, moe = load_results()

    print(
        "CREATING CLEAN TIME-SPACE FIGURES"
    )
    print("=" * 70)

    print(
        f"Links:           {len(corridor):,}"
    )

    print(
        f"Corridor length: "
        f"{corridor['link_length_ft'].sum():,.1f} ft"
    )

    # Step 2: Create all figures
    create_all_figures(
        corridor,
        moe,
    )

    print()
    print("=" * 70)
    print(
        "VISUALIZATION COMPLETE"
    )

    print(
        f"Output: {FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()