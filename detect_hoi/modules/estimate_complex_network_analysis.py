# -*- coding: utf-8 -*-
"""
Created on Friday March 6th 2025

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import pandas as pd  # type: ignore
import networkx as nx  # type: ignore
import misc_functions as mf
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mtick  # type: ignore

from ts2vg import NaturalVG  # type: ignore
from matplotlib import rcParams  # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)


# Estimate Visibility Graph ----
def estimate_vg(
    df: pd.DataFrame,
    filter_step: int = None,
    width: float = 12,
    height: float = 28,
    n_x_breaks: int = 20,
    n_y_breaks: int = 20,
    fancy_legend: bool = False,
    usetex: bool = False,
    t_range: list = [0, 100]
) -> list:
    """Estimate visibility graph and its properties for the time series of
    distances between two individuals. The data has the following columns:
        - id: Particle ID
        - permuted_id: Particle ID after smoothing process
        - time: Times (Frames)
        - position_x: Centroid position in X-axis
        - position_y: Centroid position in Y-axis
        - weighted_x: Centroid position in X-axis weighted with intensity image
        - weighted_y: Centroid position in Y-axis weighted with intensity image
        - darkest_v: Intensity of darkest pixel in the local region (particle)
        - darkest_x: Position of darkest pixel in the local region (particle)
        - darkest_y: Position of darkest pixel in the local region (particle)
        - lightest_v: Intensity of lightest pixel in the local region(particle)
        - lightest_x: Position of lightest pixel in the local region (particle)
        - lightest_y: Position of lightest pixel in the local region (particle)
        - coords_x: X values of the region's boundary
        - coords_y: Y values of the region's boundary
        - orientation: Orientation respect rows
        - corrected_orientation: Orientation after smoothing process
        - area: Region area i.e. number of pixels of the region scaled by
        pixel-area
        - axis_major: The length of the major axis of the ellipse that has the
        same normalized second central moments as the region
        - axis_minor: The length of the minor axis of the ellipse that has the
        same normalized second central moments as the region
        - eccentricity: Eccentricity of the ellipse that has the same
        second-moments as the region. The eccentricity is the ratio of the
        focal distance (distance between focal points) over the major axis
        length. The value is in the interval [0, 1). When it is 0, the ellipse
        becomes a circle

    Args:
    ---------------------------------------------------------------------------
    df : pd.DataFrame
        Dataframe with the information of tracked regions
    filter_step : int
        Number of steps between consecutive times, i.e., the number of skipped
        steps
    width : float
        Width of final plot (default value 12)
    height : float
        Height of final plot (default value 28)
    n_x_breaks : int
        Number of divisions in x-axis (default value 10)
    n_y_breaks : int
        Number of divisions in y-axis (default value 10)
    fancy_legend : bool
        Fancy legend output (default value False)
    usetex : bool
        Use LaTeX for renderized plots (default value False)
    t_range : list
        Lower ([0]) and upper ([1]) threshold for the filtration of time series
        data points range

    Returns:
    ---------------------------------------------------------------------------
    df_all : pd.DataFrame
        Dataframe with the summary of global complex network measures
    df_all_nodes : pd.DataFrame
        Dataframe with the summary of complex network measures over each node
    fig : fig
        Figure with the time series, visibility graph, and degree distribution
    ax : ax
        Axes with the time series, visibility graph, and degree distribution
    """

    # Construct time series of distances between particles
    mask = (df["time"].between(t_range[0], t_range[1]))
    df_distances = mf.estimate_distances(df=df[mask], filter_step=filter_step)
    pairs = df_distances["id_pair"].unique()

    # Initialize Plot Graph
    rcParams.update({
        "font.family": "serif",
        "text.usetex": usetex,
        "pgf.rcfonts": False,
        "text.latex.preamble": r"\usepackage{amsfonts}"
    })

    fig, ax = plt.subplots(len(pairs), 4, squeeze=False)
    fig.set_size_inches(w=width, h=height)

    graph_plot_options = {
        "with_labels": False,
        "node_size": 2.5,
        "node_color": [(0, 0, 0, 1)],
        "edge_color": [(0, 0, 0, 0.15)]
    }

    # Network analysis over pairs
    df_all = []
    df_all_nodes = []
    for j, id_pair in enumerate(pairs):
        mask = (df_distances["id_pair"] == id_pair)
        times = df_distances[mask]["time"].values
        distances = df_distances[mask]["distance"].values
        graph = NaturalVG(directed=None).build(distances, times)
        nxg = graph.as_networkx()  # From ts2vg
        df, df_local = mf.summarize_complex_network(nxg=nxg)

        # Add local information
        df["id_pair"] = id_pair
        df_local["id_pair"] = id_pair

        df["t_range"] = str(t_range[0]) + "_" + str(t_range[1])
        df_local["t_range"] = str(t_range[0]) + "_" + str(t_range[1])

        df_all.append(df)
        df_all_nodes.append(df_local)

        # Add local plot
        label = "$d_{{{}}}(t)$".format(id_pair)
        degrees = df_local["degree"].values

        nx.draw_networkx(
            nxg,
            ax=ax[j, 1],
            pos=graph.node_positions(),
            **graph_plot_options
        )
        nx.draw_networkx(
            nxg, ax=ax[j, 2],
            pos=nx.kamada_kawai_layout(nxg),
            **graph_plot_options
        )

        ax[j, 0].plot(times, distances, label=label, ls="solid", lw=3)
        ax[j, 1].plot(times, distances, label=label, ls="solid", lw=3)
        ax[j, 3].hist(
            degrees,
            label=label,
            alpha=0.19,
            facecolor="blue",
            edgecolor="darkblue",
            density=False,
            histtype="stepfilled",
            cumulative=False
        )

        # Other plot settings
        ax[j, 0].set_ylabel("$d(t)$")
        ax[j, 0].set_xlabel("Time (t)", fontsize=16)
        ax[j, 0].set_title("Distance time series")
        ax[j, 1].set_ylabel("$d(t)$")
        ax[j, 1].set_xlabel("Time (t)", fontsize=16)
        ax[j, 1].set_title("Distance and Visibility Graph")
        ax[j, 2].set_title("Visibility Graph (VG)")
        ax[j, 3].set_title("VG - Degrees distribution")

        for k in [0, 1, 3]:
            ax[j, k].legend(fancybox=fancy_legend, fontsize=14)
            ax[j, k].tick_params(which="major", direction="in", top=True, right=True, labelsize=11, length=12)  # noqa: 501
            ax[j, k].tick_params(which="minor", direction="in", top=True, right=True, labelsize=11, length=6)  # noqa: 501
            ax[j, k].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
            ax[j, k].xaxis.set_minor_locator(mtick.MaxNLocator(4 * n_x_breaks))
            ax[j, k].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
            ax[j, k].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
            ax[j, k].tick_params(axis="x", labelrotation=90)

    plt.tight_layout()
    plt.close()

    # Final merge and relocation of new columns
    df_all = pd.concat(df_all, ignore_index=True)
    df_all_nodes = pd.concat(df_all_nodes, ignore_index=True)

    df_all.insert(0, "id_pair", df_all.pop("id_pair"))
    df_all_nodes.insert(0, "id_pair", df_all_nodes.pop("id_pair"))
    df_all.insert(1, "t_range", df_all.pop("t_range"))
    df_all_nodes.insert(1, "t_range", df_all_nodes.pop("t_range"))

    return df_all, df_all_nodes, fig, ax
