# -*- coding: utf-8 -*-
"""
Created on Friday March 6th 2025

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.cm as cm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mtick  # type: ignore
import matplotlib.colors as mcolors  # type: ignore


# Summarize metrics ----
def summarize_metrics(df_network_all: pd.DataFrame) -> list:
    """
    Summarize the complexity metrics over many multiplets.

    Parameters:
    -----------
    df_network_all : pd.DataFrame
        A DataFrame containing the estimated complex network metrics like
        transitivity, clustering, shortest path, maximum degree, mean degree,
        heterogeneity, diameter, and radius.
    """
    # Generate group variables
    df = df_network_all.copy()
    df["particles"] = df["video"].str[0]
    df["sex_ratio"] = df["video"].str[0:8]

    # Groups
    g1 = ["video", "permuted_id", "size"]
    g2 = ["sex_ratio", "permuted_id", "size"]
    vars_1, vars_2 = [], []

    # Complexity Network - Metrics
    metrics = [
        "transitivity", "avg_shortest_path", "mean_degree",
        "avg_clustering", "radius", "heterogeneity"
    ]

    for col in metrics:
        for i, g in enumerate([g1, g2], start=1):
            c = col + "_count_" + str(i)
            m = col + "_mean_" + str(i)
            s = col + "_std_" + str(i)
            df[c] = df.groupby(g)[col].transform("count")
            df[m] = df.groupby(g)[col].transform("mean")
            df[s] = df.groupby(g)[col].transform("std") / np.sqrt(df[c])
            if i == 1:
                vars_1.append(c)
                vars_1.append(m)
                vars_1.append(s)
            else:
                vars_2.append(c)
                vars_2.append(m)
                vars_2.append(s)

    g1 += vars_1
    g2 += vars_2

    k1 = df[g1].drop_duplicates()
    k1["label_key"] = k1["video"] + "_" + k1["permuted_id"].astype(str)

    k2 = df[g2].drop_duplicates()
    k2["label_key"] = k2["sex_ratio"] + "_" + k2["permuted_id"].astype(str)

    return k1.sort_values(g1), k2.sort_values(g2)


# Plot Complex network measures ----
def plot_gliding_complex_network(
    df_network_all,
    width: int = 30,
    height: int = 36,
    n_x_breaks: int = 20,
    n_y_breaks: int = 20,
    fancy_legend: bool = False,
    save_figure: bool = False,
    output_path: str = "../output_files",
    output_name: str = "plot_gliding"
):
    """
    Plot the complex network metrics over many IDs for one video.

    Parameters:
    -----------
    df_network_all : pd.DataFrame
        A DataFrame containing the estimated complex network metrics like
        transitivity, clustering, shortest path, maximum degree, mean degree,
        heterogeneity, diameter, and radius.
    width : int
        Width of final plot. Default value 30
    height : int
        Width of final plot. Default value 36
    n_x_breaks : int
        Number of divisions in x-axis. Default value 20
    n_y_breaks : int
        Number of divisions in y-axis. Default value 20
    fancy_legend : bool
        Fancy legend output (default value False)
    save_figure: bool
        Save plot flag (default value False)
    output_path : string
        Local path for outputs. Default value is "../output_files"
    output_name : string
        Name of the output. Default value is "plot_gliding"
    """
    legend_entries = {}
    dicc_colors = {"2": "plasma", "3": "cool", "4": "copper"}
    fig, axes = plt.subplots(2, 4, figsize=(width, height))
    for video in df_network_all["video"].unique():
        particles = video[0]
        mask_1 = df_network_all["video"] == video
        mt = df_network_all[mask_1]["permuted_id"].unique()
        map = cm.get_cmap(dicc_colors[particles], len(mt))
        colors = {key: mcolors.to_hex(map(i)) for i, key in enumerate(mt)}
        for m in mt:
            mask = mask_1 & (df_network_all["permuted_id"] == m)
            title = video + " - " + str(m)

            # Time series data
            df = df_network_all[mask]
            s = df["size"].values
            v1 = df["transitivity"].values
            v2 = df["avg_shortest_path"].values
            v3 = df["mean_degree"].values
            v4 = df["radius"].values
            v5 = df["avg_clustering"].values
            v6 = df["maximum_degree"].values
            v7 = df["heterogeneity"].values
            v8 = df["diameter"].values

            cols = [
                "Transitivity", "Average Shortest Path", "Mean degree",
                "Radius", "Average Clustering", "Maximum degree",
                "Heterogeneity", "Diameter"
            ]

            # Plot into axes
            for j, y in enumerate([v1, v2, v3, v4, v5, v6, v7, v8]):
                x = s
                xlabel = "Window size ($\\omega$)"
                ylabel = cols[j] + " ($N=" + particles + "$)"
                line = axes[j // 4][j % 4].plot(
                    x,
                    y,
                    label=title,
                    marker="o",
                    color=colors[m],
                    ls="",
                    ms=4
                )[0]
                legend_entries[title] = line

                # Axes labels
                axes[j // 4][j % 4].set_xlabel(xlabel, fontsize=14)
                axes[j // 4][j % 4].set_ylabel(ylabel, fontsize=14)

    # Global plot settings
    for i in range(2):
        for j in range(4):
            axes[i][j].tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                labelsize=11,
                length=12
            )
            axes[i][j].tick_params(
                which="minor",
                direction="in",
                top=True,
                right=True,
                labelsize=11,
                length=6
            )
            axes[i][j].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
            axes[i][j].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))  # noqa: 501
            axes[i][j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
            axes[i][j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))  # noqa: 501
            axes[i][j].tick_params(axis="x", labelrotation=90)

    fig.legend(
        legend_entries.values(),
        legend_entries.keys(),
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=12,
        frameon=True,
        fancybox=fancy_legend
    )
    plt.tight_layout(rect=[0, 0, 0.98, 1])  # reserve space for legend

    if save_figure:
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, f"{output_name}.png")
        fig.savefig(full_path, dpi=400, bbox_inches="tight")
        print(f"Figure saved to {full_path}")
    plt.close()

    return fig, axes


# Plot complexity measures (Summary) ----
def plot_complex_network_summary(
    df_network_all: pd.DataFrame,
    width: int = 24,
    height: int = 10,
    n_x_breaks: int = 20,
    n_y_breaks: int = 20,
    fancy_legend: bool = True,
    save_figures: bool = False,
    output_path: str = "../output_files",
    output_name: str = "plot_gliding_summary"
):
    """
    Plot the complex network metrics over many IDs.

    Parameters:
    -----------
    df_network_all : pd.DataFrame
        A DataFrame containing the estimated complex network metrics like
        transitivity, clustering, shortest path, maximum degree, mean degree,
        heterogeneity, diameter, and radius.
    width : int
        Width of final plot. Default value 24
    height : int
        Width of final plot. Default value 10
    n_x_breaks : int
        Number of divisions in x-axis. Default value 20
    n_y_breaks : int
        Number of divisions in y-axis. Default value 20
    fancy_legend : bool
        Fancy legend output (default value True)
    save_figures: bool
        Save plots flag (default value False)
    output_path : string
        Local path for outputs. Default value is "../output_files"
    output_name : string
        Name of the outputs. Default value is "plot_gliding_summary"
    """
    k1, k2 = summarize_metrics(df_network_all=df_network_all)
    dicc_g = {0: "o", 1: "v", 2: "s", 3: "D"}

    # Unique keys and combinations and color mapping
    m1 = k1["label_key"].unique()
    map_1 = cm.get_cmap("plasma", len(m1))
    label_color_1 = {key: mcolors.to_hex(map_1(i)) for i, key in enumerate(m1)}

    m2 = k2["label_key"].unique()
    map_2 = cm.get_cmap("plasma", len(m2))
    label_color_2 = {key: mcolors.to_hex(map_2(i)) for i, key in enumerate(m2)}

    # Complexity Network - Metrics
    complex_network_titles = [
        "Transitivity",
        # "Average Shortest Path",
        "Mean degree",
        "Average Clustering",
        "Radius"  # ,
        # "Heterogeneity"
    ]
    complex_network_metrics = [
        "transitivity",
        # "avg_shortest_path",
        "mean_degree",
        "avg_clustering",
        "radius"  # ,
        # "heterogeneity"
    ]

    # Figure 1 - Video
    cols = len(complex_network_metrics)
    fig_1, axes_1 = plt.subplots(3, cols, figsize=(width, height))
    legend_entries_1 = {}

    for group in sorted(k1["video"].unique()):
        particles = group[0]
        males = group[3]
        females = group[6]
        mask_1 = k1["video"] == group
        for m in k1[mask_1]["permuted_id"].unique():
            label_key = group + "_" + str(m)
            color = label_color_1[label_key]
            mask = mask_1 & (k1["permuted_id"] == m)
            title = group + " - " + str(m)
            label = int(particles) - 2 if int(particles) >= 2 else -1
            if label == -1:
                continue

            for j, col in enumerate(complex_network_metrics):
                m_mean = col + "_mean_1"
                m_std = col + "_std_1"
                df_aux = k1[mask]
                size = df_aux["size"].values
                ym = df_aux[m_mean].values
                ys = df_aux[m_std].values

                x = size
                xs = np.zeros(len(df_aux))
                xlabel = "Window size ($\\omega$)"

                # Plot error bars
                line = axes_1[label][j].errorbar(
                    x,
                    ym,
                    xerr=xs,
                    yerr=ys,
                    label=title,
                    capsize=5,
                    ls="--",
                    lw=0.7,
                    fmt=dicc_g[m],
                    color=color
                )[0]
                legend_entries_1[title] = line
                ylabel = complex_network_titles[j] + " ($N=" + particles + "$)"
                axes_1[label][j].set_xlabel(xlabel, fontsize=14)
                axes_1[label][j].set_ylabel(ylabel, fontsize=14)

    # Figure 2 - Sex ratio
    fig_2, axes_2 = plt.subplots(3, cols, figsize=(width, height))
    legend_entries_2 = {}

    for group in sorted(k2["sex_ratio"].unique()):
        particles = group[0]
        males = group[3]
        females = group[6]
        mask_1 = k2["sex_ratio"] == group
        for m in k2[mask_1]["permuted_id"].unique():
            label_key = group + "_" + str(m)
            color = label_color_2[label_key]
            mask = mask_1 & (k2["permuted_id"] == m)
            title = males + "M" + females + "F - " + str(m)
            label = int(particles) - 2 if int(particles) >= 2 else -1
            if label == -1:
                continue

            for j, col in enumerate(complex_network_metrics):
                m_mean = col + "_mean_2"
                m_std = col + "_std_2"
                df_aux = k2[mask]
                size = df_aux["size"].values
                ym = df_aux[m_mean].values
                ys = df_aux[m_std].values

                x = size
                xs = np.zeros(len(df_aux))
                xlabel = "Window size ($\\omega$)"

                # Plot error bars
                line = axes_2[label][j].errorbar(
                    x,
                    ym,
                    xerr=xs,
                    yerr=ys,
                    label=title,
                    capsize=5,
                    ls="--",
                    lw=0.7,
                    fmt=dicc_g[m],
                    color=color
                )[0]
                legend_entries_2[title] = line
                ylabel = complex_network_titles[j] + " ($N=" + particles + "$)"
                axes_2[label][j].set_xlabel(xlabel, fontsize=14)
                axes_2[label][j].set_ylabel(ylabel, fontsize=14)

    # Styling
    for i in range(3):
        for j in range(cols):
            for ax in [axes_1, axes_2]:
                ax[i][j].tick_params(
                    which="major",
                    direction="in",
                    top=True,
                    right=True,
                    labelsize=11,
                    length=12
                )
                ax[i][j].tick_params(
                    which="minor",
                    direction="in",
                    top=True,
                    right=True,
                    labelsize=11,
                    length=6
                )
                ax[i][j].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
                ax[i][j].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))  # noqa: 501
                ax[i][j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax[i][j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))  # noqa: 501
                ax[i][j].tick_params(axis="x", labelrotation=90)

    fig_1.legend(
        legend_entries_1.values(),
        legend_entries_1.keys(),
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=12,
        frameon=False,
        fancybox=fancy_legend
    )
    fig_1.tight_layout(rect=[0, 0, 0.98, 1])  # reserve space for legend

    fig_2.legend(
        legend_entries_2.values(),
        legend_entries_2.keys(),
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=12,
        frameon=False,
        fancybox=fancy_legend
    )
    fig_2.tight_layout(rect=[0, 0, 0.98, 1])  # reserve space for legend

    if save_figures:
        os.makedirs(output_path, exist_ok=True)
        full_path_1 = os.path.join(output_path, f"{output_name}_video.png")
        full_path_2 = os.path.join(output_path, f"{output_name}_sexratio.png")
        fig_1.savefig(full_path_1, dpi=400, bbox_inches="tight")
        fig_2.savefig(full_path_2, dpi=400, bbox_inches="tight")
        print(f"Figure saved to {full_path_1} and {full_path_2}")
    plt.close()
    plt.close()

    return k1, k2, fig_1, fig_2, axes_1, axes_2
