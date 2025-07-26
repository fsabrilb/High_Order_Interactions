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
def summarize_metrics(df_oinfo: pd.DataFrame) -> list:
    """
    Summarize the O-information and HoI metrics over many multiplets.

    Parameters:
    -----------
    df_oinfo : pd.DataFrame
        A DataFrame containing the estimated O-information for different
        multiplets. The DataFrame includes the following columns:
            - "video": Video name.
            - "t_range": The timestamp at which the O-information was
            estimated.
            - "size": The window size at which the O-information was estimated.
            - "multiplet": The identifier for each possible multiplet.
            - "oinfo_distance": The estimated O-information over distances from
            center.
            - "oinfo_orientation": The estimated O-information over angles.
            - "sinfo_distance": The estimated exogenous information over
            distances from center.
            - "sinfo_orientation": The estimated exogenous information over
            angles.
    """
    # Generate group variables
    df = df_oinfo.copy()
    df["particles"] = df["video"].str[0]
    df["sex_ratio"] = df["video"].str[0:8]

    # Groups
    g1 = ["video", "multiplet", "size"]
    g2 = ["sex_ratio", "multiplet", "size"]
    vars_1, vars_2 = [], []

    # HoI - Metrics
    metrics = [
        "oinfo_distance", "oinfo_orientation",
        "sinfo_distance", "sinfo_orientation"
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
    k1["label_key"] = k1["video"] + "_" + k1["multiplet"].astype(str)

    k2 = df[g2].drop_duplicates()
    k2["label_key"] = k2["sex_ratio"] + "_" + k2["multiplet"].astype(str)

    return k1.sort_values(g1), k2.sort_values(g2)


# Plot high-order interactions (HoI) measures ----
def plot_gliding_oinfo(
    df_oinfo,
    width: int = 24,
    height: int = 27,
    n_x_breaks: int = 20,
    n_y_breaks: int = 20,
    fancy_legend: bool = False,
    save_figure: bool = False,
    output_path: str = "../output_files",
    output_name: str = "plot_gliding"
):
    """
    Plot the O-information and HoI metrics over many multiplets for one video.

    Parameters:
    -----------
    df_oinfo : pd.DataFrame
        A DataFrame containing the estimated O-information for different
        multiplets. The DataFrame includes the following columns:
            - "video": Video name.
            - "t_range": The timestamp at which the O-information was
            estimated.
            - "size": The window size at which the O-information was estimated.
            - "multiplet": The identifier for each possible multiplet.
            - "oinfo_distance": The estimated O-information over distances from
            center.
            - "oinfo_orientation": The estimated O-information over angles.
            - "sinfo_distance": The estimated exogenous information over
            distances from center.
            - "sinfo_orientation": The estimated exogenous information over
            angles.
    width : int
        Width of final plot. Default value 24
    height : int
        Width of final plot. Default value 27
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
    legend_labels = []
    legend_handles = []
    dicc_colors = {"2": "plasma", "3": "cool", "4": "copper"}
    fig, axes = plt.subplots(1, 4, figsize=(width, height))
    for video in df_oinfo["video"].unique():
        particles = video[0]
        mask_1 = df_oinfo["video"] == video
        mt = df_oinfo[mask_1]["multiplet"].unique()
        map = cm.get_cmap(dicc_colors[particles], len(mt))
        colors = {key: mcolors.to_hex(map(i)) for i, key in enumerate(mt)}
        for m in mt:
            mask = mask_1 & (df_oinfo["multiplet"] == m)
            title = video + " - " + str(m)
            if int(particles) < 3:
                continue  # skip unrecognized multiplets

            # Time series data
            df = df_oinfo[mask]
            s = df["size"].values
            od = df["oinfo_distance"].values
            oa = df["oinfo_orientation"].values
            sd = df["sinfo_distance"].values
            sa = df["sinfo_orientation"].values

            # Plot into axes
            for j, y in enumerate([od, oa, sd, sa]):
                axes[j].hlines(
                    0,
                    xmin=np.min(s),
                    xmax=np.max(s),
                    color="black",
                    ls="--",
                    lw=0.8
                )
                axes[j].plot(
                    s,
                    y,
                    label=title,
                    marker="o",
                    color=colors[m],
                    ls="",
                    ms=4
                )
                legend_handles.append(axes[j].lines[-1])
                legend_labels.append(title)

                # Axes labels
                axes[j].set_xlabel("Window size ($\\omega$)", fontsize=14)

            axes[0].set_ylabel(r"$\Omega_{" + particles + r"}^{D}(\omega)$", fontsize=14)  # noqa: 501
            axes[1].set_ylabel(r"$\Omega_{" + particles + r"}^{\theta}(\omega)$", fontsize=14)  # noqa: 501
            axes[2].set_ylabel(r"$S_{" + particles + r"}^{D}(\omega)$", fontsize=14)  # noqa: 501
            axes[3].set_ylabel(r"$S_{" + particles + r"}^{\theta}(\omega)$", fontsize=14)  # noqa: 501

    # Global plot settings
    for j in range(4):
        axes[j].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=11,
            length=12
        )
        axes[j].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            labelsize=11,
            length=6
        )
        axes[j].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
        axes[j].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
        axes[j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        axes[j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
        axes[j].tick_params(axis="x", labelrotation=90)

    fig.legend(
        list(set(legend_handles)),
        list(set(legend_labels)),
        loc="center left",
        bbox_to_anchor=(1.001, 0.5),
        fontsize=12,
        frameon=False,
        fancybox=fancy_legend
    )
    plt.tight_layout(rect=[0, 0, 0.99, 1])  # reserve space for legend

    if save_figure:
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, f"{output_name}.png")
        fig.savefig(full_path, dpi=400, bbox_inches="tight")
        print(f"Figure saved to {full_path}")
    plt.close()

    return fig, axes


# Plot high-order interactions (HoI) measures (Summary) ----
def plot_hoi_metrics_summary(
    df_oinfo: pd.DataFrame,
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
    Plot the O-information and HoI metrics over many multiplets.

    Parameters:
    -----------
    df_oinfo : pd.DataFrame
        A DataFrame containing the estimated O-information for different
        multiplets. The DataFrame includes the following columns:
            - "video": Video name.
            - "t_range": The timestamp at which the O-information was
            estimated.
            - "size": The window size at which the O-information was estimated.
            - "multiplet": The identifier for each possible multiplet.
            - "oinfo_distance": The estimated O-information over distances from
            center.
            - "oinfo_orientation": The estimated O-information over angles.
            - "sinfo_distance": The estimated exogenous information over
            distances from center.
            - "sinfo_orientation": The estimated exogenous information over
            angles.
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
    k1, k2 = summarize_metrics(df_oinfo=df_oinfo)

    # Unique keys and combinations and color mapping
    m1 = k1["label_key"].unique()
    map_1 = cm.get_cmap("plasma", len(m1))
    label_color_1 = {key: mcolors.to_hex(map_1(i)) for i, key in enumerate(m1)}

    m2 = k2["label_key"].unique()
    map_2 = cm.get_cmap("plasma", len(m2))
    label_color_2 = {key: mcolors.to_hex(map_2(i)) for i, key in enumerate(m2)}

    # Figure 1 - Video
    legend_labels_1 = []
    legend_handles_1 = []
    fig_1, axes_1 = plt.subplots(2, 4, figsize=(width, height))

    # HoI - Metrics
    hoi_metrics = [
        "oinfo_distance", "oinfo_orientation",
        "sinfo_distance", "sinfo_orientation"
    ]

    for group in sorted(k1["video"].unique()):
        particles = group[0]
        males = group[3]
        females = group[6]
        mask_1 = k1["video"] == group
        for m in k1[mask_1]["multiplet"].unique():
            label_key = group + "_" + str(m)
            color = label_color_1[label_key]
            mask = mask_1 & (k1["multiplet"] == m)
            title = group + " - " + str(m)
            label = int(particles) - 3 if int(particles) >= 3 else -1
            if label == -1:
                continue

            for j, col in enumerate(hoi_metrics):
                m_mean = col + "_mean_1"
                m_std = col + "_std_1"
                df_aux = k1[mask]
                size = df_aux["size"].values
                ym = df_aux[m_mean].values
                ys = df_aux[m_std].values

                # Add reference line
                axes_1[label][j].hlines(
                    0,
                    xmin=np.min(size),
                    xmax=np.max(size),
                    color="black",
                    ls="--",
                    lw=0.8
                )

                # Plot error bars
                axes_1[label][j].errorbar(
                    size,
                    ym,
                    yerr=ys,
                    label=title,
                    capsize=5,
                    ls="--",
                    lw=0.7,
                    fmt="o",
                    color=color
                )
                legend_handles_1.append(axes_1[label][j].lines[-1])
                legend_labels_1.append(title)

            # Axis labels
            for j in range(4):
                axes_1[label][j].set_xlabel(
                    "Window size ($\\omega$)",
                    fontsize=14
                )
            axes_1[label][0].set_ylabel(r"$\Omega_{" + particles + r"}^{D}(\omega)$", fontsize=14)  # noqa: 501
            axes_1[label][1].set_ylabel(r"$\Omega_{" + particles + r"}^{\theta}(\omega)$", fontsize=14)  # noqa: 501
            axes_1[label][2].set_ylabel(r"$S_{" + particles + r"}^{D}(\omega)$", fontsize=14)  # noqa: 501
            axes_1[label][3].set_ylabel(r"$S_{" + particles + r"}^{\theta}(\omega)$", fontsize=14)  # noqa: 501

    # Figure 2 - Sex ratio
    legend_labels_2 = []
    legend_handles_2 = []
    fig_2, axes_2 = plt.subplots(2, 4, figsize=(width, height))

    for group in sorted(k2["sex_ratio"].unique()):
        particles = group[0]
        males = group[3]
        females = group[6]
        mask_1 = k2["sex_ratio"] == group
        for m in k2[mask_1]["multiplet"].unique():
            label_key = group + "_" + str(m)
            color = label_color_2[label_key]
            mask = mask_1 & (k2["multiplet"] == m)
            title = males + "M" + females + "F - " + str(m)
            label = int(particles) - 3 if int(particles) >= 3 else -1
            if label == -1:
                continue

            for j, col in enumerate(hoi_metrics):
                m_mean = col + "_mean_2"
                m_std = col + "_std_2"
                df_aux = k2[mask]
                size = df_aux["size"].values
                ym = df_aux[m_mean].values
                ys = df_aux[m_std].values

                # Add reference line
                axes_2[label][j].hlines(
                    0,
                    xmin=np.min(size),
                    xmax=np.max(size),
                    color="black",
                    ls="--",
                    lw=0.8
                )

                # Plot error bars
                axes_2[label][j].errorbar(
                    size,
                    ym,
                    yerr=ys,
                    label=title,
                    capsize=5,
                    ls="--",
                    lw=0.7,
                    fmt="o",
                    color=color
                )

                legend_handles_2.append(axes_2[label][j].lines[-1])
                legend_labels_2.append(title)

            # Axis labels
            for j in range(4):
                axes_2[label][j].set_xlabel(
                    "Window size ($\\omega$)",
                    fontsize=14
                )
            axes_2[label][0].set_ylabel(r"$\Omega_{" + particles + r"}^{D}(\omega)$", fontsize=14)  # noqa: 501
            axes_2[label][1].set_ylabel(r"$\Omega_{" + particles + r"}^{\theta}(\omega)$", fontsize=14)  # noqa: 501
            axes_2[label][2].set_ylabel(r"$S_{" + particles + r"}^{D}(\omega)$", fontsize=14)  # noqa: 501
            axes_2[label][3].set_ylabel(r"$S_{" + particles + r"}^{\theta}(\omega)$", fontsize=14)  # noqa: 501

    # Styling
    for i in range(2):
        for j in range(4):
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
        list(set(legend_handles_1)),
        list(set(legend_labels_1)),
        loc="center left",
        bbox_to_anchor=(1.001, 0.5),
        fontsize=12,
        frameon=False,
        fancybox=fancy_legend
    )
    fig_1.tight_layout(rect=[0, 0, 0.99, 1])  # reserve space for legend

    fig_2.legend(
        list(set(legend_handles_2)),
        list(set(legend_labels_2)),
        loc="center left",
        bbox_to_anchor=(1.001, 0.5),
        fontsize=12,
        frameon=False,
        fancybox=fancy_legend
    )
    fig_2.tight_layout(rect=[0, 0, 0.99, 1])  # reserve space for legend

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
