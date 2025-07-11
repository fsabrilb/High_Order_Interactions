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
            - "dO_info_distance": The estimated dynamic O-information over
            distances from center.
            - "dO_info_orientation": The estimated dynamic O-information over
            angles.
    width : int
        Width of final plot. Default value 10
    height : int
        Width of final plot. Default value 10
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
    dicc_multiplet = {
        "012": 1,
        "013": 2,
        "023": 3,
        "123": 4,
        "0123": 5
    }

    legend_labels = []
    legend_handles = []
    fig, axes = plt.subplots(6, 4, figsize=(width, height))
    for video in df_oinfo["video"].unique():
        particles = video[0]
        males = video[3]
        females = video[6]

        mask_1 = df_oinfo["video"] == video
        for m in df_oinfo[mask_1]["multiplet"].unique():
            mask = mask_1 & (df_oinfo["multiplet"] == m)
            title = males + "M" + females + "F - " + str(m)
            label = 0 if int(particles) == 3 else dicc_multiplet.get(m, -1)
            if label == -1:
                continue  # skip unrecognized multiplets

            # Time series data
            df = df_oinfo[mask]
            s = df["size"].values
            od = df["oinfo_distance"].values
            oa = df["oinfo_orientation"].values
            dod = df["dO_info_distance"].values
            doa = df["dO_info_orientation"].values

            # Plot into axes
            for j, y in enumerate([od, oa, dod, doa]):
                axes[label][j].hlines(
                    0,
                    xmin=np.min(s),
                    xmax=np.max(s),
                    color="black",
                    ls="--",
                    lw=0.8
                )
                axes[label][j].plot(
                    s,
                    y,
                    label=title,
                    marker="o",
                    ls="",
                    ms=4
                )
                legend_handles.append(axes[label][j].lines[-1])
                legend_labels.append(title)

                # Axes labels
                axes[label][j].set_xlabel(
                    "Window size ($\\omega$)",
                    fontsize=14
                )

            axes[label][0].set_ylabel(r"$\Omega_{" + particles + r"}^{D}(t)$", fontsize=14)  # noqa: 501
            axes[label][1].set_ylabel(r"$\Omega_{" + particles + r"}^{\theta}(t)$", fontsize=14)  # noqa: 501
            axes[label][2].set_ylabel(r"$d\Omega_{" + particles + r"}^{D}(t)$", fontsize=14)  # noqa: 501
            axes[label][3].set_ylabel(r"$d\Omega_{" + particles + r"}^{\theta}(t)$", fontsize=14)  # noqa: 501

    # Global plot settings
    for i in range(6):
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
            axes[i][j].xaxis.set_minor_locator(mtick.MaxNLocator(4 * n_x_breaks))  # noqa: 501
            axes[i][j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
            axes[i][j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))  # noqa: 501
            axes[i][j].tick_params(axis="x", labelrotation=90)

    fig.legend(
        legend_handles,
        legend_labels,
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
        fig.savefig(full_path, dpi=300)
        print(f"Figure saved to {full_path}")
    else:
        plt.show()
    return axes


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
            - "dO_info_distance": The estimated dynamic O-information over
            distances from center.
            - "dO_info_orientation": The estimated dynamic O-information over
            angles.
    width : int
        Width of final plot. Default value 10
    height : int
        Width of final plot. Default value 10
    n_x_breaks : int
        Number of divisions in x-axis. Default value 20
    n_y_breaks : int
        Number of divisions in y-axis. Default value 20
    fancy_legend : bool
        Fancy legend output (default value False)
    save_figures: bool
        Save plots flag (default value False)
    output_path : string
        Local path for outputs. Default value is "../output_files"
    output_name : string
        Name of the outputs. Default value is "plot_gliding_summary"
    """

    df = df_oinfo.copy()
    df["particles"] = df["video"].str[0]
    df["sex_ratio"] = df["video"].str[0:8]

    hoi_metrics = [
        "oinfo_distance", "oinfo_orientation",
        "dO_info_distance", "dO_info_orientation"
    ]

    g1 = ["video", "multiplet", "size"]
    g2 = ["sex_ratio", "multiplet", "size"]
    for col in hoi_metrics:
        for i, g in enumerate([g1, g2], start=1):
            c = col + "_count_" + str(i)
            m = col + "_mean_" + str(i)
            s = col + "_std_" + str(i)
            df[c] = df.groupby(g)[col].transform("count")
            df[m] = df.groupby(g)[col].transform("mean")
            df[s] = df.groupby(g)[col].transform("std") / np.sqrt(df[c])

    # Unique keys and combinations and color mapping
    k1 = df[["video", "multiplet"]].drop_duplicates()
    k1["label_key"] = k1["video"] + "_" + k1["multiplet"].astype(str)
    map_1 = cm.get_cmap("plasma", len(k1))
    label_color_1 = {
        key: mcolors.to_hex(map_1(i)) for i, key in enumerate(k1["label_key"])
    }

    k2 = df[["sex_ratio", "multiplet"]].drop_duplicates()
    k2["label_key"] = k2["sex_ratio"] + "_" + k2["multiplet"].astype(str)
    map_2 = cm.get_cmap("plasma", len(k2))
    label_color_2 = {
        key: mcolors.to_hex(map_2(i)) for i, key in enumerate(k2["label_key"])
    }

    # Figure 1 - Video
    legend_labels_1 = []
    legend_handles_1 = []
    fig_1, axes_1 = plt.subplots(2, 4, figsize=(width, height))
    used_labels = [[set() for _ in range(4)] for _ in range(2)]  # Avoid duplic

    for group in df["video"].unique():
        particles = group[0]
        males = group[3]
        females = group[6]
        mask_1 = df["video"] == group
        for m in df[mask_1]["multiplet"].unique():
            label_key = group + "_" + str(m)
            color = label_color_1[label_key]
            mask = mask_1 & (df["multiplet"] == m)
            title = group + " - " + str(m)
            label = int(particles) - 3 if int(particles) >= 3 else -1
            if label == -1:
                continue

            for j, col in enumerate(hoi_metrics):
                m_mean = col + "_mean_1"
                m_std = col + "_std_1"
                df_aux = df[mask].drop_duplicates(subset=g1)
                size = df_aux["size"].values
                ym = df_aux[m_mean].values
                ys = df_aux[m_std].values
                show_label = title not in used_labels[label][j]

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
                    label=title if show_label else None,
                    capsize=5,
                    ls="--",
                    lw=0.7,
                    fmt="o",
                    color=color
                )
                legend_handles_1.append(axes_1[label][j].lines[-1])
                legend_labels_1.append(title)

                if show_label:
                    used_labels[label][j].add(title)

            # Axis labels
            for j in range(4):
                axes_1[label][j].set_xlabel(
                    "Window size ($\\omega$)",
                    fontsize=14
                )
            axes_1[label][0].set_ylabel(r"$\Omega_{" + particles + r"}^{D}(t)$", fontsize=14)  # noqa: 501
            axes_1[label][1].set_ylabel(r"$\Omega_{" + particles + r"}^{\theta}(t)$", fontsize=14)  # noqa: 501
            axes_1[label][2].set_ylabel(r"$d\Omega_{" + particles + r"}^{D}(t)$", fontsize=14)  # noqa: 501
            axes_1[label][3].set_ylabel(r"$d\Omega_{" + particles + r"}^{\theta}(t)$", fontsize=14)  # noqa: 501

    # Figure 2 - Sex ratio
    legend_labels_2 = []
    legend_handles_2 = []
    fig_2, axes_2 = plt.subplots(2, 4, figsize=(width, height))
    used_labels = [[set() for _ in range(4)] for _ in range(2)]  # Avoid duplic
    for group in df["sex_ratio"].unique():
        particles = group[0]
        males = group[3]
        females = group[6]
        mask_1 = df["sex_ratio"] == group
        for m in df[mask_1]["multiplet"].unique():
            label_key = group + "_" + str(m)
            color = label_color_2[label_key]
            mask = mask_1 & (df["multiplet"] == m)
            title = males + "M" + females + "F - " + str(m)
            label = int(particles) - 3 if int(particles) >= 3 else -1
            if label == -1:
                continue

            for j, col in enumerate(hoi_metrics):
                m_mean = col + "_mean_2"
                m_std = col + "_std_2"
                df_aux = df[mask].drop_duplicates(subset=g2)
                size = df_aux["size"].values
                ym = df_aux[m_mean].values
                ys = df_aux[m_std].values
                show_label = title not in used_labels[label][j]

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
                    label=title if show_label else None,
                    capsize=5,
                    ls="--",
                    lw=0.7,
                    fmt="o",
                    color=color
                )

                legend_handles_2.append(axes_2[label][j].lines[-1])
                legend_labels_2.append(title)

                if show_label:
                    used_labels[label][j].add(title)

            # Axis labels
            for j in range(4):
                axes_2[label][j].set_xlabel(
                    "Window size ($\\omega$)",
                    fontsize=14
                )
            axes_2[label][0].set_ylabel(r"$\Omega_{" + particles + r"}^{D}(t)$", fontsize=14)  # noqa: 501
            axes_2[label][1].set_ylabel(r"$\Omega_{" + particles + r"}^{\theta}(t)$", fontsize=14)  # noqa: 501
            axes_2[label][2].set_ylabel(r"$d\Omega_{" + particles + r"}^{D}(t)$", fontsize=14)  # noqa: 501
            axes_2[label][3].set_ylabel(r"$d\Omega_{" + particles + r"}^{\theta}(t)$", fontsize=14)  # noqa: 501

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
                ax[i][j].xaxis.set_minor_locator(mtick.MaxNLocator(4 * n_x_breaks))  # noqa: 501
                ax[i][j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax[i][j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))  # noqa: 501
                ax[i][j].tick_params(axis="x", labelrotation=90)

    fig_1.legend(
        legend_handles_1,
        legend_labels_1,
        loc="center left",
        bbox_to_anchor=(1.001, 0.5),
        fontsize=12,
        frameon=False,
        fancybox=fancy_legend
    )
    fig_1.tight_layout(rect=[0, 0, 0.99, 1])  # reserve space for legend

    fig_2.legend(
        legend_handles_2,
        legend_labels_2,
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
        fig_1.savefig(full_path_1, dpi=300)
        fig_2.savefig(full_path_2, dpi=300)
        print(f"Figure saved to {full_path_1} and {full_path_2}")
    else:
        plt.show()
    return df, axes_1, axes_2
