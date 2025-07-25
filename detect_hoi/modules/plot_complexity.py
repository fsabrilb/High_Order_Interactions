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


# Plot complexity measures ----
def plot_gliding_complexity(
    df_complexity,
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
    Plot the complexity metrics over many IDs for one video.

    Parameters:
    -----------
    df_complexity : pd.DataFrame
        A DataFrame containing the estimated $H(q)$, $S_{P}$ and $C_{JS}$. The
        DataFrame includes the following columns:
            - "video": Video name.
            - "t_range": The timestamp at which the Hurst exponent was
            estimated.
            - "size": The window size at which the Hurst exponent was
            estimated.
            - "permuted_id": The identifier for each distance ID.
            - "H": The estimated Hurst exponent.
            - "PE": The estimated permutation entropy.
            - "C": The estimated statistical complexity.
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
    legend_labels = []
    legend_handles = []
    fig, axes = plt.subplots(2, 3, figsize=(width, height))
    for video in df_complexity["video"].unique():
        particles = video[0]
        mask_1 = df_complexity["video"] == video
        for m in df_complexity[mask_1]["permuted_id"].unique():
            mask = mask_1 & (df_complexity["permuted_id"] == m)
            title = video + " - " + str(m)

            # Time series data
            df = df_complexity[mask]
            s = df["size"].values
            Hd = df["H_distance"].values
            Pd = df["PE_distance"].values
            Cd = df["C_distance"].values
            Ho = df["H_orientation"].values
            Po = df["PE_orientation"].values
            Co = df["C_orientation"].values

            # Plot into axes
            for j, y in enumerate([Hd, Pd, Cd, Ho, Po, Co]):
                if j == 2:
                    x = Pd
                    xlabel = r"$PE_{" + particles + r"}^{D}(\omega)$"
                elif j == 5:
                    x = Po
                    xlabel = r"$PE_{" + particles + r"}^{\theta}(\omega)$"
                else:
                    x = s
                    xlabel = "Window size ($\\omega$)"
                axes[j // 3][j % 3].plot(
                    x,
                    y,
                    label=title,
                    marker="o",
                    ls="",
                    ms=4
                )
                legend_handles.append(axes[j // 3][j % 3].lines[-1])
                legend_labels.append(title)

                # Axes labels
                axes[j // 3][j % 3].set_xlabel(xlabel, fontsize=14)

            axes[0][0].set_ylabel(r"$H_{" + particles + r"}^{D}(\omega)$", fontsize=14)  # noqa: 501
            axes[0][1].set_ylabel(r"$PE_{" + particles + r"}^{D}(\omega)$", fontsize=14)  # noqa: 501
            axes[0][2].set_ylabel(r"$C_{" + particles + r"}^{D}(\omega)$", fontsize=14)  # noqa: 501
            axes[1][0].set_ylabel(r"$H_{" + particles + r"}^{\theta}(\omega)$", fontsize=14)  # noqa: 501
            axes[1][1].set_ylabel(r"$PE_{" + particles + r"}^{\theta}(\omega)$", fontsize=14)  # noqa: 501
            axes[1][2].set_ylabel(r"$C_{" + particles + r"}^{\theta}(\omega)$", fontsize=14)  # noqa: 501

    # Global plot settings
    for i in range(2):
        for j in range(3):
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
        list(set(legend_handles)),
        list(set(legend_labels)),
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.001, 0.5),
        fontsize=12,
        frameon=True,
        fancybox=fancy_legend
    )
    plt.tight_layout(rect=[0, 0, 0.99, 1])  # reserve space for legend

    if save_figure:
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, f"{output_name}.png")
        fig.savefig(full_path, dpi=400)
        print(f"Figure saved to {full_path}")
    plt.close()

    return fig, axes


# Plot complexity measures (Summary) ----
def plot_complexity_metrics_summary(
    df_complexity: pd.DataFrame,
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
    Plot the complexity over many IDs.

    Parameters:
    -----------
    df_complexity : pd.DataFrame
        A DataFrame containing the estimated $H(q)$, $S_{P}$ and $C_{JS}$. The
        DataFrame includes the following columns:
            - "video": Video name.
            - "t_range": The timestamp at which the Hurst exponent was
            estimated.
            - "size": The window size at which the Hurst exponent was
            estimated.
            - "permuted_id": The identifier for each distance ID.
            - "H": The estimated Hurst exponent.
            - "PE": The estimated permutation entropy.
            - "C": The estimated statistical complexity.
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

    df = df_complexity.copy()
    df["particles"] = df["video"].str[0]
    df["sex_ratio"] = df["video"].str[0:8]

    complexity_metrics = [
        "H_distance", "PE_distance", "C_distance",
        "H_orientation", "PE_orientation", "C_orientation"
    ]

    g1 = ["video", "permuted_id", "size"]
    g2 = ["sex_ratio", "permuted_id", "size"]
    for col in complexity_metrics:
        for i, g in enumerate([g1, g2], start=1):
            c = col + "_count_" + str(i)
            m = col + "_mean_" + str(i)
            s = col + "_std_" + str(i)
            df[c] = df.groupby(g)[col].transform("count")
            df[m] = df.groupby(g)[col].transform("mean")
            df[s] = df.groupby(g)[col].transform("std") / np.sqrt(df[c])

    # Unique keys and combinations and color mapping
    k1 = df[["video", "permuted_id"]].drop_duplicates()
    k1["label_key"] = k1["video"] + "_" + k1["permuted_id"].astype(str)
    map_1 = cm.get_cmap("plasma", len(k1))
    label_color_1 = {
        key: mcolors.to_hex(map_1(i)) for i, key in enumerate(k1["label_key"])
    }

    k2 = df[["sex_ratio", "permuted_id"]].drop_duplicates()
    k2["label_key"] = k2["sex_ratio"] + "_" + k2["permuted_id"].astype(str)
    map_2 = cm.get_cmap("plasma", len(k2))
    label_color_2 = {
        key: mcolors.to_hex(map_2(i)) for i, key in enumerate(k2["label_key"])
    }

    # Figure 1 - Video
    legend_labels_1 = []
    legend_handles_1 = []
    fig_1, axes_1 = plt.subplots(3, 6, figsize=(width, height))
    used_labels = [[set() for _ in range(6)] for _ in range(3)]  # Avoid duplic

    for group in df["video"].unique():
        particles = group[0]
        males = group[3]
        females = group[6]
        mask_1 = df["video"] == group
        for m in df[mask_1]["permuted_id"].unique():
            label_key = group + "_" + str(m)
            color = label_color_1[label_key]
            mask = mask_1 & (df["permuted_id"] == m)
            title = group + " - " + str(m)
            label = int(particles) - 2 if int(particles) >= 2 else -1
            if label == -1:
                continue

            for j, col in enumerate(complexity_metrics):
                m_mean = col + "_mean_1"
                m_std = col + "_std_1"
                df_aux = df[mask].drop_duplicates(subset=g1)
                size = df_aux["size"].values
                ym = df_aux[m_mean].values
                ys = df_aux[m_std].values
                show_label = title not in used_labels[label][j]

                if j == 2:
                    x = df_aux["PE_distance_mean_1"].values
                    xs = df_aux["PE_distance_std_1"].values
                    xlabel = r"$S_{" + particles + r"}^{D}(t)$"
                elif j == 5:
                    x = df_aux["PE_orientation_mean_1"].values
                    xs = df_aux["PE_orientation_std_1"].values
                    xlabel = r"$S_{" + particles + r"}^{\theta}(t)$"
                else:
                    x = size
                    xs = np.zeros(len(df_aux))
                    xlabel = "Window size ($\\omega$)"

                # Plot error bars
                axes_1[label][j].errorbar(
                    x,
                    ym,
                    xerr=xs,
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
            for j in range(6):
                axes_1[label][j].set_xlabel(xlabel, fontsize=14)
            axes_1[label][0].set_ylabel(r"$H_{" + particles + r"}^{D}(\omega)$", fontsize=14)  # noqa: 501
            axes_1[label][1].set_ylabel(r"$PE_{" + particles + r"}^{D}(\omega)$", fontsize=14)  # noqa: 501
            axes_1[label][2].set_ylabel(r"$C_{" + particles + r"}^{D}(\omega)$", fontsize=14)  # noqa: 501
            axes_1[label][3].set_ylabel(r"$H_{" + particles + r"}^{\theta}(\omega)$", fontsize=14)  # noqa: 501
            axes_1[label][4].set_ylabel(r"$PE_{" + particles + r"}^{\theta}(\omega)$", fontsize=14)  # noqa: 501
            axes_1[label][5].set_ylabel(r"$C_{" + particles + r"}^{\theta}(\omega)$", fontsize=14)  # noqa: 501

    # Figure 2 - Sex ratio
    legend_labels_2 = []
    legend_handles_2 = []
    fig_2, axes_2 = plt.subplots(3, 6, figsize=(width, height))
    used_labels = [[set() for _ in range(6)] for _ in range(3)]  # Avoid duplic
    for group in df["sex_ratio"].unique():
        particles = group[0]
        males = group[3]
        females = group[6]
        mask_1 = df["sex_ratio"] == group
        for m in df[mask_1]["permuted_id"].unique():
            label_key = group + "_" + str(m)
            color = label_color_2[label_key]
            mask = mask_1 & (df["permuted_id"] == m)
            title = males + "M" + females + "F - " + str(m)
            label = int(particles) - 2 if int(particles) >= 2 else -1
            if label == -1:
                continue

            for j, col in enumerate(complexity_metrics):
                m_mean = col + "_mean_2"
                m_std = col + "_std_2"
                df_aux = df[mask].drop_duplicates(subset=g2)
                size = df_aux["size"].values
                ym = df_aux[m_mean].values
                ys = df_aux[m_std].values
                show_label = title not in used_labels[label][j]

                if j == 2:
                    x = df_aux["PE_distance_mean_2"].values
                    xs = df_aux["PE_distance_std_2"].values
                    xlabel = r"$S_{" + particles + r"}^{D}(t)$"
                elif j == 5:
                    x = df_aux["PE_orientation_mean_2"].values
                    xs = df_aux["PE_orientation_std_2"].values
                    xlabel = r"$S_{" + particles + r"}^{\theta}(t)$"
                else:
                    x = size
                    xs = np.zeros(len(df_aux))
                    xlabel = "Window size ($\\omega$)"

                # Plot error bars
                axes_2[label][j].errorbar(
                    x,
                    ym,
                    xerr=xs,
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
            for j in range(6):
                axes_2[label][j].set_xlabel(xlabel, fontsize=14)
            axes_2[label][0].set_ylabel(r"$H_{" + particles + r"}^{D}(t)$", fontsize=14)  # noqa: 501
            axes_2[label][1].set_ylabel(r"$PE_{" + particles + r"}^{D}(t)$", fontsize=14)  # noqa: 501
            axes_2[label][2].set_ylabel(r"$C_{" + particles + r"}^{D}(t)$", fontsize=14)  # noqa: 501
            axes_2[label][3].set_ylabel(r"$H_{" + particles + r"}^{\theta}(t)$", fontsize=14)  # noqa: 501
            axes_2[label][4].set_ylabel(r"$PE_{" + particles + r"}^{\theta}(t)$", fontsize=14)  # noqa: 501
            axes_2[label][5].set_ylabel(r"$C_{" + particles + r"}^{\theta}(t)$", fontsize=14)  # noqa: 501

    # Styling
    for i in range(3):
        for j in range(6):
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
        ncol=1,
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
        ncol=1,
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
        fig_1.savefig(full_path_1, dpi=400)
        fig_2.savefig(full_path_2, dpi=400)
        print(f"Figure saved to {full_path_1} and {full_path_2}")
    plt.close()
    plt.close()

    return df, fig_1, fig_2, axes_1, axes_2
