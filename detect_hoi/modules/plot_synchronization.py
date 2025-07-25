# -*- coding: utf-8 -*-
"""
Created on Friday March 6th 2025

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
# import matplotlib.cm as cm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mtick  # type: ignore
# import matplotlib.colors as mcolors  # type: ignore


# Plot cross-recurrence quantification analysis (CRQA) ----
def plot_crqa(
    df_metrics: pd.DataFrame,
    width: int = 24,
    height: int = 10,
    n_y_breaks: int = 20,
    save_figures: bool = False,
    output_path: str = "../output_files",
    output_name: str = "plot_gliding_summary"
):
    """
    Plot the cross-recurrence quantification analysis over different videos and
    sex ratios.

    Parameters:
    -----------
    df_metrics : pd.DataFrame
        CRQA metrics given by:
            - Recurrence Rate (RR): Ratio of recurrent points in the plot.
            - Determinism (DET): Fraction of recurrent points forming
            diagonal lines (indicating predictability).
            - Average Diagonal Line Length (L): Mean time over which the system
            exhibits similar behavior.
            - Entropy (ENTR): Shannon entropy of diagonal line lengths
            (complexity measure).
            - Laminarity (LAM): Fraction of points forming vertical lines
            (indicating intermittency or stationarity).
            - Trapping Time (TT): Average vertical line length.
            - Number of diagonal lines.
            - Number of vertical lines.
    width : int
        Width of final plot. Default value 10
    height : int
        Width of final plot. Default value 10
    n_y_breaks : int
        Number of divisions in y-axis. Default value 20
    save_figures: bool
        Save plots flag (default value False)
    output_path : string
        Local path for outputs. Default value is "../output_files"
    output_name : string
        Name of the outputs. Default value is "plot_gliding_summary"
    """
    # Options for plotting - Title, Position
    dicc_vars = {
        "RR": ["Recurrence ratio", [0, 0]],
        "DET": ["Determinism", [1, 0]],
        "L": ["Average diagonal line length", [0, 1]],
        "ENTR": ["Entropy of diagonal line length", [1, 1]],
        "LAM": ["Laminarity", [0, 2]],
        "TT": ["Trapping Time", [1, 2]]
    }
    df = df_metrics.copy()
    df["particles"] = df["video"].str[0]
    df["sex_ratio"] = df["video"].str[0:8]
    df["sex_ratio"] = df["sex_ratio"].str.replace("_", " ").str.capitalize()

    fig, axes = plt.subplots(2, 3, figsize=(width, height))
    sex_ratios = sorted(df["sex_ratio"].unique())
    for k, options in dicc_vars.items():
        # Plot options and data
        title = options[0]
        positions = options[1]
        x = positions[0]
        y = positions[1]
        v = [df[df["sex_ratio"] == sr][k].dropna().values for sr in sex_ratios]

        # Quantiles - Quintiles
        q1, q2, q3, q4, q5 = [], [], [], [], []
        for i in range(len(v)):
            q1_, q2_, q3_, q4_, q5_ = np.percentile(v[i], [0, 25, 50, 75, 100])
            q1.append(q1_)
            q2.append(q2_)
            q3.append(q3_)
            q4.append(q4_)
            q5.append(q5_)

        vp = axes[x][y].violinplot(
            v,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.5
        )
        indexes = np.arange(1, len(sex_ratios) + 1)
        axes[x][y].scatter(
            indexes,
            q3,
            marker="*",
            color="white",
            s=120,
            zorder=3,
            facecolors="#B0B0B0",
            edgecolor="#000000"
        )
        axes[x][y].vlines(indexes, q1, q5, color="k", linestyle="-", lw=1)
        axes[x][y].vlines(indexes, q2, q4, color="k", linestyle="-", lw=5)

        # Labels and ticks
        axes[x][y].set_xticks(indexes)
        axes[x][y].set_xticklabels(sex_ratios, rotation=90, ha="right")
        axes[x][y].set_xlabel("Sex Ratio")
        axes[x][y].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        axes[x][y].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
        axes[x][y].set_ylabel(title, fontsize=14)
        axes[x][y].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=11,
            length=12
        )
        axes[x][y].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            labelsize=11,
            length=6
        )

        # Violin options
        for body in vp["bodies"]:
            body.set_facecolor("#971010")
            body.set_edgecolor("#000000")
            body.set_alpha(1)

            # Get the center and modify the paths to not go further right
            m = np.mean(body.get_paths()[0].vertices[:, 0])
            body.get_paths()[0].vertices[:, 0] = np.clip(body.get_paths()[0].vertices[:, 0], -np.inf, m)  # noqa: 501

    fig.tight_layout()  # reserve space for legend

    if save_figures:
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, f"{output_name}.png")
        fig.savefig(full_path, dpi=400)
        print(f"Figure saved to {full_path}")
    plt.close()

    return fig, axes


# Plot Kuramoto order parameter ----
def plot_kuramoto_order_parameter(
    df_synchronization: pd.DataFrame,
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
    Plot the order parameter from the Kuramoto model over different videos.

    Parameters:
    -----------
    df_synchronization : pd.DataFrame
        A DataFrame containing the estimated Kuramoto parameter. The DataFrame
        includes the following columns:
            - "video": Video name.
            - "time": The timestamp at which the Kuramoto parameter was
            estimated.
            - "order_parameter": The estimated Kuramoto paramter.
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

    df = df_synchronization.copy()
    df["particles"] = df["video"].str[0]
    df["sex_ratio"] = df["video"].str[0:8]

    legend_labels = []
    legend_handles = []
    fig, axes = plt.subplots(3, 1, figsize=(width, height))

    for video in df["video"].unique():
        particles = video[0]
        label = int(particles) - 2 if int(particles) >= 2 else -1
        if label == -1:
            continue

        df_aux = df[df["video"] == video]
        t = df_aux["time"].values
        Rs = df_aux["order_parameter"].values

        axes[label].plot(t, Rs, label=video, marker="o", ms=4, ls="--", lw=0.7)

        legend_handles.append(axes[label].lines[-1])
        legend_labels.append(video)

        # Axes labels
        axes[label].set_xlabel("Time ($t$)", fontsize=14)

    axes[0].set_ylabel("$\\Psi_{{2}}^{{\\theta}}(t)$", fontsize=14)
    axes[1].set_ylabel("$\\Psi_{{3}}^{{\\theta}}(t)$", fontsize=14)
    axes[2].set_ylabel("$\\Psi_{{4}}^{{\\theta}}(t)$", fontsize=14)

    # Styling
    for i in range(3):
        axes[i].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=11,
            length=12
        )
        axes[i].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            labelsize=11,
            length=6
        )
        axes[i].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
        axes[i].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
        axes[i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        axes[i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
        axes[i].tick_params(axis="x", labelrotation=90)
        axes[i].set_ylim(0, 1)

    fig.legend(
        legend_handles,
        legend_labels,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.001, 0.5),
        fontsize=12,
        frameon=False,
        fancybox=fancy_legend
    )
    fig.tight_layout(rect=[0, 0, 0.99, 1])  # reserve space for legend

    if save_figures:
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, f"{output_name}.png")
        fig.savefig(full_path, dpi=400)
        print(f"Figure saved to {full_path}")
    plt.close()

    return fig, axes
