# -*- coding: utf-8 -*-
"""
Created on Friday October 4th 2024

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)


# Plot frame without additional processes ----
def plot_velocities_distribution(
    df_tracked: pd.DataFrame,
    bins: int = 10,
    velocity_threshold: float = 100,
    omega_threshold: float = np.pi / 4,
    width: int = 10,
    fancy_legend: bool = False
):
    """Plot velocities distribution for smooth evolution of the particles

    Args:
    ---------------------------------------------------------------------------
    df_tracked : pandas DataFrame
        Dataframe with the information of tracked regions with the following
        columns:
            - id: Particle ID
            - time: Time (Frame)
            - position_x: Centroid position in X-axis
            - position_y: Centroid position in Y-axis
            - weighted_x: Centroid position in X-axis weighted with intensity
            image
            - weighted_y: Centroid position in Y-axis weighted with intensity
            image
            - darkest_v: Intensity of darkest pixel in the local region
            (particle)
            - darkest_x: Position of darkest pixel in the local region
            (particle)
            - darkest_y: Position of darkest pixel in the local region
            (particle)
            - lightest_v: Intensity of lightest pixel in the local region
            (particle)
            - lightest_x: Position of lightest pixel in the local region
            (particle)
            - lightest_y: Position of lightest pixel in the local region
            (particle)
            - coords_x: X values of the region's boundary
            - coords_y: Y values of the region's boundary
            - orientation: Orientation respect rows
            - area: Region area i.e. number of pixels of the region scaled by
            pixel-area
            - area_convex: Area of the convex hull image, which is the smallest
            convex polygon that encloses the region
            - area_filled: Area of the region with all the holes filled in
            - axis_major: The length of the major axis of the ellipse that has
            the same normalized second central moments as the region
            - axis_minor: The length of the minor axis of the ellipse that has
            the same normalized second central moments as the region
            - eccentricity: Eccentricity of the ellipse that has the same
            second-moments as the region. The eccentricity is the ratio of the
            focal distance (distance between focal points) over the major axis
            length. The value is in the interval [0, 1). When it is 0, the
            ellipse becomes a circle
            - euler_number: Euler characteristic of the set of non-zero pixels.
            Computed as number of connected components subtracted by number of
            holes (input.ndim connectivity). In 3D, number of connected
            components plus number of holes subtracted by number of tunnels
            - velocity_x: Velocity in X-axis
            - velocity_y: Velocity in Y-axis
            - velocity_orientation: Angular velocity
    bins : int
        Number of numerical ranges in which the histogram data is aggregated
        (default value 10)
    velocity_threshold : float
        Maximmum velocity in X-axis or Y-axis allowed between identical IDs
        Default value is 100
    omega_threshold : float
        Maximmum angular velocity between identical IDs. Default value is pi/4
    width : int
        Width of final plot. Default value 10
    fancy_legend : bool
        Fancy legend output (default value False)

    Returns:
    ---------------------------------------------------------------------------
    None
    """

    df_tracked["mask_x"] = np.where(df_tracked["velocity_x"] >= velocity_threshold, True, False)  # noqa: 501
    df_tracked["mask_y"] = np.where(df_tracked["velocity_y"] >= velocity_threshold, True, False)  # noqa: 501
    df_tracked["mask_orientation"] = np.where(df_tracked["velocity_orientation"] >= omega_threshold, True, False)  # noqa: 501

    title = "Velocity Distribution"
    n_particles = df_tracked["id"].nunique()

    num_colors = 3 * n_particles
    colors = np.linspace(0, 1, num_colors)
    cmap = plt.get_cmap("autumn", num_colors)
    cmap.set_under("black")

    fig, ax = plt.subplots(n_particles, 3)
    fig.set_size_inches(w=3 * width, h=n_particles * width)

    for c_, id_ in enumerate(df_tracked["id"].unique()):
        df_aux = df_tracked[df_tracked["id"] == id_]
        hist_x, bins_x = np.histogram(df_aux["velocity_x"].dropna().abs(), bins=bins, density=True)  # noqa: 501
        hist_y, bins_y = np.histogram(df_aux["velocity_y"].dropna().abs(), bins=bins, density=True)  # noqa: 501
        hist_a, bins_a = np.histogram(df_aux["velocity_orientation"].dropna().abs(), bins=bins, density=True)  # noqa: 501
        percentage_x = 100 * df_aux["mask_x"].sum() / df_aux.shape[0]
        percentage_y = 100 * df_aux["mask_y"].sum() / df_aux.shape[0]
        percentage_a = 100 * df_aux["mask_orientation"].sum() / df_aux.shape[0]

        # Midpoint per bin
        bins_midpoint_x = np.zeros_like(hist_x)
        bins_midpoint_y = np.zeros_like(hist_y)
        bins_midpoint_a = np.zeros_like(hist_a)
        for k in np.arange(0, len(hist_x), 1):
            bins_midpoint_x[k] = 0.5 * (bins_x[k] + bins_x[k+1])
            bins_midpoint_y[k] = 0.5 * (bins_y[k] + bins_y[k+1])
            bins_midpoint_a[k] = 0.5 * (bins_a[k] + bins_a[k+1])

        ax[c_][0].plot(
            bins_midpoint_x,
            hist_x,
            marker="o",
            c=cmap(colors[id_]),
            ms=8,
            ls="-",
            label=r"$dx_{{{}}}$".format(id_)
        )  # Velocity X-axis
        ax[c_][0].hist(
            df_aux["velocity_x"].dropna().abs(),
            bins=bins,
            alpha=0.19,
            facecolor="blue",
            edgecolor="darkblue",
            density=True,
            histtype="stepfilled",
            cumulative=False,
            lw=6,
            label=r"$dx_{{{}}}$".format(id_)
        )  # Velocity X-axis
        ax[c_][1].plot(
            bins_midpoint_y,
            hist_y,
            marker="o",
            c=cmap(colors[id_]),
            ms=8,
            ls="-",
            label=r"$dy_{{{}}}$".format(id_)
        )  # Velocity X-axis
        ax[c_][1].hist(
            df_aux["velocity_y"].dropna().abs(),
            bins=bins,
            alpha=0.19,
            facecolor="blue",
            edgecolor="darkblue",
            density=True,
            histtype="stepfilled",
            cumulative=False,
            lw=6,
            label=r"$dy_{{{}}}$".format(id_)
        )  # Velocity Y-axis
        ax[c_][2].plot(
            bins_midpoint_a,
            hist_a,
            marker="o",
            c=cmap(colors[id_]),
            ms=8,
            ls="-",
            label=r"$d\theta_{{{}}}$".format(id_)
        )  # Velocity orientation
        ax[c_][2].hist(
            df_aux["velocity_orientation"].dropna().abs(),
            bins=bins,
            alpha=0.19,
            facecolor="blue",
            edgecolor="darkblue",
            density=True,
            histtype="stepfilled",
            cumulative=False,
            lw=6,
            label=r"$d\theta_{{{}}}$".format(id_)
        )  # Velocity orientation

        min_x, max_x = np.min(hist_x), np.max(np.abs(hist_x))
        min_y, max_y = np.min(hist_y), np.max(np.abs(hist_y))
        min_a, max_a = np.min(hist_a), np.max(np.abs(hist_a))

        # ax[c_][0].vlines(x=-velocity_threshold, ymin=min_x, ymax=max_x, lw=4, color="black")  # noqa: 501
        # ax[c_][1].vlines(x=-velocity_threshold, ymin=min_y, ymax=max_y, lw=4, color="black")  # noqa: 501
        # ax[c_][2].vlines(x=-omega_threshold, ymin=min_a, ymax=max_a, lw=4, color="black")  # noqa: 501
        ax[c_][0].vlines(x=velocity_threshold, ymin=min_x, ymax=max_x, lw=4, color="black")  # noqa: 501
        ax[c_][1].vlines(x=velocity_threshold, ymin=min_y, ymax=max_y, lw=4, color="black")  # noqa: 501
        ax[c_][2].vlines(x=omega_threshold, ymin=min_a, ymax=max_a, lw=4, color="black")  # noqa: 501

        titles = ["X", "Y", "\\theta"]
        percentages = np.array([percentage_x, percentage_y, percentage_a])
        for j in range(3):
            # ax[c_][j].set_xscale("symlog", subs=[2, 3, 4, 5, 6, 7, 8, 9])
            ax[c_][j].set_yscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])
            ax[c_][j].tick_params(axis="x", labelrotation=90)
            ax[c_][j].legend(fancybox=fancy_legend, shadow=True, ncol=1)
            ax[c_][j].set_title(
                r"({}) {} $P_{{{}}}$ - ${}$ - Loss: ${}\%$".format(
                    chr(65 + j),
                    title,
                    c_,
                    titles[j],
                    np.round(percentages[j], 2)
                ),
                loc="left",
                y=1.005
            )

    plt.show()
