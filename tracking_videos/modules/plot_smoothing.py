# -*- coding: utf-8 -*-
"""
Created on Friday October 4th 2024

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import track_particles as tp  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mtick  # type: ignore
import clustering_particles as cp  # type: ignore
import matplotlib.animation as animation  # type: ignore

from skimage.color import rgb2gray  # type: ignore


# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
# plt.rcParams['animation.ffmpeg_path'] = r"C:\ffmpeg\bin\ffmpeg.exe"


# Plot frame without additional processes ----
def plot_tracking_evolution(
    df_smooth: pd.DataFrame,
    width: int = 10,
    n_x_breaks: int = 20,
    n_y_breaks: int = 20,
    t_bounds: list = [0, 100],
    p_bounds: list = [[0, 1500], [0, 1000], [-1.6, 1.6]],
    fancy_legend: bool = False
):
    """Plot positions and orientation evolution for smoothed evolution of the
    particles

    Args:
    ---------------------------------------------------------------------------
    df_smooth : pandas DataFrame
        Dataframe with the smoothed information of tracked regions with the
        following columns:
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
            - mask_x: Flag for long-jump in x-axis
            - mask_y: Flag for long-jump in y-axis
            - mask_orientation: Flag for flip of the head-bump orientation
    width : int
        Width of final plot. Default value 10
    n_x_breaks : int
        Number of divisions in x-axis. Default value 20
    n_y_breaks : int
        Number of divisions in y-axis. Default value 20
    fancy_legend : bool
        Fancy legend output (default value False)
    t_bounds : list
        Bound in time axis. Default value is [0, 100]
    p_bounds : list
        Bound in Y-axes for positions and orientation. Default value is
        [[0, 1500], [0, 1000], [-1.6, 1.6]]

    Returns:
    ---------------------------------------------------------------------------
    None
    """

    num_colors = df_smooth["id"].nunique()
    colors = np.linspace(0, 1, num_colors)
    cmap = plt.get_cmap("autumn", num_colors)
    cmap.set_under("black")
    markers = ["o", "v", "^", "*"]

    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(w=width, h=width)
    mask_time = ((df_smooth["time"] >= t_bounds[0]) & (df_smooth["time"] <= t_bounds[1]))  # noqa: 501

    for c_, id_ in enumerate(df_smooth["id"].unique()):
        df_aux = df_smooth[(df_smooth["id"] == id_) & (mask_time)]
        ax[0].plot(
            df_aux["time"],
            df_aux["position_x"],
            marker=markers[c_%4],
            c=cmap(colors[c_]),
            ms=2,
            ls="-",
            lw=0.7,
            label=r"$x_{{{}}}$".format(id_)
        )
        ax[1].plot(
            df_aux["time"],
            df_aux["position_y"],
            marker=markers[c_%4],
            c=cmap(colors[c_]),
            ms=2,
            ls="-",
            lw=0.7,
            label=r"$y_{{{}}}$".format(id_)
        )
        ax[2].plot(
            df_aux["time"],
            df_aux["orientation"],
            marker=markers[c_%4],
            c=cmap(colors[c_]),
            ms=2,
            ls="-",
            lw=0.7,
            label=r"$\theta_{{{}}}$".format(id_)
        )

        titles = ["Position $X(t)$", "Position $Y(t)$", "Orientation $\\theta(t)$"]
        y_labels = ["Position X-axis $X(t)$", "Position Y-axis $Y(t)$", "Orientation $\theta(t)$"]
    
        for j in [0, 1, 2]:
            ax[j].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
            ax[j].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
            ax[j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
            ax[j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
            ax[j].tick_params(axis="x", labelrotation=90)
            ax[j].set_xlabel("Time $t$")  
            ax[j].set_ylabel(y_labels[j])  
            ax[j].legend(fancybox=fancy_legend, shadow=True, ncol=1)
            ax[j].set_title(r"({}) {}".format(chr(65 + j), titles[j]), loc="left", y=1.005)
            ax[j].tick_params(which = "major", direction = "in", top = True, right = True, length = 12)
            ax[j].tick_params(which = "minor", direction = "in", top = True, right = True, length = 6)
            ax[j].set_facecolor("silver")
            ax[j].set_xlim(t_bounds[0], t_bounds[1])
            ax[j].set_ylim(p_bounds[j][0], p_bounds[j][1])
    
    plt.show()


# Plot frame with tracked particle ----
def plot_tracking_frame_smoothed(
    reader,
    current_time: int,
    previous_time: int,
    df_tracked: pd.DataFrame,
    df_smooth: pd.DataFrame,
    width: int = 10,
    n_x_breaks: int = 20,
    n_y_breaks: int = 20,
    x_bounds: list = [300, 1460],
    y_bounds: list = [240, 900],
    x_zoom: list = [760, 880],
    y_zoom: list = [320, 520],
    fancy_legend: bool = False,
    x_legend: float = 1,
    y_legend: float = 1
):
    """Plot a particular frame for video analysis with profiled process and
    boundaries (edges and contours)

    Args:
    ---------------------------------------------------------------------------
    reader : imagaeio object
        Imageio array with all the frames extracted from the video
    current_time : int
        Index position of the frames detected in reader
    previous_time : int
        Index position of the frames detected in reader
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
            - mask_x: Flag for long-jump in x-axis
            - mask_y: Flag for long-jump in y-axis
            - mask_orientation: Flag for flip of the head-bump orientation
    df_smooth : pandas DataFrame
        Dataframe with the smoothed information of tracked regions with the
        same columns
    width : int
        Width of final plot. Default value 10
    n_x_breaks : int
        Number of divisions in x-axis. Default value 20
    n_y_breaks : int
        Number of divisions in y-axis. Default value 20
    x_bounds : list
        Bound in X-axis for the frame. Remember the axes are rotated with
        imageio. Default value is [300, 1460]
    y_bounds : list
        Bound in Y-axis for the frame. Remember the axes are rotated with
        imageio. Default value is [240, 900]
    x_zoom : list
        Zoom in X-axis for the frame. Remember the axes are rotated with
        imageio. Default value is [760, 880]
    y_zoom : list
        Zoom in Y-axis for the frame. Remember the axes are rotated with
        imageio. Default value is [320, 520]
    fancy_legend : bool
        Fancy legend output (default value False)
    x_legend : float
        X position of graph legend (default value 1)
    y_legend : float
        Y position of graph legend (default value 1)

    Returns:
    ---------------------------------------------------------------------------
    None
    """

    frame = reader.get_data(current_time)
    previous_frame = reader.get_data(previous_time)
    title = "Tracked Particles"
    if x_zoom is not None:
        x_bounds = x_zoom
        title = "{} - Zoom [{},{}]".format(title, x_zoom[0], x_zoom[1])
        if y_zoom is not None:
            y_bounds = y_zoom
            title = "{}$\\times$[{},{}]".format(
                title,
                x_zoom[0],
                x_zoom[1],
                y_zoom[0],
                y_zoom[1]
            )

    frame = frame[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]
    previous_frame = previous_frame[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]

    num_colors = df_tracked["id"].unique().shape[0]
    colors = np.linspace(0, 1, num_colors)
    cmap = plt.get_cmap("autumn", num_colors)
    cmap.set_under("black")

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(w=2 * width, h=2 * (frame.shape[0] * width / frame.shape[1]))
    ax[0][0].imshow(previous_frame, cmap="gray")
    ax[1][0].imshow(previous_frame, cmap="gray")
    ax[0][1].imshow(frame, cmap="gray")
    ax[1][1].imshow(frame, cmap="gray")

    for id_ in df_tracked["id"].unique():
        df_aux = df_tracked[df_tracked["id"] == id_]
        for c_, time in enumerate(df_aux["time"].unique()):
            df_aux_1 = df_tracked[(df_tracked["id"] == id_) & (df_tracked["time"] == time)]  # noqa: 501
            df_aux_2 = df_smooth[(df_smooth["id"] == id_) & (df_smooth["time"] == time)]  # noqa: 501
            
            ax[c_][0].plot(
                df_aux_1["position_x"].values,
                df_aux_1["position_y"].values,
                marker="o",
                c=cmap(colors[id_]),
                ms=4,
                ls="",
                label=r"$p_{{{}}}$".format(id_)
            )  # Centroid
            ax[c_][0].plot(
                df_aux_1["weighted_x"].values,
                df_aux_1["weighted_y"].values,
                marker="v",
                c=cmap(colors[id_]),
                ms=4,
                ls="",
                label=r"$w_{{{}}}$".format(id_)
            )  # Centroid weighted
            ax[c_][0].plot(
                df_aux_1["darkest_x"].values,
                df_aux_1["darkest_y"].values,
                marker="x",
                c=cmap(colors[id_]),
                ms=4,
                ls="",
                label=r"$d_{{{}}}$".format(id_)
            )  # Darkest pixel
            ax[c_][0].plot(
                df_aux_1["lightest_x"].values,
                df_aux_1["lightest_y"].values,
                marker="^",
                c=cmap(colors[id_]),
                ms=4,
                ls="",
                label=r"$l_{{{}}}$".format(id_)
            )  # Lightest pixel

            ax[c_][1].plot(
                df_aux_2["position_x"].values,
                df_aux_2["position_y"].values,
                marker="o",
                c=cmap(colors[id_]),
                ms=4,
                ls="",
                label=r"$p_{{{}}}^{{s}}$".format(id_)
            )  # Centroid
            ax[c_][1].plot(
                df_aux_2["weighted_x"].values,
                df_aux_2["weighted_y"].values,
                marker="v",
                c=cmap(colors[id_]),
                ms=4,
                ls="",
                label=r"$w_{{{}}}^{{s}}$".format(id_)
            )  # Centroid weighted
            ax[c_][1].plot(
                df_aux_2["darkest_x"].values,
                df_aux_2["darkest_y"].values,
                marker="x",
                c=cmap(colors[id_]),
                ms=4,
                ls="",
                label=r"$d_{{{}}}^{{s}}$".format(id_)
            )  # Darkest pixel
            ax[c_][1].plot(
                df_aux_2["lightest_x"].values,
                df_aux_2["lightest_y"].values,
                marker="^",
                c=cmap(colors[id_]),
                ms=4,
                ls="",
                label=r"$l_{{{}}}^{{s}}$".format(id_)
            )  # Lightest pixel

    times = [previous_time, current_time]
    smoothed = ["", "smoothed"]
    for i in [0, 1]:
        for j in [0, 1]:
            ax[i][j].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
            ax[i][j].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
            ax[i][j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
            ax[i][j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
            ax[i][j].tick_params(axis="x", labelrotation=90)
            ax[i][j].set_title(
                r"({}) {} {} - Time $t={}$".format(
                    chr(65 + j + 2 * i),
                    title,
                    smoothed[i],
                    times[j]
                ),
                loc="left",
                y=1.005
            )
            if y_zoom is not None:
                ax[i][j].set_xlim([0, y_zoom[1] - y_zoom[0]])
            else:
                ax[i][j].set_xlim([0, y_bounds[1] - y_bounds[0]])
            if x_zoom is not None:
                ax[i][j].set_ylim([x_zoom[1] - x_zoom[0], 0])
            else:
                ax[i][j].set_ylim([x_bounds[1] - x_bounds[0], 0])

    # Combine legends into one and drop duplicates
    handles_0, labels_0 = ax[0][0].get_legend_handles_labels()
    handles_1, labels_1 = ax[1][0].get_legend_handles_labels()
    handles_2, labels_2 = ax[0][1].get_legend_handles_labels()
    handles_3, labels_3 = ax[1][1].get_legend_handles_labels()
    unique = dict(zip(
        labels_0 + labels_1 + labels_2 + labels_3,
        handles_0 + handles_1 + handles_2 + handles_3
    ))

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        fancybox=fancy_legend,
        shadow=False,
        ncol=2,
        borderaxespad=0.01,
        bbox_to_anchor=(x_legend, y_legend),
        bbox_transform=fig.transFigure
    )

    plt.show()