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
def plot_all_process_frame(
    reader,
    times: np.ndarray,
    df_tracked: pd.DataFrame,
    df_clustered: pd.DataFrame,
    df_smoothed: pd.DataFrame,
    width: int = 10,
    n_x_breaks: int = 20,
    n_y_breaks: int = 20,
    x_bounds: list = [300, 1460],
    y_bounds: list = [240, 900],
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
    times : int
        Index positions of the frames detected in the reader
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
    df_clustered : pandas DataFrame
        Dataframe with the information of clustered regions with the same
        columns
    df_smoothed : pandas DataFrame
        Dataframe with the information of smoothed regions with the same
        columns
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
    fancy_legend : bool
        Fancy legend output (default value False)
    x_legend : float
        X position of graph legend (default value 1)
    y_legend : float
        Y position of graph legend (default value 1)

    Returns:
    ---------------------------------------------------------------------------
    fig: object
        Return of figure with size 3x3 where the columns correspond to frames
        and rows to tracking, clustering and smoothing process
    """

    frame_0 = reader.get_data(times[0])
    frame_1 = reader.get_data(times[1])
    frame_2 = reader.get_data(times[2])
    frame_0 = frame_0[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]
    frame_1 = frame_1[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]
    frame_2 = frame_2[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]
    titles = ["Tracked Particles", "Clustered Particles", "Smoothed Particles"]

    num_colors = df_tracked["id"].unique().shape[0]
    colors = np.linspace(0, 1, num_colors)
    cmap = plt.get_cmap("autumn", num_colors)
    cmap.set_under("black")

    fig, ax = plt.subplots(3, 3)
    fig.set_size_inches(w=3* width, h=3* (frame_0.shape[0] * width / frame_0.shape[1]))

    for i in [0, 1, 2]:  # Frame per column
        ax[i][0].imshow(frame_0, cmap="gray")
        ax[i][1].imshow(frame_1, cmap="gray")
        ax[i][2].imshow(frame_2, cmap="gray")

    for id_ in df_tracked["id"].unique():
        for c_, time in enumerate(times):
            try:
                df_aux_1 = df_tracked[(df_tracked["id"] == id_) & (df_tracked["time"] == time)]  # noqa: 501
                df_aux_2 = df_clustered[(df_clustered["id"] == id_) & (df_clustered["time"] == time)]  # noqa: 501
                df_aux_3 = df_smoothed[(df_smoothed["id"] == id_) & (df_smoothed["time"] == time)]  # noqa: 501
                length = 90

                # Positions (Trackes (T), Clustered (C), Smoothed (S))
                px_t = df_aux_1["position_x"].values
                wx_t = df_aux_1["weighted_x"].values
                dx_t = df_aux_1["darkest_x"].values
                lx_t = df_aux_1["lightest_x"].values
                py_t = df_aux_1["position_y"].values
                wy_t = df_aux_1["weighted_y"].values
                dy_t = df_aux_1["darkest_y"].values
                ly_t = df_aux_1["lightest_y"].values
                angle_t = df_aux_1["orientation"].values
                arx_t = px_t[0]
                ary_t = py_t[0]
                ax_t = length * np.sin(angle_t)[0]
                ay_t = length * np.cos(angle_t)[0]
                col_1 = cmap(colors[id_])

                px_c = df_aux_2["position_x"].values
                wx_c = df_aux_2["weighted_x"].values
                dx_c = df_aux_2["darkest_x"].values
                lx_c = df_aux_2["lightest_x"].values
                py_c = df_aux_2["position_y"].values
                wy_c = df_aux_2["weighted_y"].values
                dy_c = df_aux_2["darkest_y"].values
                ly_c = df_aux_2["lightest_y"].values
                angle_c = df_aux_2["orientation"].values
                arx_c = px_c[0]
                ary_c = py_c[0]
                ax_c = length * np.sin(angle_c)[0]
                ay_c = length * np.cos(angle_c)[0]
                col_2 = cmap(colors[id_])

                px_s = df_aux_3["position_x"].values
                wx_s = df_aux_3["weighted_x"].values
                dx_s = df_aux_3["darkest_x"].values
                lx_s = df_aux_3["lightest_x"].values
                py_s = df_aux_3["position_y"].values
                wy_s = df_aux_3["weighted_y"].values
                dy_s = df_aux_3["darkest_y"].values
                ly_s = df_aux_3["lightest_y"].values
                angle_s = df_aux_3["orientation"].values
                arx_s = px_s[0]
                ary_s = py_s[0]
                ax_s = length * np.sin(angle_s)[0]
                ay_s = length * np.cos(angle_s)[0]
                col_3 = cmap(colors[id_])

                print("time: {} id: {} T: {} C: {} S:{}".format(
                    time, id_,
                    np.round(angle_t[0] * 180 / np.pi, 4),
                    np.round(angle_c[0] * 180 / np.pi, 4),
                    np.round(angle_s[0] * 180 / np.pi, 4)
                ))

                # ----------------------------------------------------- Tracked -----------------------------------------------------  # noqa: 501
                ax[0][c_].plot(px_t, py_t, marker="o", c=col_1, ms=4, ls="", label=r"$p_{{{}}}$".format(id_))  # noqa: 501
                ax[0][c_].plot(wx_t, wy_t, marker="v", c=col_1, ms=4, ls="", label=r"$w_{{{}}}$".format(id_))  # noqa: 501
                ax[0][c_].plot(dx_t, dy_t, marker="x", c=col_1, ms=4, ls="", label=r"$d_{{{}}}$".format(id_))  # noqa: 501
                ax[0][c_].plot(lx_t, ly_t, marker="^", c=col_1, ms=4, ls="", label=r"$l_{{{}}}$".format(id_))  # noqa: 501
                ax[0][c_].arrow(
                    x=arx_t,
                    y=ary_t,
                    dx=ax_t,
                    dy=ay_t,
                    fc=col_1,
                    ec=col_1,
                    head_width=20,
                    head_length=20,
                    ls="-",
                    label=r"$p_{{{}}}$".format(id_)
                )  # Orientation
                ax[0][c_].plot(
                    df_aux_1["coords_x"].values[0],
                    df_aux_1["coords_y"].values[0],
                    c=col_1,
                    alpha=0.18,
                    marker="",
                    ls="--",
                    label=r"$p_{{{}}}$".format(id_)
                )  # Boundaries

                # ---------------------------------------------------- Clustered ----------------------------------------------------  # noqa: 501
                ax[1][c_].plot(px_c, py_c, marker="o", c=col_2, ms=4, ls="", label=r"$p_{{{}}}$".format(id_))  # noqa: 501
                ax[1][c_].plot(wx_c, wy_c, marker="v", c=col_2, ms=4, ls="", label=r"$w_{{{}}}$".format(id_))  # noqa: 501
                ax[1][c_].plot(dx_c, dy_c, marker="x", c=col_2, ms=4, ls="", label=r"$d_{{{}}}$".format(id_))  # noqa: 501
                ax[1][c_].plot(lx_c, ly_c, marker="^", c=col_2, ms=4, ls="", label=r"$l_{{{}}}$".format(id_))  # noqa: 501
                ax[1][c_].arrow(
                    arx_c,
                    ary_c,
                    ax_c,
                    ay_c,
                    fc=col_2,
                    ec=col_2,
                    head_width=20,
                    head_length=20,
                    ls="-",
                    label=r"$p_{{{}}}$".format(id_)
                )  # Orientation

                # ----------------------------------------------------- Smoothed -----------------------------------------------------  # noqa: 501
                ax[2][c_].plot(px_s, py_s, marker="o", c=col_3, ms=4, ls="", label=r"$p_{{{}}}$".format(id_))  # noqa: 501
                ax[2][c_].plot(wx_s, wy_s, marker="v", c=col_3, ms=4, ls="", label=r"$w_{{{}}}$".format(id_))  # noqa: 501
                ax[2][c_].plot(dx_s, dy_s, marker="x", c=col_3, ms=4, ls="", label=r"$d_{{{}}}$".format(id_))  # noqa: 501
                ax[2][c_].plot(lx_s, ly_s, marker="^", c=col_3, ms=4, ls="", label=r"$l_{{{}}}$".format(id_))  # noqa: 501
                ax[2][c_].arrow(
                    arx_s,
                    ary_s,
                    ax_s,
                    ay_s,
                    fc=col_3,
                    ec=col_3,
                    head_width=20,
                    head_length=20,
                    ls="-",
                    label=r"$p_{{{}}}$".format(id_)
                )  # Orientation
            except Exception:
                pass

    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            ax[i][j].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
            ax[i][j].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
            ax[i][j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
            ax[i][j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
            ax[i][j].tick_params(axis="x", labelrotation=90)
            ax[i][j].set_title(
                r"({}) {} - Time $t={}$".format(
                    chr(65 + j + 2 * i),
                    titles[i],
                    times[j]
                ),
                loc="left",
                y=1.005
            )
            ax[i][j].set_xlim([0, y_bounds[1] - y_bounds[0]])
            ax[i][j].set_ylim([x_bounds[1] - x_bounds[0], 0])

    # Combine legends into one and drop duplicates
    handles_0, labels_0 = ax[0][0].get_legend_handles_labels()
    handles_1, labels_1 = ax[0][1].get_legend_handles_labels()
    handles_2, labels_2 = ax[0][2].get_legend_handles_labels()
    handles_3, labels_3 = ax[1][0].get_legend_handles_labels()
    handles_4, labels_4 = ax[1][1].get_legend_handles_labels()
    handles_5, labels_5 = ax[1][2].get_legend_handles_labels()
    handles_6, labels_6 = ax[2][0].get_legend_handles_labels()
    handles_7, labels_7 = ax[2][1].get_legend_handles_labels()
    handles_8, labels_8 = ax[2][2].get_legend_handles_labels()

    unique = dict(zip(
        labels_0 + labels_1 + labels_2 + labels_3 + labels_4 + labels_5 + labels_6 + labels_7 + labels_8,  # noqa: 501
        handles_0 + handles_1 + handles_2 + handles_3 + handles_4 + handles_5 + handles_6 + handles_7 + handles_8  # noqa: 501
    ))

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        fancybox=fancy_legend,
        shadow=False,
        ncol=1,
        borderaxespad=0.01,
        bbox_to_anchor=(x_legend, y_legend),
        bbox_transform=fig.transFigure
    )

    plt.show()

    return fig
