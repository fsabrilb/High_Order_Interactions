# -*- coding: utf-8 -*-
"""
Created on Friday October 4th 2024

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import os
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import track_particles as tp  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mtick  # type: ignore
import matplotlib.animation as animation  # type: ignore

from skimage.color import rgb2gray  # type: ignore


# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
# plt.rcParams['animation.ffmpeg_path'] = r"C:\ffmpeg\bin\ffmpeg.exe"


# Plot frame without additional processes ----
def plot_normal_frame(
    reader,
    time: int,
    width: float = 10,
    width_ratio: int = 1,
    n_x_breaks: int = 20,
    n_y_breaks: int = 20,
    x_bounds: list = [300, 1460],
    y_bounds: list = [240, 900],
    x_zoom: list = [760, 880],
    y_zoom: list = [320, 520]
):
    """Plot a particular frame for video analysis without additional processes

    Args:
    ---------------------------------------------------------------------------
    reader : imagaeio object
        Imageio array with all the frames extracted from the video
    time : int
        Index position of the frames detected in reader
    width : int
        Width of final plot. Default value 10
    width_ratio : int
        Aspect ratio between the frame plot and histogram of pixels. Default
        value is 1
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

    Returns:
    ---------------------------------------------------------------------------
    None
    """

    frame = reader.get_data(time)
    title = "Original"
    if x_zoom is not None:
        x_bounds = x_zoom
        title = "Zoom [{},{}]".format(x_zoom[0], x_zoom[1])
        if y_zoom is not None:
            y_bounds = y_zoom
            title = "Zoom [{},{}]$\\times$[{},{}]".format(
                x_zoom[0],
                x_zoom[1],
                y_zoom[0],
                y_zoom[1]
            )

    frame = frame[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]
    fig, ax = plt.subplots(1, 2, width_ratios=[width_ratio, 1])
    fig.set_size_inches(
        w=2 * width,
        h=(frame.shape[0] * width / frame.shape[1])
    )
    ax[0].imshow(frame, cmap="gray")
    ax[1].hist(
        frame[frame > 0].ravel(),
        bins=256,
        range=[0, 256],
        density=True,
        color="darkgreen"
    )
    titles = [title, "Grey Scale Histogram"]
    for j in [0, 1]:
        ax[j].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
        ax[j].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
        ax[j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        ax[j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
        ax[j].tick_params(axis="x", labelrotation=90)
        ax[j].set_title(
            r"({}) {} - Time $t={}$".format(chr(65 + j), titles[j], time),
            loc="left",
            y=1.005
        )

    ax_ = ax[1].twinx()
    ax_.hist(
        frame[frame > 0].ravel(),
        bins=256,
        range=[0, 256],
        density=True,
        cumulative=True,
        color="red",
        histtype="step"
    )
    ax_.tick_params(axis="y", labelcolor="red")

    plt.show()


# Plot frame with profiled process ----
def plot_profiled_frame(
    reader,
    time: int,
    width: int = 10,
    width_ratio: int = 1,
    type: str = "equalized",
    clip_limit: float = None,
    n_x_breaks: int = 20,
    n_y_breaks: int = 20,
    x_bounds: list = [300, 1460],
    y_bounds: list = [240, 900],
    x_zoom: list = [760, 880],
    y_zoom: list = [320, 520]
):
    """Plot a particular frame for video analysis with profiled process

    Args:
    ---------------------------------------------------------------------------
    reader : imagaeio object
        Imageio array with all the frames extracted from the video
    time : int
        Index position of the frames detected in reader
    width : int
        Width of final plot. Default value 10
    width_ratio : int
        Aspect ratio between the frame plot and histogram of pixels. Default
        value is 1
    type : int
        Type of equalization used (global or local equalization). Default value
        is "equalized"
    clip_limit : float
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast). Defalut value is None
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

    Returns:
    ---------------------------------------------------------------------------
    None
    """

    frame = reader.get_data(time)
    title = type.capitalize()
    if x_zoom is not None:
        x_bounds = x_zoom
        title = "{} - Zoom [{},{}]".format(title, x_zoom[0], x_zoom[1])
        if y_zoom is not None:
            y_bounds = y_zoom
            title = "{}$\\times$[{},{}]".format(
                title,
                y_zoom[0],
                y_zoom[1]
            )

    # First profile then filter
    gray_frame_0 = tp.profile_frame(
        frame=rgb2gray(frame),
        type=type,
        clip_limit=clip_limit
    )[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]

    # First filter then profile
    gray_frame_1 = tp.profile_frame(
        frame=rgb2gray(frame[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]),  # noqa: 501
        type=type,
        clip_limit=clip_limit
    )

    fig, ax = plt.subplots(2, 2, width_ratios=[width_ratio, 1])
    fig.set_size_inches(
        w=2 * width,
        h=2 * (frame.shape[0] * width / frame.shape[1])
    )
    ax[0, 0].imshow(gray_frame_0, cmap="gray")
    ax[1, 0].imshow(gray_frame_1, cmap="gray")
    ax[0, 1].hist(
        gray_frame_0[gray_frame_0 > 0].ravel(),
        bins=256,
        range=[0, 1],
        density=True,
        color="darkgreen"
    )
    ax[1, 1].hist(
        gray_frame_1[gray_frame_1 > 0].ravel(),
        bins=256,
        range=[0, 1],
        density=True,
        color="darkgreen"
    )

    titles = [
        ["PF - {}".format(title), "PF - Grey Scale Histogram"],
        ["FP - {}".format(title), "FP - Grey Scale Histogram"]
    ]
    for i in [0, 1]:
        for j in [0, 1]:
            ax[i, j].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
            ax[i, j].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
            ax[i, j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
            ax[i, j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
            ax[i, j].tick_params(axis="x", labelrotation=90)
            ax[i, j].set_title(
                r"({}) {} - Time $t={}$".format(
                    chr(65 + 2 * i + j),
                    titles[i][j],
                    time
                ),
                loc="left",
                y=1.005
            )

    ax_0 = ax[0, 1].twinx()
    ax_1 = ax[1, 1].twinx()
    ax_0.hist(
        gray_frame_0[gray_frame_0 > 0].ravel(),
        bins=256,
        range=[0, 1],
        density=True,
        cumulative=True,
        color="red",
        histtype="step"
    )
    ax_1.hist(
        gray_frame_1[gray_frame_1 > 0].ravel(),
        bins=256,
        range=[0, 1],
        density=True,
        cumulative=True,
        color="red",
        histtype="step"
    )
    ax_0.tick_params(axis="y", labelcolor="red")
    ax_1.tick_params(axis="y", labelcolor="red")

    plt.show()


# Plot frame with profiled process and compared with boundaries ----
def plot_boundary_edge_frame(
    reader,
    time: int,
    width: int = 10,
    width_ratio: int = 1,
    type: str = "equalized",
    clip_limit: float = None,
    threshold: float = 0.2,
    sigma: float = 2,
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
    time : int
        Index position of the frames detected in reader
    width : int
        Width of final plot. Default value 10
    width_ratio : int
        Aspect ratio between the frame plot and histogram of pixels. Default
        value is 1
    type : int
        Type of equalization used (global or local equalization). Default value
        is "equalized"
    clip_limit : float
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast). Defalut value is None
    threshold : float
        Filter value for gray scale, normalized between 0 and 1 (lower values
        give darkest levels). Default value is 0.2
    sigma : float
        Standard deviation of the Gaussian filter. Defalut value is 2
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

    frame = reader.get_data(time)
    title = type.capitalize()
    if x_zoom is not None:
        x_bounds = x_zoom
        title = "{} - Zoom [{},{}]".format(title, x_zoom[0], x_zoom[1])
        if y_zoom is not None:
            y_bounds = y_zoom
            title = "{}$\\times$[{},{}]".format(
                title,
                y_zoom[0],
                y_zoom[1]
            )

    # First profile then filter
    gray_frame_0 = tp.profile_frame(
        frame=rgb2gray(frame),
        type=type,
        clip_limit=clip_limit
    )[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]

    # First filter then profile
    gray_frame_1 = tp.profile_frame(
        frame=rgb2gray(frame[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]),  # noqa: 501
        type=type,
        clip_limit=clip_limit
    )

    # Boundaries and Edges
    contours_0, edges_0 = tp.get_boundary_frames(
        frame=frame[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]],
        threshold=threshold,
        sigma=sigma
    )
    contours_1, edges_1 = tp.get_boundary_frames(
        frame=gray_frame_0,
        threshold=threshold,
        sigma=sigma
    )
    contours_2, edges_2 = tp.get_boundary_frames(
        frame=gray_frame_1,
        threshold=threshold,
        sigma=sigma
    )

    fig, ax = plt.subplots(2, 2, width_ratios=[width_ratio, 1])
    fig.set_size_inches(
        w=2 * width,
        h=2 * (frame.shape[0] * width / frame.shape[1])
    )
    ax[0, 0].imshow(
        frame[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]],
        cmap="gray"
    )
    ax[1, 0].imshow(gray_frame_0, cmap="gray")
    ax[0, 1].imshow(gray_frame_1, cmap="gray")

    # Edges
    ax[0, 0].imshow(edges_0, cmap="Greens")
    ax[1, 0].imshow(edges_1, cmap="Greens")
    ax[0, 1].imshow(edges_2, cmap="Greens")
    ax[1, 1].imshow(
        frame[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]],
        cmap="gray"
    )

    # Valid contours
    invalid_contours_0 = []
    for contour in contours_0:
        mask = (
            (contour[:, 0] >= y_bounds[0]) & (contour[:, 0] <= y_bounds[1]) &
            (contour[:, 1] >= x_bounds[0]) & (contour[:, 1] <= x_bounds[1])
        )
        contour = contour[mask]
        if len(contour) > 0:
            invalid_contours_0.append(contour)
    invalid_contours_1 = []
    for contour in contours_1:
        mask = (
            (contour[:, 0] >= y_bounds[0]) & (contour[:, 0] <= y_bounds[1]) &
            (contour[:, 1] >= x_bounds[0]) & (contour[:, 1] <= x_bounds[1])
        )
        contour = contour[mask]
        if len(contour) > 0:
            invalid_contours_1.append(contour)
    invalid_contours_2 = []
    for contour in contours_2:
        mask = (
            (contour[:, 0] >= y_bounds[0]) & (contour[:, 0] <= y_bounds[1]) &
            (contour[:, 1] >= x_bounds[0]) & (contour[:, 1] <= x_bounds[1])
        )
        contour = contour[mask]
        if len(contour) > 0:
            invalid_contours_2.append(contour)

    # Contours
    for contour in contours_0:
        ax[0, 0].plot(
            contour[:, 1],
            contour[:, 0],
            c="blue",
            lw=1,
            label="valid contour"
        )
    for contour in contours_1:
        ax[1, 0].plot(
            contour[:, 1],
            contour[:, 0],
            c="blue",
            lw=1,
            label="valid contour"
        )
    for contour in contours_2:
        ax[0, 1].plot(
            contour[:, 1],
            contour[:, 0],
            c="blue",
            lw=1,
            label="valid contour"
        )
    for contour in invalid_contours_0:
        ax[0, 0].plot(
            contour[:, 1],
            contour[:, 0],
            c="red",
            lw=2,
            label="invalid contour"
        )
    for contour in invalid_contours_1:
        ax[1, 0].plot(
            contour[:, 1],
            contour[:, 0],
            c="red",
            lw=2,
            label="invalid contour"
        )
    for contour in invalid_contours_2:
        ax[0, 1].plot(
            contour[:, 1],
            contour[:, 0],
            c="red",
            lw=2,
            label="invalid contour"
        )

    titles = [
        ["Original", "FP - {}".format(title)],
        ["PF - {}".format(title), ""]
    ]
    for i in [0, 1]:
        for j in [0, 1]:
            ax[i, j].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
            ax[i, j].xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
            ax[i, j].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
            ax[i, j].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
            ax[i, j].tick_params(axis="x", labelrotation=90)
            ax[i, j].set_title(
                r"({}) {} - Time $t={}$".format(
                    chr(65 + 2 * i + j),
                    titles[i][j],
                    time
                ),
                loc="left",
                y=1.005
            )

    # Combine legends into one and drop duplicates
    handles_0, labels_0 = ax[0, 0].get_legend_handles_labels()
    handles_1, labels_1 = ax[0, 1].get_legend_handles_labels()
    handles_2, labels_2 = ax[1, 0].get_legend_handles_labels()
    handles_3, labels_3 = ax[1, 1].get_legend_handles_labels()
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
        ncol=1,
        bbox_to_anchor=(x_legend, y_legend),
        bbox_transform=fig.transFigure
    )

    plt.subplots_adjust(wspace=0.01)
    plt.show()


# Plot frame with tracked particle ----
def plot_tracking_frame(
    reader,
    df_tracked_frame: pd.DataFrame,
    width: int = 10,
    n_x_breaks: int = 20,
    n_y_breaks: int = 20,
    x_bounds: list = [300, 1460],
    y_bounds: list = [240, 900],
    x_zoom: list = [760, 880],
    y_zoom: list = [320, 520],
    fancy_legend: bool = False,
    x_legend: float = 1,
    y_legend: float = 1,
    save_figure: bool = False,
    output_path: str = "../output_files",
    output_name: str = "tp_proof",
    time: int = 0
):
    """Plot a particular frame for video analysis with profiled process and
    boundaries (edges and contours)

    Args:
    ---------------------------------------------------------------------------
    reader : imagaeio object
        Imageio array with all the frames extracted from the video
    df_tracked_frame : pandas DataFrame
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
    save_figure: bool
        Save tracked frame plot flag (default value False)
    output_path : string
        Local path for outputs. Default value is "../output_files"
    output_name : string
        Name of the output animation. Default value is "tp_proof"
    time : int
        Index position of the frames detected in reader (default value 0)

    Returns:
    ---------------------------------------------------------------------------
    None
    """

    frame = reader.get_data(time)
    title = "Tracked Particles"
    if x_zoom is not None:
        x_bounds = x_zoom
        title = "{} - Zoom [{},{}]".format(title, x_zoom[0], x_zoom[1])
        if y_zoom is not None:
            y_bounds = y_zoom
            title = "{}$\\times$[{},{}]".format(
                title,
                y_zoom[0],
                y_zoom[1]
            )

    frame = frame[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]

    df_ = df_tracked_frame[df_tracked_frame["time"] == time]
    num_colors = df_["id"].unique().shape[0]
    colors = np.linspace(0, 1, num_colors)
    cmap = plt.get_cmap("autumn", num_colors)
    cmap.set_under("black")

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(w=width, h=(frame.shape[0] * width / frame.shape[1]))
    ax.imshow(frame, cmap="gray")

    for id in df_["id"].unique():
        df_aux = df_[df_["id"] == id]
        ax.plot(
            df_aux["position_x"].values,
            df_aux["position_y"].values,
            marker="o",
            c=cmap(colors[id]),
            ms=4,
            ls="",
            label=r"$p_{{{}}}$".format(id)
        )  # Centroid
        ax.plot(
            df_aux["weighted_x"].values,
            df_aux["weighted_y"].values,
            marker="v",
            c=cmap(colors[id]),
            ms=4,
            ls="",
            label=r"$w_{{{}}}$".format(id)
        )  # Centroid weighted
        ax.plot(
            df_aux["darkest_x"].values,
            df_aux["darkest_y"].values,
            marker="x",
            c=cmap(colors[id]),
            ms=4,
            ls="",
            label=r"$d_{{{}}}$".format(id)
        )  # Darkest pixel
        ax.plot(
            df_aux["lightest_x"].values,
            df_aux["lightest_y"].values,
            marker="^",
            c=cmap(colors[id]),
            ms=4,
            ls="",
            label=r"$l_{{{}}}$".format(id)
        )  # Lightest pixel

        length = 90
        ax.arrow(
            x=df_aux["position_x"].values[0],
            y=df_aux["position_y"].values[0],
            dx=length*np.sin(df_aux["orientation"].values)[0],
            dy=length*np.cos(df_aux["orientation"].values)[0],
            fc=cmap(colors[id]),
            ec=cmap(colors[id]),
            head_width=20,
            head_length=20,
            ls="-",
            label=r"$p_{{{}}}$".format(id)
        )  # Orientation
        if "coords_x" in df_aux.columns:
            ax.plot(
                df_aux["coords_x"].values[0],
                df_aux["coords_y"].values[0],
                c=cmap(colors[id]),
                alpha=0.18,
                marker="",
                ls="--",
                label=r"$r_{{{}}}$".format(id)
            )  # Boundaries

    ax.plot(
        [0, y_bounds[1] - y_bounds[0], y_bounds[1] - y_bounds[0], 0, 0],
        [0, 0, x_bounds[1] - x_bounds[0], x_bounds[1] - x_bounds[0], 0],
        c="red"
    )

    ax.xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
    ax.xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
    ax.yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_title(
        r"({}) {} - Time $t={}$".format(chr(65), title, time),
        loc="left",
        y=1.005
    )
    if y_zoom is not None:
        ax.set_xlim([0, y_zoom[1] - y_zoom[0]])
    else:
        ax.set_xlim([0, y_bounds[1] - y_bounds[0]])
    if x_zoom is not None:
        ax.set_ylim([x_zoom[1] - x_zoom[0], 0])
    else:
        ax.set_ylim([x_bounds[1] - x_bounds[0], 0])

    # Combine legends into one and drop duplicates
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

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

    if save_figure:
        # Create the output directory if it doesn't exist
        output_folder = "{}/{}".format(output_path, output_name)
        os.makedirs(output_folder, exist_ok=True)

        # Save figure
        fig.savefig(
            "{}/frame_{}.png".format(output_folder, str(time).zfill(6)),
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            transparent=False,
            pad_inches=0.03,
            dpi=75
        )

        plt.close(fig)


# Generate MP4 animation of the tracking alogrithm ----
def plot_tracking_animation(
    reader,
    df_tracked_frames: pd.DataFrame,
    width: int = 10,
    n_x_breaks: int = 20,
    n_y_breaks: int = 20,
    x_bounds: list = [300, 1460],
    y_bounds: list = [240, 900],
    x_zoom: list = [760, 880],
    y_zoom: list = [320, 520],
    fancy_legend: bool = False,
    x_legend: float = 1,
    y_legend: float = 1,
    interval: int = 500,
    fps: float = 1,
    output_path: str = "../output_files",
    output_name: str = "tp_proof"
):
    """Plot a particular frame for video analysis with profiled process and
    boundaries (edges and contours)

    Args:
    ---------------------------------------------------------------------------
    reader : imagaeio object
        Imageio array with all the frames extracted from the video
    df_tracked_frames : pandas DataFrame
        Dataframe with the information of tracked regions with the following
        columns:
            - id: Particle ID
            - time: Times (Frames)
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
    interval : int
        Movie time between frames measured in milliseconds (ms). Defualt
        value is 500
    fps : float
        Movie frame rate (per second). If not set, the frame rate from the
        animation's frame interval. Default value is 1 (1 frame per second)
    output_path : string
        Local path for outputs. Default value is "../output_files"
    output_name : string
        Name of the output animation. Default value is "tp_proof"

    Returns:
    ---------------------------------------------------------------------------
    None
    """

    title = "Tracked Particles"
    if x_zoom is not None:
        x_bounds = x_zoom
        title = "{} - Zoom [{},{}]".format(title, x_zoom[0], x_zoom[1])
        if y_zoom is not None:
            y_bounds = y_zoom
            title = "{}$\\times$[{},{}]".format(
                title,
                y_zoom[0],
                y_zoom[1]
            )

    frame = reader.get_data(0)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(w=width, h=(frame.shape[0] * width / frame.shape[1]))

    # Update animation
    def update_plot(time):
        ax.cla()  # Clean axis
        tracking_local = df_tracked_frames[df_tracked_frames["time"] == time]

        num_colors = tracking_local["id"].nunique()
        colors = np.linspace(0, 1, num_colors)
        cmap = plt.get_cmap("autumn", num_colors)
        cmap.set_under("black")

        frame = reader.get_data(time)
        frame = frame[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]
        ax.imshow(frame, cmap="gray")

        for id in tracking_local["id"].unique():
            df_aux = tracking_local[tracking_local["id"] == id]
            ax.plot(
                df_aux["position_x"].values,
                df_aux["position_y"].values,
                marker="o",
                c=cmap(colors[id]),
                ms=4,
                ls="",
                label=r"$p_{{{}}}$".format(id)
            )  # Centroid
            ax.plot(
                df_aux["weighted_x"].values,
                df_aux["weighted_y"].values,
                marker="v",
                c=cmap(colors[id]),
                ms=4,
                ls="",
                label=r"$w_{{{}}}$".format(id)
            )  # Centroid weighted
            ax.plot(
                df_aux["darkest_x"].values,
                df_aux["darkest_y"].values,
                marker="x",
                c=cmap(colors[id]),
                ms=4,
                ls="",
                label=r"$d_{{{}}}$".format(id)
            )  # Darkest pixel
            ax.plot(
                df_aux["lightest_x"].values,
                df_aux["lightest_y"].values,
                marker="^",
                c=cmap(colors[id]),
                ms=4,
                ls="",
                label=r"$l_{{{}}}$".format(id)
            )  # Lightest pixel
            length = 90
            ax.arrow(
                x=df_aux["position_x"].values[0],
                y=df_aux["position_y"].values[0],
                dx=length*np.sin(df_aux["orientation"].values)[0],
                dy=length*np.cos(df_aux["orientation"].values)[0],
                fc=cmap(colors[id]),
                ec=cmap(colors[id]),
                head_width=20,
                head_length=20,
                ls="-",
                label=r"$p_{{{}}}$".format(id)
            )  # Orientation
            if "coords_x" in df_aux.columns:
                ax.plot(
                    df_aux["coords_x"].values[0],
                    df_aux["coords_y"].values[0],
                    c=cmap(colors[id]),
                    alpha=0.18,
                    marker="",
                    ls="--",
                    label=r"$r_{{{}}}$".format(id)
                )  # Boundaries

        ax.plot(
            [0, y_bounds[1] - y_bounds[0], y_bounds[1] - y_bounds[0], 0, 0],
            [0, 0, x_bounds[1] - x_bounds[0], x_bounds[1] - x_bounds[0], 0],
            c="red"
        )

        ax.xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
        ax.xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
        ax.yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        ax.yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
        ax.tick_params(axis="x", labelrotation=90)
        ax.legend(
            fancybox=fancy_legend,
            borderaxespad=0.01,
            bbox_to_anchor=(x_legend, y_legend)
        )
        ax.set_title(
            r"({}) {} - Time $t={}$".format(chr(65), title, time),
            loc="left",
            y=1.005
        )

        if y_zoom is not None:
            ax.set_xlim([0, y_zoom[1] - y_zoom[0]])
        else:
            ax.set_xlim([0, y_bounds[1] - y_bounds[0]])
        if x_zoom is not None:
            ax.set_ylim([x_zoom[1] - x_zoom[0], 0])
        else:
            ax.set_ylim([x_bounds[1] - x_bounds[0], 0])

    times = df_tracked_frames["time"].unique()
    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=times,
        interval=interval
    )  # Interval in ms
    ani.save(
        "{}/{}.mp4".format(output_path, output_name),
        writer="ffmpeg",
        fps=fps
    )
