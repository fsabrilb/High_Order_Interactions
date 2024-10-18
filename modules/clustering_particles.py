# -*- coding: utf-8 -*-
"""
Created on Friday October 11th 2024

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import track_particles as tp

from skimage import draw, metrics  # type: ignore
from skimage.color import rgb2gray  # type: ignore
from sklearn.cluster import KMeans  # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)


# Update contours according of each particle (ID) ----
def update_contours(
    reader,
    df_tracked_1: pd.DataFrame,
    df_tracked_2: pd.DataFrame,
    type: str = "equalized",
    clip_limit: float = 0.2,
    similarity_threshold: float = 0.95
) -> pd.DataFrame:
    """Update contours using two tracked frames. In both cases, the data has
    the following columns:
        - id: Particle ID
        - time: Times (Frames)
        - position_x: Centroid position in X-axis
        - position_y: Centroid position in Y-axis
        - weighted_x: Centroid position in X-axis weighted with intensity image
        - weighted_y: Centroid position in Y-axis weighted with intensity image
        - darkest_v: Intensity of darkest pixel in the local region (particle)
        - darkest_x: Position of darkest pixel in the local region (particle)
        - darkest_y: Position of darkest pixel in the local region (particle)
        - coords_x: X values of the region's boundary
        - coords_y: Y values of the region's boundary
        - orientation: Orientation respect rows
        - area: Region area i.e. number of pixels of the region scaled by
        pixel-area
        - area_convex: Area of the convex hull image, which is the smallest
        convex polygon that encloses the region
        - area_filled: Area of the region with all the holes filled in
        - axis_major: The length of the major axis of the ellipse that has the
        same normalized second central moments as the region
        - axis_minor: The length of the minor axis of the ellipse that has the
        same normalized second central moments as the region
        - eccentricity: Eccentricity of the ellipse that has the same
        second-moments as the region. The eccentricity is the ratio of the
        focal distance (distance between focal points) over the major axis
        length. The value is in the interval [0, 1). When it is 0, the ellipse
        becomes a circle
        - euler_number: Euler characteristic of the set of non-zero pixels.
        Computed as number of connected components subtracted by number of
        holes (input.ndim connectivity). In 3D, number of connected components
        plus number of holes subtracted by number of tunnels
    Args:
    ---------------------------------------------------------------------------
    reader : imagaeio object
        Imageio array with all the frames extracted from the video
    df_tracked_1 : pandas DataFrame
        Dataframe with the information of tracked regions previous time
    df_tracked_2 : pandas DataFrame
        Dataframe with the information of tracked regions actual time
    type : str
        Type of equalization used (global or local equalization). Default value
        is "equalized"
    clip_limit : float
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast). Defalut value is None
    similarity_threshold : float
        Structural similarity lower bound between identical IDs contours
        allowed. Default value is 0.95

    Returns:
    ---------------------------------------------------------------------------
    df_tracked_frame_updated : pandas DataFrame
        Dataframe with the updated contours
    """

    # Assumption: Both dataframes has the same number of IDs
    time_1 = df_tracked_1["time"].unique()[0]
    time_2 = df_tracked_2["time"].unique()[0]

    gray_frame_1 = tp.profile_frame(
        frame=rgb2gray(reader.get_data(time_1)),
        type=type,
        clip_limit=clip_limit
    )

    gray_frame_2 = tp.profile_frame(
        frame=rgb2gray(reader.get_data(time_2)),
        type=type,
        clip_limit=clip_limit
    )
    min_ = np.min([gray_frame_1.min(), gray_frame_2.min()])
    max_ = np.max([gray_frame_1.max(), gray_frame_2.max()])

    df_tracked_updated = []
    for id_ in df_tracked_2["id"].unique():
        # Contours components
        df_aux_1 = df_tracked_1[df_tracked_1["id"] == id_]
        df_aux_2 = df_tracked_2[df_tracked_2["id"] == id_]
        coords_1_x = df_aux_1["coords_x"].values[0]
        coords_1_y = df_aux_1["coords_y"].values[0]
        coords_2_x = df_aux_2["coords_x"].values[0]
        coords_2_y = df_aux_2["coords_y"].values[0]

        # Contours construction
        contour_1 = np.column_stack((coords_1_x, coords_1_y))
        contour_2 = np.column_stack((coords_1_x, coords_1_y))

        # Ranges of the areas (Remember the inversion of axes in skimage)
        x_min_1, x_max_1 = coords_1_y.min(), coords_1_y.max()
        x_min_2, x_max_2 = coords_2_y.min(), coords_2_y.max()
        y_min_1, y_max_1 = coords_1_x.min(), coords_1_x.max()
        y_min_2, y_max_2 = coords_2_x.min(), coords_2_x.max()
        rank_x = np.max([x_max_1 - x_min_1, x_max_2 - x_min_2])
        rank_y = np.max([y_max_1 - y_min_1, y_max_2 - y_min_2])

        # Image construction (Contours detection)
        mask_1 = np.zeros_like(gray_frame_1, dtype=bool)
        mask_2 = np.zeros_like(gray_frame_2, dtype=bool)

        rr1, cc1 = draw.polygon(contour_1[:, 1], contour_1[:, 0], mask_1.shape)
        rr2, cc2 = draw.polygon(contour_2[:, 1], contour_2[:, 0], mask_2.shape)
        mask_1[rr1, cc1] = True
        mask_2[rr2, cc2] = True

        # Image construction (Filter outside region)
        img_1 = gray_frame_1
        img_2 = gray_frame_2
        img_1[~mask_1] = 1  # White in outside region
        img_2[~mask_2] = 1  # White in outside region

        img_1 = img_1[x_min_1: (x_min_1 + rank_x), y_min_1: (y_min_1 + rank_y)]
        img_2 = img_2[x_min_2: (x_min_2 + rank_x), y_min_2: (y_min_2 + rank_y)]

        # Now apply structural_similarity
        ssim_score, diff = metrics.structural_similarity(
            img_1,
            img_2,
            data_range=max_ - min_,
            full=True
        )
        del diff
        df_aux_2["ssim"] = ssim_score

        # Update contours if there is no similarity
        if ssim_score < similarity_threshold:
            distance_x = df_aux_2["position_x"].values[0] - df_aux_1["position_x"].values[0]  # noqa: 501
            distance_y = df_aux_2["position_y"].values[0] - df_aux_1["position_y"].values[0]  # noqa: 501
            df_aux_2["coords_x"] = [distance_x + coords_1_x]
            df_aux_2["coords_y"] = [distance_y + coords_1_y]

        df_tracked_updated.append(df_aux_2)

    df_tracked_updated = pd.concat(df_tracked_updated, ignore_index=True).sort_values(["time", "id"])  # noqa: 501

    return df_tracked_updated


# Clustering particles in two consecutive frames using K-means algorithm ----
def clustering_consecutive_frames(
    df_tracked_old_frame: pd.DataFrame,
    df_tracked_frame: pd.DataFrame,
    weight_previous_time: float = 0.5,
    n_particles: int = 2
) -> pd.DataFrame:
    """Apply K-means clustering algorithm using the data of actual time
    (df_tracked_frame) and the previous time (df_tracked_old_frame). In both
    cases, the data has the following columns:
        - id: Particle ID
        - time: Times (Frames)
        - position_x: Centroid position in X-axis
        - position_y: Centroid position in Y-axis
        - weighted_x: Centroid position in X-axis weighted with intensity image
        - weighted_y: Centroid position in Y-axis weighted with intensity image
        - darkest_v: Intensity of darkest pixel in the local region (particle)
        - darkest_x: Position of darkest pixel in the local region (particle)
        - darkest_y: Position of darkest pixel in the local region (particle)
        - coords_x: X values of the region's boundary
        - coords_y: Y values of the region's boundary
        - orientation: Orientation respect rows
        - area: Region area i.e. number of pixels of the region scaled by
        pixel-area
        - area_convex: Area of the convex hull image, which is the smallest
        convex polygon that encloses the region
        - area_filled: Area of the region with all the holes filled in
        - axis_major: The length of the major axis of the ellipse that has the
        same normalized second central moments as the region
        - axis_minor: The length of the minor axis of the ellipse that has the
        same normalized second central moments as the region
        - eccentricity: Eccentricity of the ellipse that has the same
        second-moments as the region. The eccentricity is the ratio of the
        focal distance (distance between focal points) over the major axis
        length. The value is in the interval [0, 1). When it is 0, the ellipse
        becomes a circle
        - euler_number: Euler characteristic of the set of non-zero pixels.
        Computed as number of connected components subtracted by number of
        holes (input.ndim connectivity). In 3D, number of connected components
        plus number of holes subtracted by number of tunnels
    Args:
    ---------------------------------------------------------------------------
    df_tracked_old_frame : pandas DataFrame
        Dataframe with the information of tracked regions previous time
    df_tracked_frame : pandas DataFrame
        Dataframe with the information of tracked regions actual time
    weight_previous_time: float
        Weight given to the previous tracked frame for update the actual
        tracked frame. Normalized between 0 and 1. Default value 0.5
    n_particles : int
        Number of minimal detected particles. Default value 2

    Returns:
    ---------------------------------------------------------------------------
    df_new_tracked : pandas DataFrame
        Dataframe with the updated tracked frame
    """

    # Frames data extraction
    ids_now = df_tracked_frame["id"].max() + 1
    time_now = df_tracked_frame["time"].unique()[0]

    df_new_tracked = df_tracked_frame.copy()
    if (ids_now < n_particles):  # Assumption: ids_old == n_particles
        df_new_tracked = pd.concat([df_tracked_old_frame, df_tracked_frame], ignore_index=True)  # noqa: 501
        df_new_tracked["time"] = time_now

    # K-means for centroids, weighted centroids and darkest position
    initial_centroids = df_tracked_old_frame[["position_x", "position_y"]].values  # noqa: 501
    kmeans = KMeans(n_clusters=n_particles, init=initial_centroids, n_init=1)

    dicc_positions = {"p": "position", "w": "weighted", "d": "darkest"}
    for suffix, column in dicc_positions.items():
        # New columns names
        cluster_ = "cluster_{}".format(suffix)  # Cluster label
        id_ = "id_{}".format(suffix)            # Temporal ID
        column_x = "{}_x".format(column)        # Column value X-axis
        column_y = "{}_y".format(column)        # Column value Y-axis

        # Assign cluster according to the K-means algorithm
        df_new_tracked[cluster_] = kmeans.fit_predict(df_new_tracked[[column_x, column_y]].values)  # noqa:501

        # Update ID and column values with cluster center
        map_id = dict(enumerate(df_tracked_old_frame.iloc[0: n_particles]["id"]))  # noqa:501
        df_new_tracked[id_] = df_new_tracked[cluster_].map(map_id)

        # Update column value as a weighted average of previous and actual time
        map_cluster_center_x = dict(enumerate(kmeans.cluster_centers_[:, 0]))
        map_cluster_center_y = dict(enumerate(kmeans.cluster_centers_[:, 1]))

        if ids_now != n_particles:
            df_new_tracked[column_x] = (
                weight_previous_time * df_new_tracked[cluster_].map(map_cluster_center_x)  # noqa:501
                + (1 - weight_previous_time) * df_new_tracked[column_x]
            )
            df_new_tracked[column_y] = (
                weight_previous_time * df_new_tracked[cluster_].map(map_cluster_center_y)  # noqa:501
                + (1 - weight_previous_time) * df_new_tracked[column_y]
            )

        # Average over mixing IDs (First part - Average per column)
        #   For instance, if ids_old > n_particles, then two points are
        #   associated with the same id_ (pigeonhole principle)
        df_new_tracked[column_x] = df_new_tracked.groupby(id_)[column_x].transform("mean")  # noqa:501
        df_new_tracked[column_y] = df_new_tracked.groupby(id_)[column_y].transform("mean")  # noqa:501

    # Average over mixing IDs (Second part - Average over mixing particles)
    #   For instance (id_p, id_w, id_d) = (1, 1, 0) indicates a mixture of two
    #   detected particles in previous time when they are compared with actual
    #   detected particles (actual darkest pixel of particle_1 is related with
    #   particle_0 of previous time). Ideally id_p = id_w = id_d.
    df_new_tracked["id"] = df_new_tracked["id_p"]  # Priorize centroids
    df_new_tracked["position_x"] = df_new_tracked.groupby("id")["position_x"].transform("mean")  # noqa:501
    df_new_tracked["position_y"] = df_new_tracked.groupby("id")["position_y"].transform("mean")  # noqa:501
    df_new_tracked["weighted_x"] = df_new_tracked.groupby("id")["weighted_x"].transform("mean")  # noqa:501
    df_new_tracked["weighted_y"] = df_new_tracked.groupby("id")["weighted_y"].transform("mean")  # noqa:501
    df_new_tracked["darkest_x"] = df_new_tracked.groupby("id")["darkest_x"].transform("mean")  # noqa:501
    df_new_tracked["darkest_y"] = df_new_tracked.groupby("id")["darkest_y"].transform("mean")  # noqa:501

    # Get final tracked particles with the same number of n_particles
    temp_cols = ["id_p", "id_w", "id_d", "cluster_p", "cluster_w", "cluster_d"]
    df_new_tracked.drop_duplicates(subset=["id_p"], inplace=True)
    df_new_tracked.drop(columns=temp_cols, inplace=True)

    return df_new_tracked


# Clustering particles in all frames using K-means algorithm ----
def clustering_all_frames(
    df_all_tracked: pd.DataFrame,
    weights_previous_time: np.array,
    n_particles: int = 2
) -> pd.DataFrame:
    """Apply K-means clustering algorithm using the data of consecutive frames
    saved in general dataframe df_all_tracked with the following columns:
        - id: Particle ID
        - time: Times (Frames)
        - position_x: Centroid position in X-axis
        - position_y: Centroid position in Y-axis
        - weighted_x: Centroid position in X-axis weighted with intensity image
        - weighted_y: Centroid position in Y-axis weighted with intensity image
        - darkest_v: Intensity of darkest pixel in the local region (particle)
        - darkest_x: Position of darkest pixel in the local region (particle)
        - darkest_y: Position of darkest pixel in the local region (particle)
        - coords_x: X values of the region's boundary
        - coords_y: Y values of the region's boundary
        - orientation: Orientation respect rows
        - area: Region area i.e. number of pixels of the region scaled by
        pixel-area
        - area_convex: Area of the convex hull image, which is the smallest
        convex polygon that encloses the region
        - area_filled: Area of the region with all the holes filled in
        - axis_major: The length of the major axis of the ellipse that has the
        same normalized second central moments as the region
        - axis_minor: The length of the minor axis of the ellipse that has the
        same normalized second central moments as the region
        - eccentricity: Eccentricity of the ellipse that has the same
        second-moments as the region. The eccentricity is the ratio of the
        focal distance (distance between focal points) over the major axis
        length. The value is in the interval [0, 1). When it is 0, the ellipse
        becomes a circle
        - euler_number: Euler characteristic of the set of non-zero pixels.
        Computed as number of connected components subtracted by number of
        holes (input.ndim connectivity). In 3D, number of connected components
        plus number of holes subtracted by number of tunnels
    Args:
    ---------------------------------------------------------------------------
    df_all_tracked : pandas DataFrame
        Dataframe with the information of tracked regions for all times
    weights_previous_time: float
        Weights given to each time. Normalized between 0 and 1 in each time.
    n_particles : int
        Number of minimal detected particles. Default value 2

    Returns:
    ---------------------------------------------------------------------------
    df_clustered : pandas DataFrame
        Dataframe with the updated tracked frame after clustering
    """

    # Frames extraction (Loop generation)
    times = df_all_tracked["time"].unique()
    dt = np.max(np.abs(np.diff(times)))
    df_clustered = df_all_tracked.copy()

    # Review weights for update positions
    if len(weights_previous_time) != len(times):
        weights_previous_time = 0.5 * np.ones_like(times)

    # Clustering consecutive frames
    for k, time in enumerate(times):
        print("- Init time: {}".format(time))
        if k > 0:  # Assumption: Initial frame is adjusted manually
            df_local = df_clustered[((df_clustered["time"] <= time + dt) & (df_clustered["time"] >= time - dt))]  # noqa: 501

            # K-means algorithm between consecutive frames
            df_local_clustered = clustering_consecutive_frames(
                df_tracked_old_frame=df_local[df_local["time"] == times[k-1]],
                df_tracked_frame=df_local[df_local["time"] == times[k]],
                weight_previous_time=weights_previous_time[k],
                n_particles=n_particles
            )

            # Update areas (add flag)

            # Update original data with the clustered data
            df_clustered.drop(df_clustered[df_clustered["time"] == time].index, inplace=True)  # noqa: 501
            df_clustered = pd.concat([df_clustered, df_local_clustered], ignore_index=True)  # noqa: 501
            print("- Clustered for time: {}".format(time))

    df_clustered = df_clustered.sort_values(["time", "id"])

    return df_clustered
