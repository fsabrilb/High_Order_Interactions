# -*- coding: utf-8 -*-
"""
Created on Friday October 11th 2024

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import misc_functions as mf

from functools import partial
from scipy.spatial.distance import cdist  # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)


# Permutation of matrix indexes (auxiliar funciton) ----
def make_permutation(arr: np.ndarray):
    n = len(arr)
    used = set()       # Keep track of numbers that are already used
    result = arr.copy()  # Copy of the array to modify
    available = set(range(n))  # Set of all numbers in the permutation

    # Iterate through the array to handle duplicates
    for i in range(n):
        if result[i] in used:
            # Find an unused element to replace the duplicate
            unused_value = min(available - used)
            result[i] = unused_value
            used.add(unused_value)  # Mark as used
        else:
            used.add(result[i])  # Mark the first occurrence as used

    return list(result)


# Permutation of matrix indexes (auxiliar function) ----
def find_unique_minimum_indices(matrix: np.ndarray, order: bool = False):
    """Find permutation of indexes using minimum value per row

    Args:
    ---------------------------------------------------------------------------
    matrix : np.ndarray
        Square matrix with the distances between ids of consecutive frames with
        size NxN
    order : bool
        Order selection for traverse the array
    Returns:
    ---------------------------------------------------------------------------
    permutation : list
        Permutation of the N indexes used to estimate re-ordering in tracked
        positions
    """
    n = matrix.shape[0]
    permutation = [-1] * n  # To store the permutation indices
    used_columns = set()    # To keep track of already assigned columns

    if order:
        # Flatten matrix to find global ordering of minimum values
        flat_indices = np.argsort(matrix, axis=None)

        # Iterate over sorted flattened indices to assign rows based on minimums
        for flat_index in flat_indices:
            # Convert the flattened index back to 2D indices
            row, col = divmod(flat_index, n)

            # If this row hasn't been assigned and column is free, assign it
            if permutation[row] == -1 and col not in used_columns:
                permutation[row] = col
                used_columns.add(col)

            # Stop if all indices are assigned
            if len(used_columns) == n:
                break
    else:
        permutation = matrix.argmin(axis=1)
        if len(np.unique(permutation)) != n:
            permutation = make_permutation(arr=permutation)

    return permutation


# Update ID according to the increases in position (Mixed IDs) ----
def update_mixed_ids(
    df_tracked_1: pd.DataFrame,
    df_tracked_2: pd.DataFrame,
    order: bool = False
) -> np.ndarray:
    """Update mixed IDs using two tracked frames. In both cases, the data has
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
        - lightest_v: Intensity of lightest pixel in the local region(particle)
        - lightest_x: Position of lightest pixel in the local region (particle)
        - lightest_y: Position of lightest pixel in the local region (particle)
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
        - velocity_x: Velocity in X-axis
        - velocity_y: Velocity in Y-axis
        - velocity_orientation: Angular velocity
        - mask_x: Flag for long-jump in x-axis
        - mask_y: Flag for long-jump in y-axis
        - mask_orientation: Flag for flip of the head-bump orientation

    Args:
    ---------------------------------------------------------------------------
    df_tracked_1 : pandas DataFrame
        Dataframe with the information of tracked regions previous time
    df_tracked_2 : pandas DataFrame
        Dataframe with the information of tracked regions actual time
    order : bool
        Order selection for traverse the array

    Returns:
    ---------------------------------------------------------------------------
    closest_matches : numpy array
        List of the updated IDs for the rest of the system evolution
    """

    # Assumption: Both dataframes has the same number of IDs
    ids = df_tracked_1["id"].nunique()
    closest_matches = np.arange(0, ids, 1).tolist()

    # Pairwise distances matrix
    distances = np.zeros([ids, ids])
    for i in range(0, ids):
        for j in range(0, ids):
            r1 = df_tracked_1[df_tracked_1["id"] == i][["position_x", "position_y"]].to_numpy()  # noqa: 501
            r2 = df_tracked_2[df_tracked_2["id"] == j][["position_x", "position_y"]].to_numpy()  # noqa: 501

            try:
                distances[j][i] = cdist(r2, r1, "cityblock")
            except Exception:
                pass

    # Find closest matches for each id at current time
    closest_matches = find_unique_minimum_indices(
        matrix=distances,
        order=order
    )

    return np.array(closest_matches, dtype=int)


# Get allowed frames after reordering of mixed IDs ----
def get_allowed_frame(
    df_tracked: pd.DataFrame,
    order: bool = False,
    log_path="../logs",
    log_filename="log_smooth",
    verbose=1,
    arg_list: list = [0, 100, 100, np.pi / 2, True]
) -> pd.DataFrame:
    """Get allowed tracked frames. The data has the following columns:
        - id: Particle ID
        - time: Times (Frames)
        - position_x: Centroid position in X-axis
        - position_y: Centroid position in Y-axis
        - weighted_x: Centroid position in X-axis weighted with intensity image
        - weighted_y: Centroid position in Y-axis weighted with intensity image
        - darkest_v: Intensity of darkest pixel in the local region (particle)
        - darkest_x: Position of darkest pixel in the local region (particle)
        - darkest_y: Position of darkest pixel in the local region (particle)
        - lightest_v: Intensity of lightest pixel in the local region(particle)
        - lightest_x: Position of lightest pixel in the local region (particle)
        - lightest_y: Position of lightest pixel in the local region (particle)
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
        - velocity_x: Velocity in X-axis
        - velocity_y: Velocity in Y-axis
        - velocity_orientation: Angular velocity
        - mask_x: Flag for long-jump in x-axis
        - mask_y: Flag for long-jump in y-axis
        - mask_orientation: Flag for flip of the head-bump orientation

    Args:
    ---------------------------------------------------------------------------
    df_tracked : pandas DataFrame
        Dataframe with the information of tracked regions
    order : bool
        Order selection for traverse the array
    log_path : string
        Local path for logs (default value is "../logs")
    log_filename : string
        Local filename for logs (default value is "log_smooth")
    verbose : int
        Provides additional details as to what the computer is doing when
        get_allowed_frame is running
    t0 : float
        Initial time for filtering the information
    tf : float
        Final time for filtering the information
    velocity_threshold : float
        Maximmum velocity in X-axis or Y-axis allowed between identical IDs
        Default value is 100
    omega_threshold : float
        Maximmum angular velocity between identical IDs. Default value is pi/2
    smooth : bool
        Flag for smoothing over all frames or only initial and last frame

    Returns:
    ---------------------------------------------------------------------------
    df_final : pandas DataFrame
        Updated df_tracked with allowed frames in column allowed_frame,
        correct swapped IDs in column permuted_id and initial corrected
        orientation
    """
    # Extract the parameters from arg_list
    t0 = arg_list[0]
    tf = arg_list[1]
    velocity_threshold = arg_list[2]
    omega_threshold = arg_list[3]
    smooth = arg_list[4]
    log_filename = log_filename + "_" + str(t0).zfill(5) + "_" + str(tf).zfill(5)  # noqa: 501

    # Assumption: All the frames has the same ID count
    df_final = df_tracked[((df_tracked["time"] >= t0) & (df_tracked["time"] <= tf))].copy()  # noqa: 501
    df_final["allowed_frame"] = True
    df_final["permuted_id"] = df_final["id"]
    df_final["corrected_orientation"] = df_final["orientation"]

    if smooth:
        unique_times = sorted(df_final["time"].unique())
    else:
        unique_times = [t0, tf]
        df_final.loc[~df_final["time"].isin(unique_times), "allowed_frame"] = False  # noqa: 501
    i = 0
    while (i < len(unique_times) - 1):
        if unique_times[i] > -1:
            j = i + 1
            allowed_frame = False
            while (j < len(unique_times)) and (allowed_frame is False):
                # Filter current and previous data
                current_time = unique_times[j]
                previous_time = unique_times[i]

                df_current = df_final[df_final["time"] == current_time]
                df_previous = df_final[df_final["time"] == previous_time]

                # Update mixed IDs
                closest_matches = update_mixed_ids(
                    df_tracked_1=df_previous,
                    df_tracked_2=df_current,
                    order=order
                )

                # Update local velocities
                previous_velocities_x = df_previous["position_x"].values[closest_matches]  # noqa: 501
                previous_velocities_y = df_previous["position_y"].values[closest_matches]  # noqa: 501
                current_velocities_x = df_current["position_x"].values
                current_velocities_y = df_current["position_y"].values

                velocities_x = (current_velocities_x - previous_velocities_x)  # noqa: 501
                velocities_y = (current_velocities_y - previous_velocities_y)  # noqa: 501
                mask_x = np.where(np.abs(velocities_x) <= velocity_threshold, True, False)  # noqa: 501
                mask_y = np.where(np.abs(velocities_y) <= velocity_threshold, True, False)  # noqa: 501

                if smooth is False:
                    mask_x = np.array([True])
                    mask_y = np.array([True])

                # Update dataframe
                if (np.all(mask_x)) and (np.all(mask_y)):  # Correct swaping
                    allowed_frame = True
                    new_ids = df_previous["permuted_id"].values[closest_matches]
                    df_final.loc[df_final["time"] == current_time, "permuted_id"] = new_ids  # noqa: 501

                    # Update orientation (False for flips)
                    previous_omega = df_previous["corrected_orientation"].values[closest_matches]  # noqa: 501
                    current_omega = df_current["corrected_orientation"].values
                    previous_sign = np.sign(previous_omega)
                    current_sign = np.sign(current_omega)
                    omega = np.where(
                        previous_sign == current_sign,
                        current_omega - previous_omega,
                        current_omega + previous_omega
                    )
                    orientation = np.where(
                        current_omega - previous_omega <= omega_threshold,
                        current_omega,
                        -current_omega  # noqa: 501
                    )
                    df_final.loc[df_final["time"] == current_time, "corrected_orientation"] = orientation  # noqa: 501

                    # Update velocities and angular velocity (omega)
                    df_final.loc[df_final["time"] == current_time, "velocity_x"] = velocities_x  # noqa: 501
                    df_final.loc[df_final["time"] == current_time, "velocity_y"] = velocities_y  # noqa: 501
                    df_final.loc[df_final["time"] == current_time, "velocity_orientation"] = omega  # noqa: 501

                    # Function development
                    if verbose >= 1:
                        with open("{}/{}.txt".format(log_path, log_filename), "a") as file:  # noqa: 501
                            file.write(
                                "Smooth interval [{},{}], Previous time: {}, Current time: {}\n".format(  # noqa: 501
                                    t0,
                                    tf,
                                    previous_time,
                                    current_time
                                )
                            )

                else:  # Uncorrect swaping (error in tracking as lightly points)
                    # Function development
                    if verbose >= 1:
                        with open("{}/{}.txt".format(log_path, log_filename), "a") as file:  # noqa: 501
                            file.write(
                                "Nonsmooth interval [{},{}], Previous time: {}, Dropped time: {}\n".format(  # noqa: 501
                                    t0,
                                    tf,
                                    previous_time,
                                    current_time
                                )
                            )
                    df_final.loc[df_final["time"] == current_time, "allowed_frame"] = allowed_frame  # noqa: 501
                    j += 1
            i = j

    # Final processing (drop missed values)
    df_final = df_final.sort_values(["time", "permuted_id"])
    cols_dropped = [
        "position_x", "weighted_x", "darkest_x", "lightest_x",
        "position_y", "weighted_y", "darkest_y", "lightest_y",
        "orientation", "corrected_orientation",
        "velocity_x", "velocity_y", "velocity_orientation"
    ]
    for col_ in cols_dropped:
        df_final.loc[df_final["allowed_frame"] == False, col_] = None
    df_final["smooth_chunk"] = "chunk_{}_{}".format(str(t0).zfill(5), str(tf).zfill(5))  # noqa: 501

    if smooth is False:
        # Linear interpolation over dropped frames
        for id_ in range(len(df_final["permuted_id"].unique())):
            mask = df_final.loc[:,"permuted_id"]==df_final["permuted_id"].unique()[id_]  # noqa: 501
            df_final[mask]=df_final[mask].interpolate(method="linear")

    # Relocate new columns
    df_final.insert(0, "allowed_frame", df_final.pop("allowed_frame"))
    df_final.insert(1, "time", df_final.pop("time"))
    df_final.insert(2, "smooth_chunk", df_final.pop("smooth_chunk"))
    df_final.insert(4, "permuted_id", df_final.pop("permuted_id"))
    df_final.insert(16, "corrected_orientation", df_final.pop("corrected_orientation"))  # noqa: 501

    return df_final


# Deployment of all smoothing process (swapping IDs, orientation, allowed frames) ----  # noqa: 501
def smooth_frames(
    df_tracked: pd.DataFrame,
    arg_list: list = [[0, 100, 100, np.pi / 2, True]],
    order: bool = False,
    log_path="../logs",
    log_filename="log_smooth",
    verbose=1,
    tqdm_bar=True
):
    """Get allowed tracked frames. The data has the following columns:
        - id: Particle ID
        - time: Times (Frames)
        - position_x: Centroid position in X-axis
        - position_y: Centroid position in Y-axis
        - weighted_x: Centroid position in X-axis weighted with intensity image
        - weighted_y: Centroid position in Y-axis weighted with intensity image
        - darkest_v: Intensity of darkest pixel in the local region (particle)
        - darkest_x: Position of darkest pixel in the local region (particle)
        - darkest_y: Position of darkest pixel in the local region (particle)
        - lightest_v: Intensity of lightest pixel in the local region(particle)
        - lightest_x: Position of lightest pixel in the local region (particle)
        - lightest_y: Position of lightest pixel in the local region (particle)
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
        - velocity_x: Velocity in X-axis
        - velocity_y: Velocity in Y-axis
        - velocity_orientation: Angular velocity
        - mask_x: Flag for long-jump in x-axis
        - mask_y: Flag for long-jump in y-axis
        - mask_orientation: Flag for flip of the head-bump orientation

    Args:
    ---------------------------------------------------------------------------
    df_tracked : pandas DataFrame
        Dataframe with the information of tracked regions
    arg_list : int
        Time interval bounds per chunk, velocity threshold per chunk, omega
        threshold per chunk and smooth flag per chunk
    order : bool
        Order selection for traverse the array
    log_path : string
        Local path for logs (default value is "../logs")
    log_filename : string
        Local filename for logs (default value is "log_smooth")
    verbose : int
        Provides additional details as to what the computer is doing when
        get_allowed_frame is running

    Returns:
    ---------------------------------------------------------------------------
    df_final : pandas DataFrame
        Updated df_tracked with allowed frames in column allowed_frame,
        correct swapped IDs in column permuted_id and initial corrected
        orientation
    """

    # Auxiliary function for simulations of Brownian paths
    fun_local = partial(
        get_allowed_frame,
        df_tracked,
        order,
        log_path,
        log_filename,
        verbose
    )

    # Parallel loop for chunks of smoothing process
    df_final = mf.parallel_run(
        fun=fun_local,
        arg_list=arg_list,
        tqdm_bar=tqdm_bar
    )
    df_final = pd.concat(df_final)
    df_final = df_final.sort_values(["time", "smooth_chunk", "permuted_id"])

    # Swapping for merging chunks
    mask = (df_final["allowed_frame"] == True)  # noqa: 501
    unique_times = df_final.groupby("smooth_chunk")["time"].max().values
    allowed_times = df_final[mask].groupby("smooth_chunk")["time"].max().values
    dt = unique_times - allowed_times + 1  # Time difference between consecutive chunks

    def apply_permutation(group, permutation):
        group["permuted_id"] = group["permuted_id"].apply(lambda x: permutation[x])  # noqa: 501
        return group

    for i in range(len(unique_times)):
        df_aux = df_final[df_final["time"] == unique_times[i]]
        chunks = df_aux["smooth_chunk"].unique()

        tau = unique_times[i]
        if df_aux["allowed_frame"].values[0] is False:
            tau = allowed_times[i]
        df_aux = df_final[df_final["time"] == tau]

        if len(chunks) > 1:
            df_1 = df_aux[df_aux["smooth_chunk"] == chunks[0]].copy()
            df_2 = df_final[df_final["time"] == tau + dt[i]].copy()

            df_1["id"] = df_1["permuted_id"]
            df_2["id"] = df_2["permuted_id"]

            # Update mixed IDs
            closest_matches = update_mixed_ids(
                df_tracked_1=df_1,
                df_tracked_2=df_2,
                order=order
            )

            new_ids = df_1["permuted_id"].values[closest_matches]
            # new_ids = df_previous["permuted_id"].values[closest_matches]
            # df_final.loc[df_final["time"] == current_time, "permuted_id"] = new_ids  # noqa: 501
            print(
                "Time:", unique_times[i],
                "Chunks", len(chunks),
                "Allowed time", allowed_times[i],
                "Permutation", new_ids
            )

            tau = unique_times[i]
            mask = ((df_final["time"] == tau) & (df_final["smooth_chunk"].isin(chunks[1:])))  # noqa: 501
            df_final = df_final[~mask]
            if i == len(unique_times) - 1:
                delta = df_final["time"].max()
            else:
                delta = unique_times[i + 1]
            df_final.loc[((df_final["time"] > tau) & (df_final["time"] <= delta)), "permuted_id"] = (  # noqa: 501
                df_final[((df_final["time"] > tau) & (df_final["time"] <= delta))]  # noqa: 501
                .groupby("time")
                .apply(lambda g: apply_permutation(g, new_ids))["permuted_id"]
                .values
            )
        df_final = df_final.sort_values(["time", "smooth_chunk", "permuted_id"])

    # Linear interpolation over dropped frames
    # for id_ in range(len(df_final["permuted_id"].unique())):
    #     mask = df_final.loc[:,"permuted_id"]==df_final["permuted_id"].unique()[id_]  # noqa: 501
    #     # df_final[mask]=df_final[mask].interpolate(method="linear")
    #     df_final[mask]=(
    #         df_final[mask].interpolate(method="nearest", order=3, limit=None, limit_direction=None)
    #         .fillna(df_final[mask].interpolate(method="spline", order=3, limit=None, limit_direction='both'))
    #      )

    return df_final
