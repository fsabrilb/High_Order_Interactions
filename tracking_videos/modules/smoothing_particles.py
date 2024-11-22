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


# Smooth the evolution reordering of mixed IDs ----
def smooth_evolution(
    df_tracked: pd.DataFrame,
    velocity_threshold: float = 100,
    omega_threshold: float = np.pi / 2,
    order: bool = False
) -> pd.DataFrame:
    """Smooth evolution of tracked frames. The data has the following columns:
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
    velocity_threshold : float
        Maximmum velocity in X-axis or Y-axis allowed between identical IDs
        Default value is 100
    omega_threshold : float
        Maximmum angular velocity between identical IDs. Default value is pi/2
    order : bool
        Order selection for traverse the array

    Returns:
    ---------------------------------------------------------------------------
    df_final : pandas DataFrame
        Updated df_tracked with smoothed evolution
    """
    # Assumption: All the frames has the same ID count
    df_final = df_tracked.copy()
    df_final["permuted_id"] = df_final["id"]

    unique_times = sorted(df_final["time"].unique())
    for i in range(1, len(unique_times), 1):
        if unique_times[i - 1] > -1:
            # Filter current and previous data
            current_time = unique_times[i]
            previous_time = unique_times[i - 1]
            dt = current_time - previous_time

            df_current = df_final[df_final["time"] == current_time]
            df_previous = df_final[df_final["time"] == previous_time]

            mask_x = df_current["mask_x"].values[0]  # False for long jumps
            mask_y = df_current["mask_y"].values[0]  # False for long jumps
            final_mask = (~mask_x) | (~mask_y)

            if previous_time % 100 == 0:
                print(
                    "Previous Time:", previous_time,
                    "Current Time:", current_time,
                    "mask_x:", mask_x,
                    "mask_y:", mask_y,
                    "Final mask:", final_mask
                )

            # Update orientation (False for flips)
            df_final["orientation"] = df_final["orientation"].mask(
                cond=(df_final["mask_orientation"] == False),
                other=-df_final["orientation"]
            )

            # Update mixed IDs
            closest_matches = update_mixed_ids(
                df_tracked_1=df_previous,
                df_tracked_2=df_current,
                order=order
            )

            previous_ids = df_previous["permuted_id"].values
            new_ids = previous_ids[closest_matches]
            df_final.loc[df_final["time"] == current_time, "permuted_id"] = new_ids

            # Update velocities and associated masks
            df_final["velocity_x"] = df_final.groupby(["permuted_id"])["position_x"].diff() / dt  # noqa: 501
            df_final["velocity_y"] = df_final.groupby(["permuted_id"])["position_y"].diff() / dt # noqa: 501
            df_final["velocity_orientation"] = df_final.groupby(["permuted_id"])["orientation"].diff() / dt  # noqa: 501

            df_final["mask_x"] = np.where(np.abs(df_final["velocity_x"]) <= velocity_threshold, True, False)  # noqa: 501
            df_final["mask_y"] = np.where(np.abs(df_final["velocity_y"]) <= velocity_threshold, True, False)  # noqa: 501
            df_final["mask_orientation"] = np.where(np.abs(df_final["velocity_orientation"]) <= omega_threshold, True, False)  # noqa: 501

    df_final = df_final.sort_values(["time", "id"])

    return df_final
