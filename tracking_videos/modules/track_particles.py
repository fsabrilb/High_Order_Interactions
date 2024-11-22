# -*- coding: utf-8 -*-
"""
Created on Thursday October 2nd 2024

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import misc_functions as mf

from skimage import feature  # type: ignore
from skimage import measure  # type: ignore
from skimage import exposure  # type: ignore
from functools import partial
from skimage.color import rgb2gray  # type: ignore
from skimage.morphology import disk, opening, remove_small_holes, remove_small_objects  # type: ignore # noqa: 501

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)


# Profiling frames with histogram equalization ----
def profile_frame(
    frame: np.ndarray,
    type: str = "equalized",
    clip_limit: float = None
) -> np.ndarray:
    """Apply histogram equalization over a frame, i.e, spreads out the most
    frequent intensity values. For details review:
    https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html

    Args:
    ---------------------------------------------------------------------------
    frame : numpy array dtype float
        Frame representation in numpy array where each pixel has an array of
        size 3 for RGB representation or size 2 for grey scale representation
    type : str
        Type of equalization used (global or local equalization). Default value
        is "equalized"
    clip_limit : float
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast). Defalut value is None

    Returns:
    ---------------------------------------------------------------------------
    adjusted_frame : numpy array dtype float
        Adjusted frame after equalization
    """  # noqa: 501

    adjusted_frame = frame
    if type == "equalized":
        adjusted_frame = exposure.equalize_hist(adjusted_frame)
    if type == "local equalized":
        adjusted_frame = exposure.equalize_adapthist(
            adjusted_frame,
            clip_limit=clip_limit
        )

    return adjusted_frame


# Get contours and edges of a frame using a threshold and Gaussian filter ----
def get_boundary_frames(
    frame: np.ndarray,
    threshold: float = 0.2,
    sigma: float = 2
):
    """Get contours and edges of a frame using Marching square and Canny Edge
    detection algorithm

    Args:
    ---------------------------------------------------------------------------
    frame : numpy array dtype float
        Frame representation in numpy array where each pixel has an array of
        size 3 for RGB representation or size 2 for grey scale representation
    threshold : float
        Filter value for gray scale, normalized between 0 and 1 (lower values
        give darkest levels). Default value is 0.2
    sigma : float
        Standard deviation of the Gaussian filter. Defalut value is 2

    Returns:
    ---------------------------------------------------------------------------
    contours : numpy array dtype float
        Contours detected using Marching squares
    edges : numpy array dtype bool
        Edges detected with Canny algorithm in bool array of the same size as
        the initial frame
    """

    # Convert to grayscale (if not already grayscale)
    if len(frame.shape) == 3:
        gray_frame = rgb2gray(frame)
    else:
        gray_frame = frame

    # Contours - Marching squares
    contours = measure.find_contours(gray_frame, threshold)

    # Edges - Canny Edge detection
    edges = feature.canny(gray_frame, sigma=sigma)

    return contours, edges


# Process frame applying tracking algorithm ----
def process_frame(
    reinforce_boundaries: bool,
    remove_holes: bool,
    type: str,
    clip_limit: float,
    threshold: float,
    sigma: float,
    x_bounds: list,
    y_bounds: list,
    region_area_min: float,
    axis_major_min: float,
    eccentricity_max: float,
    tracking_list
) -> pd.DataFrame:
    """Get the tracked particles profiling the regions of the frame with:
        - time = tracking_list[0]
        - frame = tracking_list[1]

    Args:
    ---------------------------------------------------------------------------
    frame : numpy array dtype float
        Frame representation in numpy array where each pixel has an array of
        size 3 for RGB representation or size 2 for grey scale representation
    time : int
        Index position of the frames detected in reader
    reinforce_boundaries : bool
        Reinforce boundaries using contours ann edges detected in the frame.
        Default value is True
    remove_holes : bool
        Remove holes and small objects for increasing performance in tracking
        algorithm
    type : str
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
    x_bounds : list
        Bound in X-axis for the frame. Remember the axes are rotated with
        imageio. Default value is [300, 1460]
    y_bounds : list
        Bound in Y-axis for the frame. Remember the axes are rotated with
        imageio. Default value is [240, 900]
    region_area_min : float
        Minimal area for the detection of regions. Default value is 200
    axis_major_min : float
        Minimum axis major length when a region is approximated as a ellipse.
        Default value is 94
    eccentricity_max : float
        Maximum eccentricity value allowed. Default value is 0.99

    Returns:
    ---------------------------------------------------------------------------
    tracking_results : pandas DataFrame
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
    """

    # Extract the current frame
    time = tracking_list[0]
    frame = tracking_list[1]

    # First profile then filter (PF step)
    gray_frame = profile_frame(
        frame=rgb2gray(frame),
        type=type,
        clip_limit=clip_limit
    )[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]

    # Filter data after PF step
    adjusted_frame = frame[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1]]

    # Remove holes and small objects in original frame
    if remove_holes:
        mask = ~remove_small_objects(remove_small_holes(gray_frame < threshold, 5000), 5000)  # noqa: 501
        mask = opening(mask, disk(1))
        gray_frame = np.where(mask == 1, gray_frame, np.min(gray_frame.flatten()))  # noqa: 501

    # Reinforce boundaries in original frame
    if reinforce_boundaries:
        # Contours (Marching squares) and Edges (Canny detection)
        contours_0, edges_0 = get_boundary_frames(
            frame=adjusted_frame,
            threshold=threshold,
            sigma=sigma
        )
        contours_1, edges_1 = get_boundary_frames(
            frame=gray_frame,
            threshold=threshold,
            sigma=sigma
        )
        del edges_1

        # Valid contours
        invalid_contours_0 = []
        for contour in contours_0:
            mask = (
                (contour[:, 0] >= y_bounds[0]) &
                (contour[:, 0] <= y_bounds[1]) &
                (contour[:, 1] >= x_bounds[0]) &
                (contour[:, 1] <= x_bounds[1])
            )
            contour = contour[mask]
            if len(contour) > 0:
                invalid_contours_0.append(contour)
        invalid_contours_1 = []
        for contour in contours_1:
            mask = (
                (contour[:, 0] >= y_bounds[0]) &
                (contour[:, 0] <= y_bounds[1]) &
                (contour[:, 1] >= x_bounds[0]) &
                (contour[:, 1] <= x_bounds[1])
            )
            contour = contour[mask]
            if len(contour) > 0:
                invalid_contours_1.append(contour)

        # Reinforce boundaries values
        min_grey_scale = np.min(gray_frame.flatten())
        for contour in invalid_contours_0:
            contour = np.round(contour).astype(int)
            contour[:, 0] = np.clip(contour[:, 0], 0, frame.shape[0] - 1)
            contour[:, 1] = np.clip(contour[:, 1], 0, frame.shape[1] - 1)
            gray_frame[contour[:, 0], contour[:, 1]] = min_grey_scale
        for contour in invalid_contours_1:
            contour = np.round(contour).astype(int)
            contour[:, 0] = np.clip(contour[:, 0], 0, frame.shape[0] - 1)
            contour[:, 1] = np.clip(contour[:, 1], 0, frame.shape[1] - 1)
            gray_frame[contour[:, 0], contour[:, 1]] = min_grey_scale
        gray_frame[edges_0] = min_grey_scale

    # Use skimage to measure the orientation of each detected region
    labeled_frame = measure.label(gray_frame < threshold)
    regions = measure.regionprops(
        labeled_frame,
        intensity_image=rgb2gray(adjusted_frame)
    )

    # ------------------------- Tracking position -------------------------
    # Only consider regions with:
    #   - region_area_min: Minimum enough area (to exclude noise)
    #   - axis_major_min: Minimum axis major length (Particle elliptical size)
    #   - eccentricity_max: Maximum eccentricity (to exclude lines shapes)

    tracking_results = []
    for region in regions:
        # Darkest intensity
        try:
            min_intensity = np.min(region.intensity_image)
            min_intensity_coord = np.argmin(region.intensity_image)
            min_intensity_coord = region.coords[min_intensity_coord]

            max_intensity = np.max(region.intensity_image)
            max_intensity_coord = np.argmax(region.intensity_image)
            max_intensity_coord = region.coords[max_intensity_coord]
        except Exception:
            min_intensity = None
            min_intensity_coord = region.centroid

            max_intensity = None
            max_intensity_coord = region.centroid

        # Final dataframe
        tracking_results.append({
            "id": region.label,                         # Particle ID
            "time": time,                               # Time (Frame)
            "position_x": region.centroid[1],           # Centroid[1] is x
            "position_y": region.centroid[0],           # Centroid[0] is y
            "weighted_x": region.centroid_weighted[1],  # Weighted X
            "weighted_y": region.centroid_weighted[0],  # Weighted Y
            "darkest_v": min_intensity,                 # Darkest value
            "darkest_x": min_intensity_coord[1],        # Darkest X
            "darkest_y": min_intensity_coord[0],        # Darkest Y
            "lightest_v": max_intensity,                # Lightest value
            "lightest_x": max_intensity_coord[1],       # Lightest X
            "lightest_y": max_intensity_coord[0],       # Lightest Y
            "coords_x": region.coords[:, 1],            # Boundary in X
            "coords_y": region.coords[:, 0],            # Boundary in Y
            "orientation": region.orientation,          # Orientation
            "area": region.area,                        # Region area
            "area_convex": region.area_convex,          # Convex area
            "area_filled": region.area_filled,          # Filled area
            "axis_major": region.axis_major_length,     # Major axis
            "axis_minor": region.axis_minor_length,     # Minor axis
            "eccentricity": region.eccentricity,        # Eccentricity
            "euler_number": region.euler_number         # Euler feature
        })

    tracking_results = pd.DataFrame(tracking_results)

    # Final filtering based on empirical results
    mask_final = (
        # (tracking_results["euler_number"] <= 3) &             # Works in some cases # noqa: 501
        # (tracking_results["euler_number"] >= -2) &            # Works in some cases # noqa: 501
        (tracking_results["area"] >= region_area_min) &         # Minimal particle area # noqa: 501
        (tracking_results["axis_major"] >= axis_major_min) &    # Minimal particle size # noqa: 501
        (tracking_results["eccentricity"] <= eccentricity_max)  # Avoid line likewise boundaries # noqa: 501
    )
    tracking_results = tracking_results[mask_final]
    tracking_results["id"] = np.arange(tracking_results.shape[0])

    return tracking_results


# Process frame applying tracking algorithm at multiple frames ----
def process_multiple_frames(
    reader,
    times: list,
    reinforce_boundaries: bool = True,
    remove_holes: bool = False,
    type: str = "equalized",
    clip_limit: float = 0.2,
    threshold: float = 0.2,
    sigma: float = 3.5,
    x_bounds: list = [300, 1460],
    y_bounds: list = [240, 900],
    region_area_min: float = 200,
    axis_major_min: float = 94,
    eccentricity_max: float = 0.99,
    tqdm_bar: bool = True
) -> pd.DataFrame:
    """Get the tracked particles profiling the regions of each frame

    Args:
    ---------------------------------------------------------------------------
    reader : imagaeio object
        Imageio array with all the frames extracted from the video
    times : list dtype int
        Index positions of the frames detected in reader
    reinforce_boundaries : bool
        Reinforce boundaries using contours ann edges detected in the frame.
        Default value is True
    remove_holes : bool
        Remove holes and small objects for increasing performance in tracking
        algorithm
    type : str
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
    x_bounds : list
        Bound in X-axis for the frame. Remember the axes are rotated with
        imageio. Default value is [300, 1460]
    y_bounds : list
        Bound in Y-axis for the frame. Remember the axes are rotated with
        imageio. Default value is [240, 900]
    region_area_min : float
        Minimal area for the detection of regions. Default value is 200
    axis_major_min : float
        Minimum axis major length when a region is approximated as a ellipse.
        Default value is 94
    eccentricity_max : float
        Maximum eccentricity value allowed. Default value is 0.99
    tqdm_bar : bool
        Progress bar in parallel run (default value is True)

    Returns:
    ---------------------------------------------------------------------------
    tracking_results : pandas DataFrame
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
    """

    # Frames extraction (Parallel loop generation)
    tracking_list = []
    for time in times:
        tracking_list.append([time, reader.get_data(time)])

    # Auxiliary function for tracking video
    fun_local = partial(
        process_frame,
        reinforce_boundaries,
        remove_holes,
        type,
        clip_limit,
        threshold,
        sigma,
        x_bounds,
        y_bounds,
        region_area_min,
        axis_major_min,
        eccentricity_max
    )

    # Parallel loop for tracking video
    tracking_results = mf.parallel_run(
        fun=fun_local,
        arg_list=tracking_list,
        tqdm_bar=tqdm_bar
    )
    tracking_results = pd.concat(
        tracking_results,
        ignore_index=True
    ).sort_values(["time", "id"])

    tracking_results["id_count"] = tracking_results.groupby(["time"])["id"].transform("count")  # noqa: 501

    return tracking_results
