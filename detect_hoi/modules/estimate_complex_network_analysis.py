# -*- coding: utf-8 -*-
"""
Created on Friday March 6th 2025

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import misc_functions as mf

from ts2vg import NaturalVG  # type: ignore
from functools import partial

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)


# Estimate Visibility Graph over multiple periods and distance pairs ----
def estimate_gliding_vg(
    df: pd.DataFrame,
    log_path: str,
    log_filename: str,
    verbose: int,
    arg_list: list
) -> pd.DataFrame:
    """Estimate visibility graph and its properties for the time series of
    distances between two individuals. The data has the following columns:
        - id: Particle ID
        - permuted_id: Particle ID after smoothing process
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
        - corrected_orientation: Orientation after smoothing process
        - area: Region area i.e. number of pixels of the region scaled by
        pixel-area
        - axis_major: The length of the major axis of the ellipse that has the
        same normalized second central moments as the region
        - axis_minor: The length of the minor axis of the ellipse that has the
        same normalized second central moments as the region
        - eccentricity: Eccentricity of the ellipse that has the same
        second-moments as the region. The eccentricity is the ratio of the
        focal distance (distance between focal points) over the major axis
        length. The value is in the interval [0, 1). When it is 0, the ellipse
        becomes a circle

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with the information of tracked regions.
    log_path : str
        Directory path where log files will be saved.
    log_filename : str
        Name of the log file to store output messages.
    verbose : int
        Verbosity level. If >=1, progress messages and errors will be logged.
    arg_list : list
        - arg_list[0]: Window size for the evaluation of complex network
        measures.

    Returns:
    --------
    df_all : pd.DataFrame
        Dataframe of distances and orientation with the summary of global
        complex network measures
    df_nodes : pd.DataFrame
        Dataframe of distances and orientation with the summary of complex
        network measures over each node
    """

    # Extract parameters from arg_list (Window size)
    ws = arg_list[0]
    t_min = df["time"].min()
    t_max = df["time"].max()
    windows = (t_max - t_min) // ws
    video = df["video"].unique()[0]

    # Network analysis over pairs
    df_all_d = []
    df_all_o = []
    df_all_nodes_d = []
    df_all_nodes_o = []
    for w in range(windows):
        t0 = ws * w
        tf = ws * (w + 1)
        mask = (df["time"].between(t0, tf))
        for idx in df[mask]["permuted_id"].unique():
            # Get data for complexity measures
            df1 = df[((mask) & (df["permuted_id"] == idx))]
            t_min = df1["time"].min()
            t_max = df1["time"].max()
            dt = t_max - t_min
            t = str(t_min) + " - " + str(t_max)

            x = df1[df1["permuted_id"] == idx]["position_x"].values
            y = df1[df1["permuted_id"] == idx]["position_y"].values
            o = df1[df1["permuted_id"] == idx]["corrected_orientation"].values

            # Estimate distances time series
            d = np.power(np.power(x, 2) + np.power(y, 2), 1 / 2)

            # Generate Natural Visibility Graph
            graph_d = NaturalVG(directed=None).build(d, df1["time"].values)
            graph_o = NaturalVG(directed=None).build(o, df1["time"].values)
            nxgd = graph_d.as_networkx()  # From ts2vg
            nxgo = graph_o.as_networkx()  # From ts2vg
            df_d, df_local_d = mf.summarize_complex_network(nxg=nxgd)
            df_o, df_local_o = mf.summarize_complex_network(nxg=nxgo)

            # Add local information
            dicc_v = {
                "video": video,
                "t_range": t,
                "size": dt,
                "permuted_id": idx
            }
            for k, v in dicc_v.items():
                df_d[k] = v
                df_o[k] = v
                df_local_d[k] = v
                df_local_o[k] = v

            df_d["type"] = "distance"
            df_o["type"] = "orientation"
            df_local_d["type"] = "distance"
            df_local_o["type"] = "orientation"

            df_all_d.append(df_d)
            df_all_o.append(df_o)
            df_all_nodes_d.append(df_local_d)
            df_all_nodes_o.append(df_local_o)

        # Function development (Logging if verbosity is enabled)
        if verbose >= 1:
            with open(f"{log_path}/{log_filename}.txt", "a") as file:
                file.write(
                    "Network metrics video: {} - window: {} - s:{}\n".format(
                        video,
                        w,
                        ws
                    )
                )

    # Final merge and relocation of new columns
    df_all_d = pd.concat(df_all_d, ignore_index=True)
    df_all_o = pd.concat(df_all_o, ignore_index=True)
    df_all_nodes_d = pd.concat(df_all_nodes_d, ignore_index=True)
    df_all_nodes_o = pd.concat(df_all_nodes_o, ignore_index=True)

    df_all = pd.concat([df_all_d, df_all_o], ignore_index=True)
    df_nodes = pd.concat([df_all_nodes_d, df_all_nodes_o], ignore_index=True)

    cols = ["type", "video", "t_range", "size", "permuted_id"]
    for i, v in enumerate(cols):
        df_all.insert(i, v, df_all.pop(v))
        df_nodes.insert(i, v, df_nodes.pop(v))

    return df_all, df_nodes


# Estime Visibility Graph for a given period (interval of frames) ----
def estimate_multiple_vg(
    df: pd.DataFrame,
    window_sizes: list,
    log_path: str = "../logs",
    log_filename: str = "log_network",
    verbose: int = 1,
    tqdm_bar: bool = True
) -> pd.DataFrame:
    """
    Estimate the Visibility Graph over many multiplets and windows.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with the information of tracked regions
    window_sizes : list
        Window sizes for estimating Visibility Graph.
    log_path : str
        Directory path where log files will be saved.
    log_filename : str
        Name of the log file to store output messages.
    verbose : int
        Verbosity level. If >=1, progress messages and errors will be logged.
    tqdm_bar : bool
        Progress bar in parallel run (default value is True)

    Returns:
    --------
    df_all : pd.DataFrame
        Dataframe of distances and orientation with the summary of global
        complex network measures
    df_nodes : pd.DataFrame
        Dataframe of distances and orientation with the summary of complex
        network measures over each node
    """

    # Auxiliary function for parallel running
    fun_local = partial(
        estimate_gliding_vg,
        df,
        log_path,
        log_filename,
        verbose
    )

    # Parallel loop for complexity metrics
    data = mf.parallel_run(
        fun=fun_local,
        arg_list=window_sizes,
        tqdm_bar=tqdm_bar
    )
    data = zip(*data)
    df_all, df_nodes = [pd.concat(group, ignore_index=True) for group in data]

    return df_all, df_nodes
