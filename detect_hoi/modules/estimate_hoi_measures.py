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

from functools import partial
from hoi.metrics import Oinfo, InfoTopo  # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)


# Estime O-information for a given period (interval of frames) ----
def estimate_oinfo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the O-information over many multiplets.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with the information of tracked regions

    Returns:
    --------
    df_oinfo : pd.DataFrame
        A DataFrame containing the estimated O-information for different
        multiplets. The DataFrame includes the following columns:
            - "video": Video name.
            - "t_range": The timestamp at which the O-information was
            estimated.
            - "size": The window size at which the O-information was estimated.
            - "multiplet": The identifier for each possible multiplet.
            - "oinfo_distance": The estimated O-information over distances from
            center.
            - "oinfo_orientation": The estimated O-information over angles.
            - "dO_info_distance": The estimated dynamic O-information over
            distances from center.
            - "dO_info_orientation": The estimated dynamic O-information over
            angles.
    """

    # Get data of multiplets
    t_min = df["time"].min()
    t_max = df["time"].max()
    video = df["video"].unique()[0]
    dt = t_max - t_min
    t = str(t_min) + " - " + str(t_max)

    # Get data for O-information
    x1 = df[df["permuted_id"] == 0]["n_x"].values
    x2 = df[df["permuted_id"] == 1]["n_x"].values
    x3 = df[df["permuted_id"] == 2]["n_x"].values
    y1 = df[df["permuted_id"] == 0]["n_y"].values
    y2 = df[df["permuted_id"] == 1]["n_y"].values
    y3 = df[df["permuted_id"] == 2]["n_y"].values
    o1 = df[df["permuted_id"] == 0]["corrected_orientation"].values
    o2 = df[df["permuted_id"] == 1]["corrected_orientation"].values
    o3 = df[df["permuted_id"] == 2]["corrected_orientation"].values

    # Estimate distances time series
    d1 = np.power(np.power(x1, 2) + np.power(y1, 2), 1 / 2)
    d2 = np.power(np.power(x2, 2) + np.power(y2, 2), 1 / 2)
    d3 = np.power(np.power(x3, 2) + np.power(y3, 2), 1 / 2)

    if int(video[0]) == 3:
        # Generate data for estimate O-information of triplet interaction
        data_1 = np.column_stack((d1, d2, d3))
        data_2 = np.column_stack((o1, o2, o3))

        # Define O-information model from HoI library classes
        model_1 = Oinfo(data_1, verbose=0)
        model_2 = Oinfo(data_2, verbose=0)
        model_3 = InfoTopo(data_1, verbose=0)
        model_4 = InfoTopo(data_2, verbose=0)

        # Compute HoI metric using the Gaussian Copula entropy
        hoi_value_1 = model_1.fit(minsize=3, maxsize=3, method="gc")
        hoi_value_2 = model_2.fit(minsize=3, maxsize=3, method="gc")
        hoi_value_3 = model_3.fit(minsize=3, maxsize=3, method="gc")
        hoi_value_4 = model_4.fit(minsize=3, maxsize=3, method="gc")

        # Append results
        df_oinfo = pd.DataFrame({
            "video": [video],
            "t_range": [t],
            "size": [dt],
            "multiplet": ["012"],
            "oinfo_distance": hoi_value_1[0],
            "oinfo_orientation": hoi_value_2[0],
            "dO_info_distance": hoi_value_3[0],
            "dO_info_orientation": hoi_value_4[0]
        })

    if int(video[0]) == 4:
        # Get data for O-information (additional ID)
        x4 = df[df["permuted_id"] == 3]["n_x"].values
        y4 = df[df["permuted_id"] == 3]["n_y"].values
        o4 = df[df["permuted_id"] == 3]["n_orientation"].values

        # Estimate distances time series (additional ID)
        d4 = np.power(np.power(x4, 2) + np.power(y4, 2), 1 / 2)

        # Generate data for estimate O-information of triplet interaction
        data_1 = np.column_stack((d1, d2, d3, d4))
        data_2 = np.column_stack((o1, o2, o3, o4))

        # Define O-information model from HoI library classes
        model_1 = Oinfo(data_1, verbose=0)
        model_2 = Oinfo(data_2, verbose=0)
        model_3 = InfoTopo(data_1, verbose=0)
        model_4 = InfoTopo(data_2, verbose=0)

        # Compute HoI metric using the Gaussian Copula entropy
        hoi_value_1 = model_1.fit(minsize=3, maxsize=4, method="gc")
        hoi_value_2 = model_2.fit(minsize=3, maxsize=4, method="gc")
        hoi_value_3 = model_3.fit(minsize=3, maxsize=4, method="gc")
        hoi_value_4 = model_4.fit(minsize=3, maxsize=4, method="gc")

        # Append results
        df_oinfo = pd.DataFrame({
            "video": [video, video, video, video, video],
            "t_range": [t, t, t, t, t],
            "size": [dt, dt, dt, dt, dt],
            "multiplet": ["012", "013", "023", "123", "0123"],
            "oinfo_distance": hoi_value_1.reshape(1, 5)[0],
            "oinfo_orientation": hoi_value_2.reshape(1, 5)[0],
            "dO_info_distance": hoi_value_3.reshape(1, 5)[0],
            "dO_info_orientation": hoi_value_4.reshape(1, 5)[0]
        })

    return df_oinfo


# Estime O-information for a given period (interval of frames) ----
def estimate_oinfo_gliding_window(
    df: pd.DataFrame,
    log_path: str,
    log_filename: str,
    verbose: int,
    arg_list: list
) -> pd.DataFrame:
    """
    Estimate the O-information over many multiplets.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with the information of tracked regions
    log_path : str
        Directory path where log files will be saved.
    log_filename : str
        Name of the log file to store output messages.
    verbose : int
        Verbosity level. If >=1, progress messages and errors will be logged.
    arg_list : list
        - arg_list[0]: Window size for the evaluation of O-information.

    Returns:
    --------
    df_oinfo : pd.DataFrame
        A DataFrame containing the estimated O-information for different
        multiplets. The DataFrame includes the following columns:
            - "video": Video name.
            - "t_range": The timestamp at which the O-information was
            estimated.
            - "size": The window size at which the O-information was estimated.
            - "multiplet": The identifier for each possible multiplet.
            - "oinfo_distance": The estimated O-information over distances from
            center.
            - "oinfo_orientation": The estimated O-information over angles.
            - "dO_info_distance": The estimated dynamic O-information over
            distances from center.
            - "dO_info_orientation": The estimated dynamic O-information over
            angles.
    """

    # Extract parameters from arg_list (Window size)
    ws = arg_list[0]
    t_min = df["time"].min()
    t_max = df["time"].max()
    windows = (t_max - t_min) // ws
    video = df["video"].unique()[0]

    # Get O-information
    df_oinfo = []
    for w in range(windows):
        t0 = ws * w
        tf = ws * (w + 1)
        mask = (df["time"].between(t0, tf))
        df_oinfo.append(estimate_oinfo(df=df[mask]))

        # Function development (Logging if verbosity is enabled)
        if verbose >= 1:
            with open(f"{log_path}/{log_filename}.txt", "a") as file:
                file.write(
                    "HoI metrics for video: {} - window: {} - s:{}\n".format(
                        video,
                        w,
                        ws
                    )
                )

    # Final merge and relocation of new columns
    df_oinfo = pd.concat(df_oinfo, ignore_index=True)

    return df_oinfo


# Estime O-information for a given period (interval of frames) ----
def estimate_oinfo_multiple_windows(
    df: pd.DataFrame,
    window_sizes: list,
    log_path: str = "../logs",
    log_filename: str = "log_hoi",
    verbose: int = 1,
    tqdm_bar: bool = True
) -> pd.DataFrame:
    """
    Estimate the O-information over many multiplets and windows.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with the information of tracked regions
    window_sizes : list
        Window sizes for estimating O-information
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
    df_oinfo : pd.DataFrame
        A DataFrame containing the estimated O-information for different
        multiplets. The DataFrame includes the following columns:
            - "video": Video name.
            - "t_range": The timestamp at which the O-information was
            estimated.
            - "size": The window size at which the O-information was estimated.
            - "multiplet": The identifier for each possible multiplet.
            - "oinfo_distance": The estimated O-information over distances from
            center.
            - "oinfo_orientation": The estimated O-information over angles.
            - "dO_info_distance": The estimated dynamic O-information over
            distances from center.
            - "dO_info_orientation": The estimated dynamic O-information over
            angles.
    """

    # Auxiliary function for simulations of Polydisperse mixtures paths
    fun_local = partial(
        estimate_oinfo_gliding_window,
        df,
        log_path,
        log_filename,
        verbose
    )

    # Parallel loop for simulations of IbM paths
    df_oinfo = mf.parallel_run(
        fun=fun_local,
        arg_list=window_sizes,
        tqdm_bar=tqdm_bar
    )
    df_oinfo = pd.concat(df_oinfo)

    return df_oinfo
