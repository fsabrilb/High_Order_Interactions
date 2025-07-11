# -*- coding: utf-8 -*-
"""
Created on Friday March 6th 2025

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from collections import Counter

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)


# Estimate recurrence plot (matrix) ----
def compute_recurrence_matrix(
    x: np.ndarray,
    y: np.ndarray,
    epsilon_factor: float = 0.05,
    plot: bool = True
) -> np.ndarray:
    """
    Compute and optionally plot the recurrence matrix from 2D trajectory data.
    The recurrence plot is a nonlinear method for studying the dynamics and
    synchronization of complex systems. It visualizes when a system revisits
    the same or similar state in its phase space.

    Parameters:
    -----------
    x : np.ndarray
        One-dimensional time series data.
    y : np.ndarray
        One-dimensional time series data.
    epsilon_factor : float
        Multiplier for (std_x + std_y) to set recurrence threshold, where std_x
        (std_y) is the standard deviation of x (y). A reasonable value is below
        0.10.
    plot : bool
        Whether to plot the recurrence matrix.

    Returns:
    --------
    R : np.ndarray
        Recurrence matrix (binary)
    """
    data = np.vstack([x, y]).T  # shape (n_times, 2)
    std_x, std_y = np.std(x), np.std(y)
    epsilon = epsilon_factor * np.power(std_x**2 + std_y**2, 1 / 2)

    # Compute pairwise distances
    diff = data[:, None, :] - data[None, :, :]  # shape (n, n, 2)
    dist = np.linalg.norm(diff, axis=2)         # shape (n, n)

    R = (dist <= epsilon).astype(int)
    epsilon = np.round(epsilon, 4)

    if plot:
        plt.figure(figsize=(4, 4))
        plt.imshow(R, cmap="binary", origin="lower")
        plt.title("Recurrence Plot ($\\varepsilon$ = {})".format(epsilon))
        plt.xlabel("Time")
        plt.ylabel("Time")
        plt.tight_layout()
        plt.show()

    return R


# Estimate line lengths from recurrence matrix ----
def get_line_lengths(
    matrix: np.ndarray,
    min_length: int = 2,
    axis: int = 0
) -> list:
    """
    Get line lengths greater than min_length in a binary matrix. Select axis=0
    for vertical lines, and axis=1 for diagonal lines.

    Parameters:
    -----------
    matrix : np.ndarray
        Recurrence matrix (binary)
    min_length : int
        Minimum accepted length for the line lengths.
    axis : int
        Selection of vertical or diagonal lines

    Returns:
    --------
    lines : list
        Lengths of the lines greater than the minimum length threshold
    """
    if axis == 1:  # diagonals
        lines = []
        for k in range(-matrix.shape[0] + 1, matrix.shape[1]):
            diag = np.diag(matrix, k)
            lines += list(map(
                len,
                filter(
                    lambda x: len(x) >= min_length,
                    "".join(map(str, diag)).split("0")
                )
            ))
    else:  # vertical lines
        lines = []
        for col in matrix.T:
            lines += list(map(
                len,
                filter(
                    lambda x: len(x) >= min_length,
                    "".join(map(str, col)).split("0")
                )
            ))

    return lines


# Develop cross-recurrence quantification analysis (CRQA) ----
def estimate_crqa_metrics(
    R: np.ndarray,
    min_diagonal_length: int = 2,
    min_vertical_length: int = 2
) -> pd.DataFrame:
    """
    Do the cross-recurrence quantification analysis (CRQA) with some selected
    metrics.

    Parameters:
    -----------
    R : np.ndarray
        Recurrence matrix (binary)
    min_diagonal_length : int
        Minimum accepted length for the diagonal line lengths.
    min_vertical_length : int
        Minimum accepted length for the vertical line lengths.

    Returns:
    --------
    df : pd.DataFrame
        CRQA metrics given by:
            - Recurrence Rate (RR): Ratio of recurrent points in the plot.
            - Determinism (DET): Fraction of recurrent points forming
            diagonal lines (indicating predictability).
            - Average Diagonal Line Length (L): Mean time over which the system
            exhibits similar behavior.
            - Entropy (ENTR): Shannon entropy of diagonal line lengths
            (complexity measure).
            - Laminarity (LAM): Fraction of points forming vertical lines
            (indicating intermittency or stationarity).
            - Trapping Time (TT): Average vertical line length.
            - Number of diagonal lines.
            - Number of vertical lines.
    """
    N = R.shape[0]
    total_points = N * N
    recurrence_points = np.sum(R)

    # Recurrence Rate (RR)
    RR = recurrence_points / total_points

    # Diagonal lines (excluding main diagonal)
    diag_lines = get_line_lengths(
        matrix=R,
        min_length=min_diagonal_length,
        axis=1
    )
    diag_total = sum(diag_lines)

    DET = diag_total / recurrence_points if recurrence_points > 0 else 0
    L = np.mean(diag_lines) if diag_lines else 0
    ENTR = 0
    for c in Counter(diag_lines).values():
        if diag_lines:
            ENTR -= (c / len(diag_lines)) * np.log(c / len(diag_lines))

    # Vertical lines
    vert_lines = get_line_lengths(
        matrix=R,
        min_length=min_vertical_length,
        axis=0
    )
    vert_total = sum(vert_lines)

    LAM = vert_total / recurrence_points if recurrence_points > 0 else 0
    TT = np.mean(vert_lines) if vert_lines else 0

    # Final results
    df = pd.DataFrame({
        "RR": [RR],
        "DET": [DET],
        "L": [L],
        "ENTR": [ENTR],
        "LAM": [LAM],
        "TT": [TT],
        "num_diag_lines": [len(diag_lines)],
        "num_vert_lines": [len(vert_lines)]
    })

    return df


# Estimate cross-recurrence quantification analysis over tracked data ----
def estimate_multiple_crqa(
    df: pd.DataFrame,
    epsilon_factor: float = 0.05,
    plot: bool = True,
    min_diagonal_length: int = 2,
    min_vertical_length: int = 2
) -> list:
    """
    Do the cross-recurrence quantification analysis (CRQA) with the recurrence
    matrix and some selected metrics for a given tracked video dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with the information of tracked regions
    epsilon_factor : float
        Multiplier for (std_x + std_y) to set recurrence threshold, where std_x
        (std_y) is the standard deviation of x (y). A reasonable value is below
        0.10.
    plot : bool
        Whether to plot the recurrence matrix.
    min_diagonal_length : int
        Minimum accepted length for the diagonal line lengths.
    min_vertical_length : int
        Minimum accepted length for the vertical line lengths.

    Returns:
    --------
    R_dict : dict
        Recurrence matrix (binary) for multiple videos
    df_metrics : pd.DataFrame
        CRQA metrics given by:
            - Recurrence Rate (RR): Ratio of recurrent points in the plot.
            - Determinism (DET): Fraction of recurrent points forming
            diagonal lines (indicating predictability).
            - Average Diagonal Line Length (L): Mean time over which the system
            exhibits similar behavior.
            - Entropy (ENTR): Shannon entropy of diagonal line lengths
            (complexity measure).
            - Laminarity (LAM): Fraction of points forming vertical lines
            (indicating intermittency or stationarity).
            - Trapping Time (TT): Average vertical line length.
            - Number of diagonal lines.
            - Number of vertical lines.

    """
    # Selected columns per video
    cols = [
        "video", "id_pair",
        "RR", "DET", "L", "ENTR", "LAM", "TT",
        "num_diag_lines", "num_vert_lines"
    ]

    # Cross-recurrence quantification analysis by video
    R_dict = {}
    df_metrics = []
    for video in df["video"].unique():
        df1 = df[df["video"] == video]
        distances = []
        for idx in df1["permuted_id"].unique():
            df2 = df1[df1["permuted_id"] == idx]
            x, y = df2["n_x"].values, df2["n_y"].values
            distances.append(np.power(np.power(x, 2) + np.power(y, 2), 1 / 2))
        for i, d1 in enumerate(distances):
            for j, d2 in enumerate(distances):
                if i <= j:
                    # Recurrence matrix
                    R = compute_recurrence_matrix(
                        x=d1,
                        y=d2,
                        epsilon_factor=epsilon_factor,
                        plot=plot
                    )
                    R_dict.update({video + "_" + str(i) + str(j): R})

                    # CRQA metrics
                    df_aux = estimate_crqa_metrics(
                        R=R,
                        min_diagonal_length=min_diagonal_length,
                        min_vertical_length=min_vertical_length
                    )
                    df_aux["video"] = video
                    df_aux["id_pair"] = str(i) + str(j)
                    df_metrics.append(df_aux[cols])

    df_metrics = pd.concat(df_metrics, ignore_index=True)
    return R_dict, df_metrics


# Estimate Kuramoto order parameter ----
def estimate_kuramoto_order_parameter(
    df: pd.DataFrame
) -> list:
    """
    Estimate the Kuramoto order parameter. This metric summarizes global
    coherence in the system and is often visualized as a function of time or
    coupling strength.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with the information of tracked regions

    Returns:
    --------
    times : np.ndarray
        Times understand it as the number of frames in the tracked video
    Rt : np.ndarray
        Kuramoto order parameter

    """
    times = df["time"].unique()

    # Pivot table to shape (n_ids, n_times)
    phase_matrix = df.pivot(
        index="permuted_id",
        columns="time",
        values="corrected_orientation"
    ).values
    Rt = np.abs(np.mean(np.exp(1j * phase_matrix), axis=0))  # shape:(n_times,)
    return times, Rt
