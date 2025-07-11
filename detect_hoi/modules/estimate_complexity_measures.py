# -*- coding: utf-8 -*-
"""
Created on Friday March 6th 2025

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import ordpy  # type: ignore
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import misc_functions as mf

from functools import partial

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)


# Estime Generalized Hurst exponent ----
def estimate_hurst_complexity(
    ts: np.ndarray,
    q: int,
    dx: int,
    taux: int
) -> list:
    r"""
    Computes the generalized Hurst exponent $H(q)$ from the scaling of the
    renormalized q-moments of the distribution. The function estimates H(q)
    based on the relation:

    .. math::

        \frac{\langle|x(t+r) - x(t)|^{q}\rangle}{\langle|x(t)|^{q}\rangle} &
            ~ r^{qH(q)}

    Also, the normalized permutation entropy $S_{P}$ and the statistical
    complexity $C_{JS}$ are estimated as

    .. math::

        S_{P}  &= \sum_{\pi\in\mathcal{S}_{N}}p_{\pi}\log{(p_{\pi})}\\
        C_{JS} &= S_{P}D_{JS}

    where $\mathcal{S}_{N}$ is the Symmetric group of size $N$, $p_{\pi}$ is
    the permutation probabilities, and $D_{JS}$ is the disequilibrium or
    Jensen–Shannon (JS) divergence between $p_{\pi}$ and uniform distribution.

    Parameters:
    -----------
    ts : np.ndarray
        One-dimensional time series data (length greater than 300 is
        recommended).
    q : int
        The q-th moment order for the Hurst exponent calculation.
    dx : int
        Embedding dimension (horizontal axis).
    taux : int
        Embedding delay (horizontal axis).

    Returns:
    --------
    list
        A list containing the estimated generalized Hurst exponent given a
        q-th moment. Also, the permutation entropy and statistical complexity.

    Raises:
    -------
    Warning:
        If the data series length is below 100, a warning is issued.

    References:
    -----------
    - T. Di Matteo et al. Physica A 324 (2003) 183-188
    - T. Di Matteo et al. Journal of Banking & Finance 29 (2005) 827-851
    - T. Di Matteo Quantitative Finance, 7 (2007) 21-36
    - C. Bandt et. al. Physical Review Letter 88 (2002) 174102

    Example:
    --------
    >>> import numpy as np
    """

    try:
        if len(ts) < 300:
            warnings.warn("Data series is very short.")

        hurst_exponents = []
        for t_max in range(5, 20):
            x_vals = np.arange(1, t_max + 1)
            mcord = np.zeros((t_max, 1))

            for tt in range(1, t_max + 1):
                indices = np.arange(tt, len(ts), tt)
                dV = ts[indices] - ts[indices - tt]
                VV = ts[indices - tt]

                N = len(dV) + 1
                X = np.arange(1, N, dtype=np.float64)
                Y = VV

                # Linear regression coefficients
                mx = np.mean(X)
                my = np.mean(Y)
                SSxx = np.sum(X**2) - N * mx**2
                SSxy = np.sum(np.multiply(X, Y)) - N * mx * my
                slope = SSxy / SSxx
                intercept = my - slope * mx

                # Residuals
                ddVd = dV - slope
                VVVd = VV - np.multiply(slope, X) - intercept
                mcord[tt - 1] = np.mean(np.abs(ddVd)**q) / np.mean(np.abs(VVVd)**q)  # noqa: 501

            # Log-log regression
            log_x = np.log10(x_vals)
            log_mcord = np.log10(mcord)
            mx_log = np.mean(log_x)
            my_log = np.mean(log_mcord)
            SSxx_log = np.sum(log_x**2) - t_max * mx_log**2
            SSxy_log = np.sum(np.multiply(log_x, np.transpose(log_mcord)))
            SSxy_log -= t_max * mx_log * my_log

            hurst_exponents.append(SSxy_log / SSxx_log)

        # Estimate normalized permutation entropy and statistical complexity
        perm_ent, complexity = ordpy.complexity_entropy(ts, dx=dx, taux=taux)

        # Final dataframe with regressions
        H = np.mean(hurst_exponents) / q

    except Exception:
        # Handle errors by assigning zero as a fallback value
        H, perm_ent, complexity = 0, 0, 0

    return H, perm_ent, complexity


# Estime Hurst exponent over multiple periods and distance pairs ----
def estimate_gliding_hurst_complexity(
    df: pd.DataFrame,
    q: int,
    dx: int,
    taux: int,
    log_path: str,
    log_filename: str,
    verbose: int,
    arg_list: list
) -> pd.DataFrame:
    """
    Estimate the generalized Hurst exponent $H(q)$, permutation entropy $S_{P}$
    and the statistical complexity $C_{JS}$ over multiple IDs.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with the information of tracked regions
    q : int
        The q-th moment order for the Hurst exponent estimation.
    dx : int
        Embedding dimension (horizontal axis)
    taux : int
        Embedding delay (vertical axis)
    log_path : str
        Directory path where log files will be saved.
    log_filename : str
        Name of the log file to store output messages.
    verbose : int
        Verbosity level. If >=1, progress messages and errors will be logged.
    arg_list : list
        - arg_list[0]: Window size for the evaluation of $H(q)$, $S_{P}$ and
        $C_{JS}$.

    Returns:
    --------
    df_complexity : pd.DataFrame
        A DataFrame containing the estimated $H(q)$, $S_{P}$ and $C_{JS}$. The
        DataFrame includes the following columns:
            - "video": Video name.
            - "t_range": The timestamp at which the O-information was
            estimated.
            - "size": The window size at which the O-information was estimated.
            - "permuted_id": The identifier for each distance ID.
            - "H": The estimated Hurst exponent.
            - "PE": The estimated permutation entropy.
            - "C": The estimated statistical complexity.
    """

    # Extract parameters from arg_list (Window size)
    ws = arg_list[0]
    t_min = df["time"].min()
    t_max = df["time"].max()
    windows = (t_max - t_min) // ws
    video = df["video"].unique()[0]

    # Get Hurst and Complexity metrics
    df_complexity = []
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

            Hd, Ed, Sd = estimate_hurst_complexity(ts=d, q=q, dx=dx, taux=taux)
            Ho, Eo, So = estimate_hurst_complexity(ts=o, q=q, dx=dx, taux=taux)

            # Append results
            df_local = pd.DataFrame({
                "video": [video],
                "t_range": [t],
                "size": [dt],
                "permuted_id": [idx],
                "H_distance": [Hd],
                "PE_distance": [Ed],
                "C_distance": [Sd],
                "H_orientation": [Ho],
                "PE_orientation": [Eo],
                "C_orientation": [So]
            })

            df_complexity.append(df_local)

        # Function development (Logging if verbosity is enabled)
        if verbose >= 1:
            with open(f"{log_path}/{log_filename}.txt", "a") as file:
                file.write(
                    "Complex metrics video: {} - window: {} - s:{}\n".format(
                        video,
                        w,
                        ws
                    )
                )

    # Final merge and relocation of new columns
    df_complexity = pd.concat(df_complexity, ignore_index=True)

    return df_complexity


# Estime O-information for a given period (interval of frames) ----
def estimate_multiple_hurst_complexity(
    df: pd.DataFrame,
    window_sizes: list,
    q: int = 1,
    dx: int = 1,
    taux: int = 1,
    log_path: str = "../logs",
    log_filename: str = "log_complexity",
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
    q : int
        The q-th moment order for the Hurst exponent estimation.
    dx : int
        Embedding dimension (horizontal axis)
    taux : int
        Embedding delay (vertical axis)
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
    df_complexity : pd.DataFrame
        A DataFrame containing the estimated $H(q)$, $S_{P}$ and $C_{JS}$. The
        DataFrame includes the following columns:
            - "video": Video name.
            - "t_range": The timestamp at which the O-information was
            estimated.
            - "size": The window size at which the O-information was estimated.
            - "permuted_id": The identifier for each distance ID.
            - "H": The estimated Hurst exponent.
            - "PE": The estimated permutation entropy.
            - "C": The estimated statistical complexity.
    """

    # Auxiliary function for parallel running
    fun_local = partial(
        estimate_gliding_hurst_complexity,
        df,
        q,
        dx,
        taux,
        log_path,
        log_filename,
        verbose
    )

    # Parallel loop for complexity metrics
    df_complexity = mf.parallel_run(
        fun=fun_local,
        arg_list=window_sizes,
        tqdm_bar=tqdm_bar
    )
    df_complexity = pd.concat(df_complexity)

    return df_complexity
