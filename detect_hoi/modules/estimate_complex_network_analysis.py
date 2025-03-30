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
import networkx as nx  # type: ignore
import misc_functions as mf
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mtick  # type: ignore

from ts2vg import NaturalVG  # type: ignore
from matplotlib import rcParams  # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)


# Estime Generalized Hurst exponent ----
def estimate_generalized_hurst_and_complexity(
    ts: np.ndarray,
    q: int,
    dx: int,
    taux: int,
    log_path: str = "../logs",
    log_filename: str = "log_hurst_global",
    verbose: int = 1
) -> pd.DataFrame:
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
    log_path : str
        Directory path where log files will be saved (default is "../logs").
    log_filename : str
        Name of the log file to store output messages (default is
        "log_hurst_global").
    verbose : int
        Verbosity level. If >=1, progress messages and errors will be logged
        (default is 1).

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the estimated generalized Hurst exponent given a
        q-th moment. The DataFrame includes the following columns:
            - "q_order": The q-th moment.
            - "hurst": The estimated Hurst exponent.

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
    >>> H = genhurst(np.random.rand(10000), 3)
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
        perm_ent, complex = ordpy.complexity_entropy(ts, dx=dx, taux=taux)

        # Final dataframe with regressions
        H = np.mean(hurst_exponents) / q
        df_hurst = pd.DataFrame({
            "q_order": [q],
            "hurst": [H],
            "permutation_entropy": [perm_ent],
            "statistical_complexity": [complex]
        })

        # Function development (Logging if verbosity is enabled)
        if verbose >= 1:
            with open(f"{log_path}/{log_filename}.txt", "a") as file:
                file.write("Estimated Hurst {} for q = {}\n".format(H, q))

    except Exception as e:
        # Handle errors by assigning zero as a fallback value
        df_hurst = pd.DataFrame({
            "q_order": [q],
            "hurst": [0],
            "permutation_entropy": [0],
            "statistical_complexity": [0]
        })

        # Function development (Logging if verbosity is enabled)
        if verbose >= 1:
            with open(f"{log_path}/{log_filename}.txt", "a") as file:
                file.write("Non-estimated Hurst for q = {}\n".format(q))
                file.write("Error: {}\n".format(e))

    return df_hurst


# Estime Hurst exponent over multiple periods and distance pairs ----
def estimate_hurst_and_complexity_df(
    df: pd.DataFrame,
    filter_step: int,
    dx: int,
    taux: int,
    log_path: str,
    log_filename: str,
    verbose: int,
    arg_list: list
):
    """
    Estimate the the generalized Hurst exponent $H(q)$ from the scaling of the
    renormalized q-moments of the distribution over multiple id_pairs.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with the information of tracked regions
    filter_step : int
        Number of steps between consecutive times, i.e., the number of skipped
        steps
    dx : int
        Embedding dimension (horizontal axis)
    taux : int
        Embedding delay (horizontal axis)
    log_path : str
        Directory path where log files will be saved.
    log_filename : str
        Name of the log file to store output messages.
    verbose : int
        Verbosity level. If >=1, progress messages and errors will be logged.
    arg_list : list
        - arg_list[0]: Exponent order (q-moment) used in the H(q).
        - arg_list[1]: Lower threshold for the filtration of time series data.
        - arg_list[2]: Upper threshold for the filtration of time series data.

    Returns:
    --------
    df_hurst : pd.DataFrame
        A DataFrame containing the estimated Hurst exponent for different
        q-th moment and times. The DataFrame includes the following columns:
            - "id_pair": The identifier for each distance pair.
            - "time": The timestamp at which the Hurst exponent was estimated.
            - "q_order": The q-th moment.
            - "hurst": The estimated Hurst exponent.
    """

    # Extract parameters from arg_list
    q_moment = arg_list[0]
    t_range_0 = arg_list[1]
    t_range_1 = arg_list[2]

    # Construct time series of distances between particles
    mask = (df["time"].between(t_range_0, t_range_1))
    df_distances = mf.estimate_distances(df=df[mask], filter_step=filter_step)
    pairs = df_distances["id_pair"].unique()

    # Hurst analysis
    df_hurst = []
    for id_pair in pairs:
        mask = (df_distances["id_pair"] == id_pair)
        times = df_distances[mask]["time"].unique()

        for t in times:
            # Hurst exponent from distance data
            mask_ = ((mask) & (df_distances["time"] <= t))

            df_local = estimate_generalized_hurst_and_complexity(
                ts=df_distances[mask_]["distance"].values,
                q=q_moment,
                dx=dx,
                taux=taux,
                log_path=log_path,
                log_filename=log_filename,
                verbose=verbose
            )
            df_local["id_pair"] = id_pair
            df_local["time"] = t

            # Append results
            df_hurst.append(df_local)

            # Function development (Logging if verbosity is enabled)
            if verbose >= 1:
                with open(f"{log_path}/{log_filename}.txt", "a") as file:
                    file.write("Estimated H for Pair: {}, Time: {}\n".format(
                        id_pair,
                        t
                    ))

    # Final merge and relocation of new columns
    df_hurst = pd.concat(df_hurst).reset_index(drop=True)
    df_hurst["t_range"] = str(t_range_0) + "_" + str(t_range_1)
    df_hurst.insert(0, "id_pair", df_hurst.pop("id_pair"))
    df_hurst.insert(1, "t_range", df_hurst.pop("t_range"))

    return df_hurst


# Estimate Visibility Graph ----
def estimate_vg(
    df: pd.DataFrame,
    filter_step: int = None,
    width: float = 12,
    height: float = 28,
    n_x_breaks: int = 20,
    n_y_breaks: int = 20,
    fancy_legend: bool = False,
    usetex: bool = False,
    t_range: list = [0, 100]
) -> list:
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

    Args:
    ---------------------------------------------------------------------------
    df : pd.DataFrame
        Dataframe with the information of tracked regions
    filter_step : int
        Number of steps between consecutive times, i.e., the number of skipped
        steps
    width : float
        Width of final plot (default value 12)
    height : float
        Height of final plot (default value 28)
    n_x_breaks : int
        Number of divisions in x-axis (default value 10)
    n_y_breaks : int
        Number of divisions in y-axis (default value 10)
    fancy_legend : bool
        Fancy legend output (default value False)
    usetex : bool
        Use LaTeX for renderized plots (default value False)
    t_range : list
        Lower ([0]) and upper ([1]) threshold for the filtration of time series
        data points range

    Returns:
    ---------------------------------------------------------------------------
    df_all : pd.DataFrame
        Dataframe with the summary of global complex network measures
    df_all_nodes : pd.DataFrame
        Dataframe with the summary of complex network measures over each node
    fig : fig
        Figure with the time series, visibility graph, and degree distribution
    ax : ax
        Axes with the time series, visibility graph, and degree distribution
    """

    # Construct time series of distances between particles
    mask = (df["time"].between(t_range[0], t_range[1]))
    df_distances = mf.estimate_distances(df=df[mask], filter_step=filter_step)
    pairs = df_distances["id_pair"].unique()

    # Initialize Plot Graph
    rcParams.update({
        "font.family": "serif",
        "text.usetex": usetex,
        "pgf.rcfonts": False,
        "text.latex.preamble": r"\usepackage{amsfonts}"
    })

    fig, ax = plt.subplots(len(pairs), 4, squeeze=False)
    fig.set_size_inches(w=width, h=height)

    graph_plot_options = {
        "with_labels": False,
        "node_size": 2.5,
        "node_color": [(0, 0, 0, 1)],
        "edge_color": [(0, 0, 0, 0.15)]
    }

    # Network analysis over pairs
    df_all = []
    df_all_nodes = []
    for j, id_pair in enumerate(pairs):
        mask = (df_distances["id_pair"] == id_pair)
        times = df_distances[mask]["time"].values
        distances = df_distances[mask]["distance"].values
        graph = NaturalVG(directed=None).build(distances, times)
        nxg = graph.as_networkx()  # From ts2vg
        df, df_local = mf.summarize_complex_network(nxg=nxg)

        # Add local information
        df["id_pair"] = id_pair
        df_local["id_pair"] = id_pair

        df["t_range"] = str(t_range[0]) + "_" + str(t_range[1])
        df_local["t_range"] = str(t_range[0]) + "_" + str(t_range[1])

        df_all.append(df)
        df_all_nodes.append(df_local)

        # Add local plot
        label = "$d_{{{}}}(t)$".format(id_pair)
        degrees = df_local["degree"].values

        nx.draw_networkx(
            nxg,
            ax=ax[j, 1],
            pos=graph.node_positions(),
            **graph_plot_options
        )
        nx.draw_networkx(
            nxg, ax=ax[j, 2],
            pos=nx.kamada_kawai_layout(nxg),
            **graph_plot_options
        )

        ax[j, 0].plot(times, distances, label=label, ls="solid", lw=3)
        ax[j, 1].plot(times, distances, label=label, ls="solid", lw=3)
        ax[j, 3].hist(
            degrees,
            label=label,
            alpha=0.19,
            facecolor="blue",
            edgecolor="darkblue",
            density=False,
            histtype="stepfilled",
            cumulative=False
        )

        # Other plot settings
        ax[j, 0].set_ylabel("$d(t)$")
        ax[j, 0].set_xlabel("Time (t)", fontsize=16)
        ax[j, 0].set_title("Distance time series")
        ax[j, 1].set_ylabel("$d(t)$")
        ax[j, 1].set_xlabel("Time (t)", fontsize=16)
        ax[j, 1].set_title("Distance and Visibility Graph")
        ax[j, 2].set_title("Visibility Graph (VG)")
        ax[j, 3].set_title("VG - Degrees distribution")

        for k in [0, 1, 3]:
            ax[j, k].legend(fancybox=fancy_legend, fontsize=14)
            ax[j, k].tick_params(which="major", direction="in", top=True, right=True, labelsize=11, length=12)  # noqa: 501
            ax[j, k].tick_params(which="minor", direction="in", top=True, right=True, labelsize=11, length=6)  # noqa: 501
            ax[j, k].xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
            ax[j, k].xaxis.set_minor_locator(mtick.MaxNLocator(4 * n_x_breaks))
            ax[j, k].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
            ax[j, k].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
            ax[j, k].tick_params(axis="x", labelrotation=90)

    plt.tight_layout()
    plt.close()

    # Final merge and relocation of new columns
    df_all = pd.concat(df_all, ignore_index=True)
    df_all_nodes = pd.concat(df_all_nodes, ignore_index=True)

    df_all.insert(0, "id_pair", df_all.pop("id_pair"))
    df_all_nodes.insert(0, "id_pair", df_all_nodes.pop("id_pair"))
    df_all.insert(1, "t_range", df_all.pop("t_range"))
    df_all_nodes.insert(1, "t_range", df_all_nodes.pop("t_range"))

    return df_all, df_all_nodes, fig, ax
