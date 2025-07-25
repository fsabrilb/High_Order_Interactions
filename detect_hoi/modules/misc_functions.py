# -*- coding: utf-8 -*-
"""
Created on Friday March 6th 2025

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import networkx as nx  # type: ignore

from tqdm import tqdm  # type: ignore
from multiprocessing import Pool, cpu_count

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)


# Deployment of parallel run in function of arguments list ----
def parallel_run(
    fun,
    arg_list,
    tqdm_bar: bool = False
):
    """Implement parallel run in arbitrary function with input arg_list:

    Args:
    ---------------------------------------------------------------------------
    fun : function
        Function to implement in parallel
    arg_list : list of tuples
        List of arguments to pass in function
    tqdm_bar : bool
        Progress bar (default value is False)

    Returns:
    ---------------------------------------------------------------------------
    m : list of objects
        Function evaluation in all possible combination of tuples
    """

    if tqdm_bar:
        m = []
        with Pool(processes=cpu_count()) as p:
            with tqdm(total=len(arg_list), ncols=60) as pbar:
                for _ in p.imap(fun, arg_list):
                    m.append(_)
                    pbar.update()
            p.terminate()
            p.join()
    else:
        p = Pool(processes=cpu_count())
        m = p.map(fun, arg_list)
        p.terminate()
        p.join()

    return m


# Estimate elementary symmetric polynomials (ESP) ----
def estimate_esp(
    df: pd.DataFrame,
    filter_step: int = None
) -> pd.DataFrame:
    """Estimate elementary symmetric polynomials (ESP) between the time series
    of the individuals. The data has the following columns:
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

    Returns:
    ---------------------------------------------------------------------------
    df_final : pd.DataFrame
        Dataframe with the ESP between the different IDs with the
        following columns:
            - time: Times (frames)
            - order: Order of the elementary symmetric polynomial
            - n_x: Component in X-axis for the estimated ESP
            - n_y: Component in Y-axis for the estimated ESP
            - corrected_orientation: Orientation of the individuals
    """
    ids = df["permuted_id"].unique()
    print("Estimate elementary symmetric polynomials for:", len(ids), "ids")

    time = df[df["permuted_id"] == 0]["time"].values
    x1 = df[df["permuted_id"] == 0]["n_x"].values + 0.5
    y1 = df[df["permuted_id"] == 0]["n_y"].values + 0.5
    o1 = df[df["permuted_id"] == 0]["corrected_orientation"].values
    x2 = df[df["permuted_id"] == 1]["n_x"].values + 0.5
    y2 = df[df["permuted_id"] == 1]["n_y"].values + 0.5
    o2 = df[df["permuted_id"] == 1]["corrected_orientation"].values

    # For 2 individuals
    if len(ids) == 2:
        # Component definition
        e1x = x1 + x2
        e1y = y1 + y2
        e2x = x1 * x2
        e2y = y1 * y2

        # Append data
        df_final = pd.concat([
            pd.DataFrame({"time": time, "order": [0]*len(time), "n_x": e1x, "n_y": e1y, "corrected_orientation": o1}),  # noqa: 501
            pd.DataFrame({"time": time, "order": [1]*len(time), "n_x": e2x, "n_y": e2y, "corrected_orientation": o2})  # noqa: 501
        ])

    # For 3 individuals
    elif len(ids) == 3:
        x3 = df[df["permuted_id"] == 2]["n_x"].values + 0.5
        y3 = df[df["permuted_id"] == 2]["n_y"].values + 0.5
        o3 = df[df["permuted_id"] == 2]["corrected_orientation"].values

        # Component definition
        e1x = x1 + x2 + x3
        e1y = y1 + y2 + y3
        e2x = x1 * x2 + x1 * x3 + x2 * x3
        e2y = y1 * y2 + y1 * y3 + y2 * y3
        e3x = x1 * x2 * x3
        e3y = y1 * y2 * y3

        # Append data
        df_final = pd.concat([
            pd.DataFrame({"time": time, "order": [0]*len(time), "n_x": e1x, "n_y": e1y, "corrected_orientation": o1}),  # noqa: 501
            pd.DataFrame({"time": time, "order": [1]*len(time), "n_x": e2x, "n_y": e2y, "corrected_orientation": o2}),  # noqa: 501
            pd.DataFrame({"time": time, "order": [2]*len(time), "n_x": e3x, "n_y": e3y, "corrected_orientation": o3})  # noqa: 501
        ])

    # For 4 individuals
    else:
        x3 = df[df["permuted_id"] == 2]["n_x"].values + 0.5
        y3 = df[df["permuted_id"] == 2]["n_y"].values + 0.5
        o3 = df[df["permuted_id"] == 2]["corrected_orientation"].values
        x4 = df[df["permuted_id"] == 3]["n_x"].values + 0.5
        y4 = df[df["permuted_id"] == 3]["n_y"].values + 0.5
        o4 = df[df["permuted_id"] == 3]["corrected_orientation"].values

        # Component definition
        e1x = x1 + x2 + x3 + x4
        e1y = y1 + y2 + y3 + y4
        e2x = x1 * x2 + x1 * x3 + x1 * x4 + x2 * x3 + x2 * x4 + x3 * x4
        e2y = y1 * y2 + y1 * y3 + y1 * y4 + y2 * y3 + y2 * y4 + y3 * y4
        e3x = x1 * x2 * x3 + x1 * x2 * x4 + x2 * x3 * x4
        e3y = y1 * y2 * y3 + y1 * y2 * y4 + y2 * y3 * y4
        e4x = x1 * x2 * x3 * x4
        e4y = y1 * y2 * y3 * y4

        # Append data
        df_final = pd.concat([
            pd.DataFrame({"time": time, "order": [0]*len(time), "n_x": e1x, "n_y": e1y, "corrected_orientation": o1}),  # noqa: 501
            pd.DataFrame({"time": time, "order": [1]*len(time), "n_x": e2x, "n_y": e2y, "corrected_orientation": o2}),  # noqa: 501
            pd.DataFrame({"time": time, "order": [2]*len(time), "n_x": e3x, "n_y": e3y, "corrected_orientation": o3}),  # noqa: 501
            pd.DataFrame({"time": time, "order": [3]*len(time), "n_x": e4x, "n_y": e4y, "corrected_orientation": o4})  # noqa: 501
        ])

    if filter_step is not None:
        df_final = df_final[df_final["time"] % filter_step == 0]
        print("-- Skipped data every {} points".format(filter_step))

    return df_final


# Get a summary of network measures for a graph ----
def summarize_complex_network(nxg) -> list:
    r"""Estimate different complex network measures, namely:
        - Triangles (clustering): Compute the number of triangles.
        - Transitivity (clustering): Possible triangles are identified by the
        number of "triads" (two edges with a shared vertex). The transitivity
        is :math:`T = 3\frac{\#triangles}{\#triads}`
        - Average clustering (clustering): For unweighted graphs, the
        clustering of a node :math:`v` is the fraction of possible triangles
        through that node that exist. Thus, The clustering coefficient for the
        graph is the average, :math:`C = \frac{1}{n}\sum_{v \in G} c_v`,
        where :math:`n` is the number of nodes in `G`, and
        :math:`c_v = \frac{2 T(v)}{deg(v)(deg(v)-1)}`, where :math:`T(v)` is
        the number of triangles through node :math:`v` and :math:`deg(v)` is
        the degree of :math:`v`.
        - Degree centrality (centrality): The degree centrality for a node
        :math:`v` is the fraction of nodes it is connected to.
        - Betweenness centrality (centrality): The betweenness centrality of a
        node $v$ is the sum of the fraction of all-pairs shortest paths that
        pass through $v$,
        :math:`c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}`,
        where $V$ is the set of nodes, $\sigma(s, t)$ is the number of shortest
        $(s, t)$-paths,  and $\sigma(s, t|v)$ is the number of those paths
        passing through some  node $v$ other than $s, t$.
        - Average shortest path length (Shortest path): The average shortest
        path length is
        :math:`a =\sum_{\substack{s,t\in V \\ s\neq t}}\frac{d(s, t)}{n(n-1)}`,
        where `V` is the set of nodes in `G`, `d(s, t)` is the shortest path
        from `s` to `t`, and `n` is the number of nodes in `G`.
        - deg_dist: Degree value per node.
        - eccentricity (distance): The eccentricity of a node v is the maximum
        distance from v to all other nodes in G.
        - diameter (distance): The diameter is the maximum eccentricity.
        - radius (distance): The radius is the minimum eccentricity.

    Args:
    ---------------------------------------------------------------------------
    nxg : graph
        Complex network representation of a time series. For instance, the
        visibility graph of time series

    Returns:
    ---------------------------------------------------------------------------
    df : pd.DataFrame
        Dataframe with the summary of global complex network measures
    df_local : pd.DataFrame
        Dataframe with the summary of complex network measures over each node
    """
    # Degree distribution
    deg_dist = dict(nxg.degree())

    # Local measures over nodes
    df_local = pd.DataFrame({
        "node": np.arange(len(deg_dist)),
        "degree": deg_dist.values(),
        "clustering": nx.clustering(nxg).values(),
        "triangles": nx.triangles(nxg).values(),
        "degree_centrality": nx.degree_centrality(nxg).values(),
        "betweenness_centrality": nx.betweenness_centrality(nxg).values(),
        "eccentricity": nx.eccentricity(nxg).values()
    })

    # Global measures
    max_degree = max(deg_dist.values())
    mean_degree = sum(deg_dist.values()) / len(deg_dist.values())
    heterogeneity = np.std(list(deg_dist.values())) / mean_degree

    df = pd.DataFrame({
        "transitivity": [nx.transitivity(nxg)],
        "avg_clustering": [nx.average_clustering(nxg)],
        "avg_shortest_path": [nx.average_shortest_path_length(nxg)],
        "maximum_degree": [max_degree],
        "mean_degree": [mean_degree],
        "heterogeneity": [heterogeneity],
        "diameter": [nx.diameter(nxg)],
        "radius": [nx.radius(nxg)]
    })

    return df, df_local
