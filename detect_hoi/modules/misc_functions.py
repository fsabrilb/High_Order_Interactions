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


# Estimate distance between two positions time series ----
def estimate_distances(
    df: pd.DataFrame,
    filter_step: int = None
) -> pd.DataFrame:
    """Estimate distances between the time series of two individuals. The data
    has the following columns:
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
        Dataframe with the distances between two different IDs with the
        following columns:
            - time: Times (frames)
            - id_pair: Pair of IDs coupled for the distance estimation
            - distance: Euclidean distance between the pair of IDs
    """

    df_final = []
    print("Estimate distances for:", len(df["permuted_id"].unique()), "ids")
    for id_1 in sorted(df["permuted_id"].unique()):
        for id_2 in sorted(df["permuted_id"].unique()):
            if id_1 < id_2:
                id_pair = str(id_1) + str(id_2)
                print("- Pair: {}".format(id_pair))
                time = df[df["permuted_id"] == id_1]["time"].values
                x1 = df[df["permuted_id"] == id_1]["n_x"].values
                x2 = df[df["permuted_id"] == id_2]["n_x"].values
                y1 = df[df["permuted_id"] == id_1]["n_y"].values
                y2 = df[df["permuted_id"] == id_2]["n_y"].values
                distance = np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2))
                df_final.append(
                    pd.DataFrame({
                        "time": time,
                        "id_pair": [id_pair]*len(time),
                        "distance": distance
                    })
                )
    df_final = pd.concat(df_final, ignore_index=True)

    if filter_step is not None:
        df_final = df_final[df_final["time"] % filter_step == 0]
        print("-- Skipped data every {} points".format(filter_step))
    print("")
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
