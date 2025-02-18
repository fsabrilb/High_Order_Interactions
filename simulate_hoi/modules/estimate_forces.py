# -*- coding: utf-8 -*-
"""
Created on Wednesday October 2 2024

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)


# Pairs forces according to hard-core repulsion ----
def estimate_pairs_forces(
    r_i: np.ndarray,
    r_j: np.ndarray,
    interaction_distance: float,
    interaction_strength: float
) -> np.ndarray:
    """Estimation of pairs forces using hard-core repulsion

    Args
    ---------------------------------------------------------------------------
    r_i: np.ndarray
        Position of the i-th particle
    r_j: np.ndarray
        Position of the j-th particle
    interaction_distance: float
        Effective distance for the interaction
    interaction_strength: float
        Strength of coupling between pairs interactions such that the
        interaction is repulsive (attractive) if interaction_strength is
        greater (less) than 0

    Returns
    ---------------------------------------------------------------------------
    force_ij: np.ndarray
        Force between pairs according to pair interaction
    """

    r_ij = r_i - r_j
    r_ij_norm = np.linalg.norm(r_ij)
    force_magnitude = 0
    if r_ij_norm < interaction_distance:
        # force_magnitude = interaction_strength * (interaction_distance - r_ij_norm)  # noqa: E501
        force_magnitude = interaction_strength * (interaction_distance**2 - r_ij_norm**2)  # noqa: E501
    return force_magnitude * (r_ij / r_ij_norm)


# Triplet forces according to hard-core repulsion ----
def estimate_triplets_forces(
    r_i: np.ndarray,
    r_j: np.ndarray,
    r_k: np.ndarray,
    interaction_distance: float,
    interaction_strength: float
) -> np.ndarray:
    """Estimation of triplet forces using hard-core repulsion

    Args
    ---------------------------------------------------------------------------
    r_i: np.ndarray
        Position of the i-th particle
    r_j: np.ndarray
        Position of the j-th particle
    r_k: np.ndarray
        Position of the k-th particle
    interaction_distance: float
        Effective distance for the interaction
    interaction_strength: float
        Strength of coupling between triplet interactions such that the
        interaction is repulsive (attractive) if interaction_strength is
        greater (less) than 0

    Returns
    ---------------------------------------------------------------------------
    force_ijk: np.ndarray
        Force between triplets according to triplet interaction
    """

    r_ij = np.linalg.norm(r_i - r_j)
    r_ik = np.linalg.norm(r_i - r_k)
    force_magnitude = 0
    if (r_ij < interaction_distance) and (r_ik < interaction_distance):  # noqa: E501
        centroid = (r_i + r_j + r_k) / 3.0
        r_ijk = r_i - centroid
        r_ijk_norm = np.linalg.norm(r_ijk)
        # force_magnitude = interaction_strength * (interaction_distance - r_ijk_norm)  # noqa: E501
        force_magnitude = interaction_strength * r_ijk_norm

    return force_magnitude * (r_ijk / r_ijk_norm)
