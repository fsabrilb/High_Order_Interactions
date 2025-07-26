# -*- coding: utf-8 -*-
"""
Created on Wednesday October 2 2024

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np  # type: ignore

# Global options ----
warnings.filterwarnings("ignore")


# Pairs forces according to hard-core repulsion ----
def estimate_pairs_forces(
    particles,
    interaction_distance: float,
    interaction_strength: float
) -> np.ndarray:
    """Estimation of pairs forces using hard-core repulsion

    Args
    ---------------------------------------------------------------------------
    particles: object
        Particles with default features:
            - Position
            - Velocity
            - Mass
            - Radius
            - Moving time
            - Rest time
            - Moving: Move or rest
            - Timer: Time elapsed until the transition between motion and rest
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
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            p_i, p_j = particles[i], particles[j]
            r_ij = p_i.position - p_j.position
            r_ij_d = np.linalg.norm(r_ij)
            if r_ij_d < interaction_distance and r_ij_d > 0:
                force = interaction_strength * (interaction_distance - r_ij_d)
                p_i.velocity += force * (r_ij / r_ij_d)
                p_j.velocity -= force * (r_ij / r_ij_d)


# Triplet forces according to hard-core repulsion ----
def estimate_triplets_forces(
    particles,
    interaction_distance: float,
    interaction_strength: float
) -> np.ndarray:
    """Estimation of triplet forces using hard-core repulsion

    Args
    ---------------------------------------------------------------------------
    particles: object
        Particles with default features:
            - Position
            - Velocity
            - Mass
            - Radius
            - Moving time
            - Rest time
            - Moving: Move or rest
            - Timer: Time elapsed until the transition between motion and rest
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
    for i in range(len(particles)):
        for j in range(i+1, len(particles)):
            for k in range(j+1, len(particles)):
                p_i, p_j, p_k = particles[i], particles[j], particles[k]
                r_ij = p_i.position - p_j.position
                r_ik = p_i.position - p_k.position
                r_jk = p_j.position - p_k.position
                r_ij_v = np.linalg.norm(r_ij)
                r_ik_v = np.linalg.norm(r_ik)
                r_jk_v = np.linalg.norm(r_jk)
                R = np.max([r_ij_v, r_ik_v, r_jk_v])
                if R < interaction_distance:
                    centroid = (p_i.position + p_j.position + p_k.position) / 3
                    for p in [p_i, p_j, p_k]:
                        r = p.position - centroid
                        r_d = np.linalg.norm(r)
                        if r_d > 0:
                            p.velocity -= interaction_strength * (r / r_d)
