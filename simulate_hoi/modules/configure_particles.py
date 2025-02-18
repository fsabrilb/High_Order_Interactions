# -*- coding: utf-8 -*-
"""
Created on Thursday August 29 2024

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import misc_functions as mf
import estimate_forces as ef

from functools import partial
from scipy.stats import levy_stable  # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)


# Definition of attributes of particles ----
class Particles:
    # Initialization of system of many particles ----
    def __init__(
        self,
        n_particles: int,
        n_dimensions: int
    ):
        """Initialize a collection of particles with default features:
            - Position
            - Velocity
            - Mass
            - Radius
            - Stopping time
            - Rest time
            - Moving: Move or rest
            - Timer: Time elapsed until the transition between motion and rest

        Args
        ---------------------------------------------------------------------------
        n_particles: int
            The number of particles to initialize
        n_dimensions: int
            The dimensionality of the system

        Returns
        ---------------------------------------------------------------------------
        particles: object
            Particles and its features
        """

        self.num_particles = n_particles    # Number of particles
        self.num_dimensions = n_dimensions  # System dimensionality

        # Initialize positions (x0_1, x0_2, ..., x0_d) to zeros
        self.positions = np.zeros((n_particles, n_dimensions))

        # Initialize velocities (v0_1, v0_2, ..., v0_d) to zeros
        self.velocities = np.zeros((n_particles, n_dimensions))

        # Initialize masses (m_1, m_2, ..., m_d) to ones
        self.masses = np.ones(n_particles)

        # Initialize sizes (a_1, a_2, ..., a_d) to ones
        self.radii = np.ones(n_particles)

        # Initialize stopping times (tau_1, tau_2, ..., tau_d) to ones
        self.stopping_times = np.ones(n_particles)

        # Initialize rest times (rho_1, rho_2, ..., rho_d) to ones
        self.rest_times = np.ones(n_particles)

        # Identify all particle in moving (0 -> False or 1 -> True)
        self.moving_types = np.ones(n_particles, dtype=bool)

        # Initialize timers (b_1, b_2, ..., b_d) to ones
        self.timers = np.ones(n_particles)

    # Set initial configuration of particles ----
    def set_initial_conditions(
        self,
        positions: np.ndarray = None,
        velocities: np.ndarray = None,
        masses: np.ndarray = None,
        radii: np.ndarray = None,
        stopping_times: np.ndarray = None,
        rest_times: np.ndarray = None,
        moving_types: np.ndarray = None
    ):
        """Set initial conditions for the particles

        Args
        ---------------------------------------------------------------------------
        positions: np.ndarray
            An array of shape (n_particles, n_dimensions) specifying initial
            positions (x0_1, x0_2, ..., x0_d)
        velocities: np.ndarray
            An array of shape (n_particles, n_dimensions) specifying initial
            velocities (v0_1, v0_2, ..., v0_d)
        masses: np.ndarray
            An array of shape (n_particles) specifying masses of each particle
            (m_1, m_2, ..., m_d)
        radii: np.ndarray
            An array of shape (n_particles) specifying radii of each particle
            (a_1, a_2, ..., a_d)
        stopping_times: np.ndarray
            An array of shape (n_particles) specifying the stopping times of
            each particle (tau_1, tau_2, ..., tau_d)
        rest_times: np.ndarray
            An array of shape (n_particles) specifying the rest times of each
            particle (rho_1, rho_2, ..., rho_d)
        moving_types: np.ndarray
            An array of shape (n_particles) specifying if particle is in motion
            or not

        Returns
        ---------------------------------------------------------------------------
        particles: object
            Particles and its features
        """

        if positions is not None:
            self.positions = np.array(positions)

        if velocities is not None:
            self.velocities = np.array(velocities)

        if masses is not None:
            self.masses = np.array(masses)

        if radii is not None:
            self.radii = np.array(radii)

        if stopping_times is not None:  # Time to move
            self.stopping_times = np.array(stopping_times)

        if rest_times is not None:  # Rest time
            self.rest_times = np.array(rest_times)

        if moving_types is not None:
            self.moving_types = np.array(moving_types)

        self.timers = np.where(
            self.moving_types,    # Select "in motion" particles
            self.stopping_times,  # Real motion
            self.rest_times       # Rest
        )


# Definition of Higher Order Interactions ----
class Higher_Order_Interactions:
    # Initialization of system of many particles ----
    def __init__(
        self,
        n_particles: int,
        n_dimensions: int,
        n_degree: int = 2
    ):
        """Initialize higher order interactions (HoI) coefficients:
            - mu_x: Stochastic drift coefficient for the position
            - mu_v: Stochastic drift coefficient for the velocity
            - eta_x: Diffusion coefficient for the position
            - eta_v: Diffusion coefficient for the velocity

        Args
        ---------------------------------------------------------------------------
        n_particles: int
            The number of particles to initialize
        n_dimensions: int
            The dimensionality of the system
        n_degree: int
            The degree of the highest order interaction

        Returns
        ---------------------------------------------------------------------------
        collider: object
            Magnitude of higher order interactions
        """

        self.num_particles = n_particles    # Number of particles
        self.num_dimensions = n_dimensions  # System dimensionality
        self.num_hoi = n_degree             # Highest Order Interaction degree

        # Initialize drift coefficient for position
        self.mu_x = np.zeros((n_particles, n_dimensions))

        # Initialize drift coefficient for velocity
        self.mu_v = np.zeros((n_particles, n_dimensions))

        # Initialize diffusion coefficient for position
        self.eta_x = np.zeros((n_particles, n_dimensions))

        # Initialize diffusion coefficient for velocity
        self.eta_v = np.zeros((n_particles, n_dimensions))

    # Update stochastic drift over positions ----
    def update_stochastic_drift_position(
        self,
        particles: Particles
    ):
        """Estimate stochastic drift term over positions (contribution of all
        deterministic factors that contributes to position of the particles)

        Args
        ---------------------------------------------------------------------------
        particles: Particles object class
            Array of n_particles with n_dimensions

        Returns
        ---------------------------------------------------------------------------
        collider: object
            Magnitude of higher order interactions
        """

        self.mu_x = particles.velocities

    # Update stochastic drift over velocities ----
    def update_stochastic_drift_velocity(
        self,
        particles: Particles,
        interaction_distance: float,
        interaction_strength: float
    ) -> np.ndarray:
        """Estimate all deterministic forces over every particle (stochastic
        drift term over velocities)

        Args
        ---------------------------------------------------------------------------
        particles: Particles object class
            Array of n_particles with n_dimensions
        interaction_distance: float
            Effective distance for the interaction
        interaction_strength: float
            Strength of coupling between pairs interactions such that the
            interaction is repulsive (attractive) if interaction_strength is
            greater (less) than 0

        Returns
        ---------------------------------------------------------------------------
        collider: object
            Magnitude of higher order interactions
        """

        self.mu_v = np.zeros_like(particles.velocities)
        for i in range(self.num_particles):
            # Add all individual deterministic forces

            # Accumulate the force pairwise forces: From j over i
            for j in range(self.num_particles):
                if i != j:
                    self.mu_v[i] += ef.estimate_pairs_forces(
                        r_i=particles.positions[i],
                        r_j=particles.positions[j],
                        interaction_distance=interaction_distance,
                        interaction_strength=interaction_strength
                    )

            # Taking account the effect of mass
            self.mu_v[i] = self.mu_v[i] / particles.masses[i]

    # Update stochastic diffusion coefficient over velocities ----
    def update_stochastic_diffusion_coefficient_velocity(
        self,
        noise_type: str,
        noise_params: np.ndarray,
        dt: float
    ):
        """Estimate all stochastic effects over every particle (stochastic
        diffusion term over velocities)

        Args
        ---------------------------------------------------------------------------
        noise_type: str
            Define the type of the noise between:
                - Random uniform angle (RUA)
                - Brownian motion (white noise) (BM)
                - Levy alpha-stable flight noise (coloured noise) (LF)
        noise_params: np.ndarray
            Params for the definition of every noise, namely:
                - RUA: Angle between 0 and 2 pi and speed (1D)
                - BM: Location and Scale (2D)
                - LF: Stability, Skewness, Location and Scale (4D)
        dt: float
            Infinitesimal time for the stochastic integrator

        Returns
        ---------------------------------------------------------------------------
        collider: object
            Magnitude of higher order interactions
        """

        # Noise construction
        if noise_type == "RUA":
            if self.num_dimensions == 2:  # 2D
                angle = np.random.uniform(low=0, high=2*np.pi, size=(self.num_particles))  # noqa: E501
                for n_ in range(self.num_particles):
                    self.eta_v[n_] = noise_params[0] * np.array([np.cos(angle[n_]), np.sin(angle[n_])])  # noqa: E501

        if noise_params == "BM":
            self.eta_v = np.random.normal(
                loc=noise_params[0],  # Location parameter
                scale=noise_params[1]*np.sqrt(dt),  # Scale parameter
                size=(self.num_particles, self.num_dimensions)
            )
        if noise_params == "LF":
            self.eta_v = levy_stable.rvs(
                alpha=noise_params[0],  # Stability parameter
                beta=noise_params[1],  # Skewnnes parameter
                loc=noise_params[2],  # Location parameter
                scale=noise_params[3]*np.power(dt, 1 / noise_params[0]),  # Scale parameter # noqa: E501
                size=(self.num_particles, self.num_dimensions)
            )

    # Update moving type variable (estimate jump diffusion component) ----
    def update_moving(self, particles: Particles, dt: float):
        """Update moving type of each particle

        Args
        ---------------------------------------------------------------------------
        particles: Particles object class
            Array of n_particles with n_dimensions
        dt: float
            Infinitesimal time for the stochastic integrator

        Returns
        ---------------------------------------------------------------------------
        collider: object
            Magnitude of higher order interactions
        """

        # Update moving type
        mask = (particles.timers <= 0)  # Select "in transition" particles
        particles.moving_types[mask] = ~particles.moving_types[mask]

        # Update timers and velocities with new moving type value
        for i in range(self.num_particles):
            if particles.moving_types[i] == True:  # Switch from rest to move  # noqa: 501
                if particles.timers[i] <= 0:
                    particles.timers[i] += particles.stopping_times[i]
            else:  # Switch from move to rest
                if particles.timers[i] <= 0:
                    particles.timers[i] += particles.rest_times[i]

                particles.velocities[i] = np.zeros(self.num_dimensions)
                self.eta_v[i] = np.zeros(self.num_dimensions)

        particles.timers -= dt

    # Define the stochastic integrator (Euler algorithm) ----
    def integrator_euler(
        self,
        particles: Particles,
        interaction_distance: float,
        interaction_strength: float,
        noise_type: str,
        noise_params: np.ndarray,
        dt: float,
        periodic: bool,
        box_sizes: np.ndarray
    ):
        """Integrate the stochastic differential equation using Euler algorithm

        Args
        ---------------------------------------------------------------------------
        particles: Particles object class
            Array of n_particles with n_dimensions
        interaction_distance: float
            Effective distance for the interaction
        interaction_strength: float
            Strength of coupling between pairs interactions such that the
            interaction is repulsive (attractive) if interaction_strength is
            greater (less) than 0
        noise_type: str
            Define the type of the noise between:
                - Random uniform angle (RUA)
                - Brownian motion (white noise) (BM)
                - Levy alpha-stable flight noise (coloured noise) (LF)
        noise_params: np.ndarray
            Params for the definition of every noise, namely:
                - RUA: Angle between 0 and 2 pi and speed (1D)
                - BM: Location and Scale (2D)
                - LF: Stability, Skewness, Location and Scale (4D)
        dt: float
            Infinitesimal time for the stochastic integrator
        periodic : bool
            Periodic boundary conditions
        box_sizes : np.ndarray
            An array of shape (n_dimensions) specifying size of the box when
            the Interaction Based Model has periodic boundaries

        Returns
        ---------------------------------------------------------------------------
        collider: object
            Magnitude of higher order interactions
        """

        # Euler step - Update stochastic diffusion variables
        self.update_stochastic_diffusion_coefficient_velocity(
            noise_type=noise_type,
            noise_params=noise_params,
            dt=dt
        )

        # Filter moving type for each particle
        self.update_moving(particles=particles, dt=dt)

        # Euler step - Update stochastic drift variables
        self.update_stochastic_drift_position(particles=particles)
        self.update_stochastic_drift_velocity(
            particles=particles,
            interaction_distance=interaction_distance,
            interaction_strength=interaction_strength
        )

        # Euler step - Stochastic integration
        particles.positions += self.mu_x * dt + self.eta_x
        particles.velocities += self.mu_v * dt + self.eta_v

        # Ensure particles wrap around the screen (toroidal space)
        if periodic is True:
            particles.positions = particles.positions % np.array(box_sizes)

    # Define the stochastic integrator (Heun algorithm) ----
    def integrator_heun(
        self,
        particles: Particles,
        interaction_distance: float,
        interaction_strength: float,
        noise_type: str,
        noise_params: np.ndarray,
        dt: float,
        periodic: bool,
        box_sizes: np.ndarray
    ):
        """Integrate the stochastic differential equation using Heun algorithm

        Args
        ---------------------------------------------------------------------------
        particles: Particles object class
            Array of n_particles with n_dimensions
        interaction_distance: float
            Effective distance for the interaction
        interaction_strength: float
            Strength of coupling between pairs interactions such that the
            interaction is repulsive (attractive) if interaction_strength is
            greater (less) than 0
        noise_type: str
            Define the type of the noise between:
                - Random uniform angle (RUA)
                - Brownian motion (white noise) (BM)
                - Levy alpha-stable flight noise (coloured noise) (LF)
        noise_params: np.ndarray
            Params for the definition of every noise, namely:
                - RUA: Angle between 0 and 2 pi and speed (1D)
                - BM: Location and Scale (2D)
                - LF: Stability, Skewness, Location and Scale (4D)
        dt: float
            Infinitesimal time for the stochastic integrator
        periodic : bool
            Periodic boundary conditions
        box_sizes : np.ndarray
            An array of shape (n_dimensions) specifying size of the box when
            the Interaction Based Model has periodic boundaries

        Returns
        ---------------------------------------------------------------------------
        collider: object
            Magnitude of higher order interactions
        """

        # Euler step - Update stochastic diffusion variables
        self.update_stochastic_diffusion_coefficient_velocity(
            noise_type=noise_type,
            noise_params=noise_params,
            dt=dt
        )

        # Filter moving type for each particle
        self.update_moving(particles=particles, dt=dt)

        # Euler step - Update stochastic drift variables
        self.update_stochastic_drift_position(particles=particles)
        self.update_stochastic_drift_velocity(
            particles=particles,
            interaction_distance=interaction_distance,
            interaction_strength=interaction_strength
        )

        # Euler step - Stochastic integration
        particles.positions += self.mu_x * dt + self.eta_x
        particles.velocities += self.mu_v * dt + self.eta_v

        # Heun step - Update stochastic drift and diffusion variables
        mu_x_h = self.mu_x
        mu_v_h = self.mu_v
        self.update_stochastic_drift_position(particles=particles)
        self.update_stochastic_drift_velocity(
            particles=particles,
            interaction_distance=interaction_distance,
            interaction_strength=interaction_strength
        )

        # Heun step - Stochastic integration
        particles.positions += 0.5 * (mu_x_h + self.mu_x) * dt + self.eta_x
        particles.velocities += 0.5 * (mu_v_h + self.mu_v) * dt + self.eta_v

        if periodic is True:
            particles.positions = particles.positions % np.array(box_sizes)


# Simulate a sample of Interaction based Model ----
def estimate_sample(
    interaction_distance: float,
    interaction_strength: float,
    noise_type: str,
    noise_params: np.ndarray,
    periodic: bool,
    box_sizes: np.ndarray,
    log_path: str,
    log_filename: str,
    verbose: int,
    initial_conditions
) -> pd.DataFrame:
    """Simulation of individual based model (IbM) sample according to:
        n0 = initial_conditions[0]            # Number of particles
        d0 = initial_conditions[1]            # Degrees of freedom per particle
        x0 = initial_conditions[2]            # Initial position per particle
        v0 = initial_conditions[3]            # Initial velocity per particle
        m_ = initial_conditions[4]            # Masses per particle
        r_ = initial_conditions[5]            # Radii per particle
        taus_ = initial_conditions[6]         # Stopping times per particle
        rhos_ = initial_conditions[7]         # Rest times per particle
        types_ = initial_conditions[8]        # Moving types per particle
        t0 = initial_conditions[9]            # Initial time
        tf = initial_conditions[10]           # Final time
        n_steps = initial_conditions[11]      # Number of temporal steps
        n_simulation = initial_conditions[12] # Number of simulations

    Args
    ---------------------------------------------------------------------------
    interaction_distance : float
            Effective distance for the interaction
    interaction_strength : float
        Strength of coupling between pairs interactions such that the
        interaction is repulsive (attractive) if interaction_strength is
        greater (less) than 0
    noise_type : str
        Define the type of the noise between:
            - Random uniform angle (RUA)
            - Brownian motion (white noise) (BM)
            - Levy alpha-stable flight noise (coloured noise) (LF)
    noise_params : np.ndarray
        Params for the definition of every noise, namely:
            - RUA: Angle between 0 and 2 pi and speed (1D)
            - BM: Location and Scale (2D)
            - LF: Stability, Skewness, Location and Scale (4D)
    periodic : bool
        Periodic boundary conditions
    box_sizes : np.ndarray
        An array of shape (n_dimensions) specifying size of the box when the
        IbM has periodic boundaries
    log_path : string
        Local path for logs (default value is "../logs")
    log_filename : string
        Local filename for logs (default value is "log_ibm")
    verbose : int
        Provides additional details as to what the computer is doing when
        sample of IbM is running

    Returns
    ---------------------------------------------------------------------------
    df : pandas DataFrame
        Individual Based Model sample
    """

    # Definition of initial parameters
    n0 = initial_conditions[0]             # Number of particles
    d0 = initial_conditions[1]             # Degrees of freedom per particle
    x0 = initial_conditions[2]             # Initial position per particle
    v0 = initial_conditions[3]             # Initial velocity per particle
    m_ = initial_conditions[4]             # Masses per particle
    r_ = initial_conditions[5]             # Radii per particle
    taus_ = initial_conditions[6]          # Stopping times per particle
    rhos_ = initial_conditions[7]          # Rest times per particle
    types_ = initial_conditions[8]         # Moving types per particle
    t0 = initial_conditions[9]             # Initial time
    tf = initial_conditions[10]            # Final time
    n_steps = initial_conditions[11]       # Number of temporal steps
    n_simulation = initial_conditions[12]  # Number of simulations

    # Initialization of particles and higher order interactions
    particles = Particles(n_particles=int(n0), n_dimensions=int(d0))
    collider = Higher_Order_Interactions(n_particles=int(n0), n_dimensions=int(d0), n_degree=2)  # noqa: 501

    # Set initial conditions of particles and time interval elapsed
    particles.set_initial_conditions(
        positions=x0,
        velocities=v0,
        masses=m_,
        radii=r_,
        stopping_times=taus_,
        rest_times=rhos_,
        moving_types=types_
    )

    n_steps = int(n_steps)
    t = np.linspace(start=t0, stop=tf, num=n_steps)
    dt = (tf - t0) / n_steps  # Temporal step
    x = np.zeros((n_steps, particles.num_particles, particles.num_dimensions))
    v = np.zeros((n_steps, particles.num_particles, particles.num_dimensions))
    masses = np.zeros((n_steps, particles.num_particles))
    radii_ = np.zeros((n_steps, particles.num_particles))
    taus__ = np.zeros((n_steps, particles.num_particles))
    rhos__ = np.zeros((n_steps, particles.num_particles))
    types_ = np.zeros((n_steps, particles.num_particles))
    timers = np.zeros((n_steps, particles.num_particles))

    x[0] = particles.positions
    v[0] = particles.velocities
    masses[0] = particles.masses
    radii_[0] = particles.radii
    taus__[0] = particles.stopping_times
    rhos__[0] = particles.rest_times
    types_[0] = particles.moving_types
    timers[0] = particles.timers

    # Integration of stochastic differential equation (Heun algorithm O(dt^3))
    for j in range(0, n_steps - 1):
        # Euler integrator step
        collider.integrator_euler(
            particles=particles,
            interaction_distance=interaction_distance,
            interaction_strength=interaction_strength,
            noise_type=noise_type,
            noise_params=noise_params,
            dt=dt,
            periodic=periodic,
            box_sizes=box_sizes
        )

        x[j+1] = particles.positions
        v[j+1] = particles.velocities
        masses[j+1] = particles.masses
        radii_[j+1] = particles.radii
        taus__[j+1] = particles.stopping_times
        rhos__[j+1] = particles.rest_times
        types_[j+1] = particles.moving_types
        timers[j+1] = particles.timers

    # -------------------- Definition of final dataframe -------------------- #
    s_simulation = np.repeat(
        n_simulation,
        n_steps * particles.num_particles * particles.num_dimensions
    ).astype(str)

    s_particle = np.tile(
        np.repeat(
            np.char.zfill(
                np.arange(start=1, stop=particles.num_particles + 1).astype(str),  # noqa: E501
                int(np.log10(particles.num_particles)) + 1
            ),
            particles.num_dimensions
        ),
        n_steps
    ).astype(str)

    s_axis = np.tile(
        np.arange(particles.num_dimensions) + 1,
        n_steps * particles.num_particles
    ).astype(str)

    df = pd.DataFrame(
        {
            "simulation": np.char.add(np.array(["Simulation "]), s_simulation),
            "particle": np.char.add(np.array(["Particle "]), s_particle),
            "axis": np.char.add(np.array(["x_"]), s_axis),
            "mass": np.tile(masses.flatten(), particles.num_dimensions),  # noqa: E501
            "radius": np.tile(radii_.flatten(), particles.num_dimensions),  # noqa: E501
            "stopping_time": np.tile(taus__.flatten(), particles.num_dimensions),  # noqa: E501
            "rest_time": np.tile(rhos__.flatten(), particles.num_dimensions),  # noqa: E501
            "moving_type": np.tile(types_.flatten(), particles.num_dimensions),  # noqa: E501
            "timers": np.repeat(timers, particles.num_dimensions, axis=1).flatten(),  # noqa: E501
            "time": np.repeat(t, particles.num_particles * particles.num_dimensions),  # noqa: E501
            "position": x.flatten(),
            "velocity": v.flatten()
        }
    )

    # Function development
    if verbose >= 1:
        with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
            file.write(
                "n0={}, d0={}, t0={}, tf={}, n_steps={}, n_simulation={}\n".format(  # noqa: E501
                    n0,
                    d0,
                    t0,
                    tf,
                    n_steps,
                    n_simulation
                )
            )

    return df


# Simulation of multiple IbM with the same initial conditions ----
def simulate_ibm(
    interaction_distance: float,
    interaction_strength: float,
    noise_type: str,
    noise_params: np.ndarray,
    periodic: bool,
    box_sizes: np.ndarray,
    initial_conditions,
    log_path: str = "../logs",
    log_filename: str = "log_pm",
    verbose: int = 1,
    tqdm_bar: bool = True
) -> pd.DataFrame:
    """Simulation of multiple individual based model (IbM) sample according to:
        n0 = initial_conditions[0]            # Number of particles
        d0 = initial_conditions[1]            # Degrees of freedom per particle
        x0 = initial_conditions[2]            # Initial position per particle
        v0 = initial_conditions[3]            # Initial velocity per particle
        m_ = initial_conditions[4]            # Masses per particle
        r_ = initial_conditions[5]            # Radii per particle
        taus_ = initial_conditions[6]         # Stopping times per particle
        rhos_ = initial_conditions[7]         # Rest times per particle
        types_ = initial_conditions[8]        # Moving types per particle
        t0 = initial_conditions[9]            # Initial time
        tf = initial_conditions[10]           # Final time
        n_steps = initial_conditions[11]      # Number of temporal steps
        n_simulation = initial_conditions[12] # Number of simulations

    for n_samples entitled by n_simulation.

    Args
    ---------------------------------------------------------------------------
    interaction_distance : float
            Effective distance for the interaction
    interaction_strength : float
        Strength of coupling between pairs interactions such that the
        interaction is repulsive (attractive) if interaction_strength is
        greater (less) than 0
    noise_type : str
        Define the type of the noise between:
            - Random uniform angle (RUA)
            - Brownian motion (white noise) (BM)
            - Levy alpha-stable flight noise (coloured noise) (LF)
    noise_params : np.ndarray
        Params for the definition of every noise, namely:
            - RUA: Angle between 0 and 2 pi and speed (1D)
            - BM: Location and Scale (2D)
            - LF: Stability, Skewness, Location and Scale (4D)
    periodic : bool
        Periodic boundary conditions
    box_sizes : np.ndarray
        An array of shape (n_dimensions) specifying size of the box when the
        IbM has periodic boundaries
    log_path : string
        Local path for logs (default value is "../logs")
    log_filename : string
        Local filename for logs (default value is "log_ibm")
    verbose : int
        Provides additional details as to what the computer is doing when
        sample of IbM is running
    tqdm_bar : bool
        Progress bar in parallel run (default value is True)

    Returns
    ---------------------------------------------------------------------------
    df : pandas DataFrame
        Multiple Individual Based Model samples
    """

    # Auxiliary function for simulations of Polydisperse mixtures paths
    fun_local = partial(
        estimate_sample,
        interaction_distance,
        interaction_strength,
        noise_type,
        noise_params,
        periodic,
        box_sizes,
        log_path,
        log_filename,
        verbose
    )

    # Parallel loop for simulations of IbM paths
    df = mf.parallel_run(
        fun=fun_local,
        arg_list=initial_conditions,
        tqdm_bar=tqdm_bar
    )
    df = pd.concat(df)

    return df


# Get initial conditions for the simulations ----
def initialize_periodic_boundary_conditions(
    particles_n: int,
    particle_df: int,
    particle_size: float,
    particle_fraction: float = 0.5
):
    """Estimation of the initial positions and velocities assuming periodic
    bounds conditions given the fraction of volume occupied by the bath and
    spherical particles

    Args
    ---------------------------------------------------------------------------
    particles_n : int
        The number of particles to initialize per axis
    particles_df : int
        Degrees of freedom per particle in the system
    particles_size : float
        Radius of particles of the bath (no tracer particle)
    particles_fraction : float
        Fraction of volume occupied by the particles of the bath (no tracer
        particle) (default value 0.5 (50%))

    Returns
    ---------------------------------------------------------------------------
    positions : np.ndarray
        Initial positions of the bath of the Polydisperse mixtures sample
    velocities : np.ndarray
        Initial velocities of the bath of the Polydisperse mixtures sample
    box_size : float
        Size of the box with periodic bounds for the correct particle_fraction
    """

    # Total number of particles
    n0 = int(np.power(particles_n, particle_df))

    # Position initialization
    positions = np.zeros((n0, particle_df))
    particle_volume = particle_size
    if particle_df == 3:
        particle_volume = 4 * np.pi * np.power(particle_size, particle_df) / 3
    elif particle_df == 2:
        particle_volume = np.pi * np.power(particle_size, particle_df)
    else:
        print("Boundary condition is not estimated in 2D or 3D space")

    box_size = particles_n * np.power(particle_volume / particle_fraction, 1 / particle_df)  # noqa: E501
    positions_n = np.linspace(start=0, stop=box_size, num=particles_n)

    if particle_df == 3:
        positions_x, positions_y, positions_z = np.meshgrid(positions_n, positions_n, positions_n)  # noqa: E501
        positions = np.column_stack([positions_x.ravel(), positions_y.ravel(), positions_z.ravel()])  # noqa: E501
    if particle_df == 2:
        positions_x, positions_y = np.meshgrid(positions_n, positions_n)
        positions = np.column_stack([positions_x.ravel(), positions_y.ravel()])

    # Velocity initialization
    velocities = np.zeros((n0, particle_df))

    return positions, velocities, box_size

# TODO: Revie if numpy.random.seed() works well (every iteration needs
# different noises)
