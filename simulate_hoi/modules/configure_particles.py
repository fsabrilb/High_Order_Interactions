# -*- coding: utf-8 -*-
"""
Created on Thursday August 29 2024

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import random
import pygame  # type: ignore
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import estimate_forces as ef

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)


# Definition of attributes of particles ----
class Particle:
    # Initialization of one particle ----
    def __init__(
        self,
        width: float,
        height: float,
        dt: float,
        mass: float,
        radius: float,
        move_time_min: float,
        move_time_max: float,
        rest_time_min: float,
        rest_time_max: float,
        speed: float,
        boundary_type: str = "Hard"
    ):
        """Initialize a particle with default features:
            - Position
            - Velocity
            - Mass
            - Radius
            - Moving time
            - Rest time
            - Moving: Move or rest
            - Timer: Time elapsed until the transition between motion and rest

        Args
        -----------------------------------------------------------------------
        width: float
            Width of the box of the simulation
        height: float
            Height of the box of the simulation
        dt: float
            Infinitesimal increasement in time
        mass: float
            Mass of the particle
        radius: float
            Radius of the particle
        move_time_min: float
            Minimum time in movement when the particles have a non-null rest
            time
        move_time_max: float
            Maximum time in movement when the particles have a non-null rest
            time
        rest_time_min: float
            Minimum rest time when the particles have a non-null rest time
        rest_time_max: float
            Maximum rest time when the particles have a non-null rest time
        speed: float
            Norm of the velocity after a rest period
        boundary: str
            Boundary type:
                - 'Hard': Closed box
                - 'Periodic': Toroidal space

        Returns
        -----------------------------------------------------------------------
        particles: object
            Particle and its features
        """
        # Initialize parameters of simulation
        self.width = width
        self.height = height
        self.dt = dt
        self.mass = mass
        self.radius = radius
        self.move_time_min = move_time_min
        self.move_time_max = move_time_max
        self.rest_time_min = rest_time_min
        self.rest_time_max = rest_time_max
        self.speed = speed
        self.boundary_type = boundary_type

        # Initialize position (x, y) to random position in the box
        self.position = np.array([
            random.uniform(0, self.width),
            random.uniform(0, self.height)
        ])

        # Initialize velocities (vx, vy) to zero
        self.velocity = np.zeros(2)

        # Initialize mass to one
        self.mass = 1

        # Initialize size to one
        self.radius = 1

        # Initialize moving time to random time in a uniform interval
        self.move_time = random.uniform(self.move_time_min, self.move_time_max)

        # Initialize rest time to random time in a uniform interval
        self.rest_time = random.uniform(self.rest_time_min, self.rest_time_max)

        # Identify particle in moving (0 -> False or 1 -> True)
        self.moving_type = True

        # Initialize timers as moving time
        self.timer = self.move_time

    # Update position and velocity of the particle ----
    def update(self):
        """Update particle position and velocity"""
        if self.moving_type:
            self.position += self.velocity * self.dt
            self.timer -= self.dt
            self.handle_movement_end()
        else:
            self.timer -= self.dt
            if self.timer <= 0:
                self.start_moving()

        if self.boundary_type == "Hard":
            self.apply_hard_boundary()
        elif self.boundary_type == "Periodic":
            self.position = np.mod(self.position, [self.width, self.height])

    # Finish movement according to the moving_type flag ----
    def handle_movement_end(self):
        if self.timer <= 0:
            self.timer = self.rest_time
            self.moving_type = False
            self.velocity = np.zeros(2)

    # Start movement according to the moving_type flag ----
    def start_moving(self):
        """Update particle position and velocity after rest period"""
        self.timer = self.move_time
        self.moving = True
        angle = random.uniform(0, 2 * np.pi)
        self.velocity = self.speed * np.array([np.cos(angle), np.sin(angle)])

    # Apply hard boundary condition
    def apply_hard_boundary(self):
        """Update particle position and velocity after rest period"""
        for dim in range(2):
            if self.position[dim] <= 0:
                self.position[dim] = 0
                self.velocity[dim] *= -1
            elif self.position[dim] >= [self.width, self.height][dim]:
                self.position[dim] = [self.width, self.height][dim]
                self.velocity[dim] *= -1


# Deployment of simulation ----
def run_simulation(
    num_particles: float,
    width: float,
    height: float,
    dt: float,
    mass: float,
    radius: float,
    move_time_min: float,
    move_time_max: float,
    rest_time_min: float,
    rest_time_max: float,
    speed: float,
    boundary_type: str,
    simulation_duration: float,
    interaction_strength_2: float,
    interaction_strength_3: float,
    interaction_distance_2: float,
    interaction_distance_3: float,
    time_step_record: float,
    pygame_image: bool = False
) -> pd.DataFrame:
    """Initialize a particle with default features:
        - Position
        - Velocity
        - Mass
        - Radius
        - Moving time
        - Rest time
        - Moving: Move or rest
        - Timer: Time elapsed until the transition between motion and rest

    Args
    ---------------------------------------------------------------------------
    num_particles: float
        Number of particles
    width: float
        Width of the box of the simulation
    height: float
        Height of the box of the simulation
    dt: float
        Infinitesimal increasement in time
    mass: float
        Mass of the particle
    radius: float
        Radius of the particle
    move_time_min: float
        Minimum time in movement when the particles have a non-null rest
        time
    move_time_max: float
        Maximum time in movement when the particles have a non-null rest
        time
    rest_time_min: float
        Minimum rest time when the particles have a non-null rest time
    rest_time_max: float
        Maximum rest time when the particles have a non-null rest time
    speed: float
        Norm of the velocity after a rest period
    boundary: str
        Boundary type:
            - 'Hard': Closed box
            - 'Periodic': Toroidal space
    simulation_duration: float
        Duration of simulation
    interaction_strength_2: float
        Effective distance for the pairwise interaction
    interaction_strength_3: float
        Effective distance for the triplet interaction
    interaction_distance_2: float
        Strength of coupling between pairs interactions such that the
        interaction is repulsive (attractive) if interaction_strength is
        greater (less) than 0
    interaction_distance_3: float
        Strength of coupling between triplet interactions such that the
        interaction is repulsive (attractive) if interaction_strength is
        greater (less) than 0
    time_step_record: float
        Time step for recording data in a dataframe with columns of positions
        and velocities
    pygame_image: bool
        Pygame output flag. Default value False

    Returns
    ---------------------------------------------------------------------------
    particles: object
        Particle and its features
    """
    if pygame_image:
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()

    # Initialize particles
    particles = [
        Particle(
            width=width,
            height=height,
            dt=dt,
            mass=mass,
            radius=radius,
            move_time_min=move_time_min,
            move_time_max=move_time_max,
            rest_time_min=rest_time_min,
            rest_time_max=rest_time_max,
            speed=speed,
            boundary_type=boundary_type
        ) for _ in range(num_particles)
    ]

    # Initialize dataframe record
    data, current_time, next_record_time = [], 0.0, 0.0
    columns = (
        ["time"]
        + [f"{r}{i+1}" for i in range(num_particles) for r in ("x", "y")]
        + [f"{v}{i+1}" for i in range(num_particles) for v in ("vx", "vy")]
    )

    # Initialize simulation
    running = True
    while running and current_time <= simulation_duration:
        if pygame_image:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # Update particles positions and velocities
        for p in particles:
            p.update()

        # Estimate forces
        ef.estimate_pairs_forces(
            particles=particles,
            interaction_distance=interaction_distance_2,
            interaction_strength=interaction_strength_2
        )
        ef.estimate_triplets_forces(
            particles=particles,
            interaction_distance=interaction_distance_3,
            interaction_strength=interaction_strength_3
        )

        # Add data to recorded data
        if current_time >= next_record_time:
            data.append(
                [current_time]
                + [r for p in particles for r in p.position]
                + [v for p in particles for v in p.velocity]
            )
            next_record_time += time_step_record

        # Update pygame frame
        if pygame_image:
            screen.fill((0, 0, 0))
            if boundary_type == "Hard":
                pygame.draw.rect(
                    screen,
                    (255, 255, 255),
                    (0, 0, width, height),
                    2
                )

            for p in particles:
                pygame.draw.circle(
                    screen,
                    (0, 255, 0),
                    p.position.astype(int),
                    radius
                )

            pygame.display.flip()
            clock.tick(60)

        current_time += dt

    if pygame_image:
        pygame.quit()

    df = pd.DataFrame(data, columns=columns)
    return df
