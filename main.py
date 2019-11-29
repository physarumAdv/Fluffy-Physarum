from random import randint

import numpy as np

from simulation_clastions import Polyhedron, TrailDot, MapDot, \
    Particle, transmission_matrix, face_to_space, get_distance
from visualizer import Visualizer


cel = 1000 # cube_edge_length

polyhedron = Polyhedron(
    vertices=np.asarray([(0, 0, 0), (0, 0, cel), \
        (cel, 0, cel), (cel, 0, 0), (cel, cel, 0), \
        (0, cel, 0), (0, cel, cel), (cel, cel, cel)]),
    faces=np.asarray([(0, 1, 2, 3), (0, 5, 6, 1), \
        (0, 3, 4, 5), (2, 7, 4, 3), (1, 6, 7, 2), \
        (4, 7, 6, 5)])
    )
initializing_face = (0, 3, 4, 5)

start_point_coordinates = np.asarray((cel//2, cel//2, 0))

def get_map_dot(coords, simulation_map):
    """Returns the map dot info
    Parameters:
        coords (np.ndarray of three `int`s): Coordinates of the dot
        simulation_map (dict from tuple of three `int`s to MapDot):
            the simulation map to get data from
    Returns:
        MapDot: the information about the dot
    """
    coords = np.round(np.asarray(coords)).astype(int).tolist()
    coords = tuple(coords)
    if coords not in simulation_map.keys():
        simulation_map[coords] = MapDot()
    return simulation_map[coords]


def add_particle(particles, coordinates, face, polyhedron):
    """Adds a prticle to the simulation
    Parameters:
        particles (list of `Particle`s): The model's particles
        coordinates (np.ndarray of three `int`s): The coordinates
            to initialize particle at
        face (one-dimensional np.ndarray of `int`s):
            the polyhedron's face to initialize the particle on
            (given as an array of face's vertices' numbers)
        polyhedron (Polyhedron): the polyhedron to initialize
            particle on
    """
    angle = np.radians(randint(1, 360))
    central_sensor = np.asarray([
        Particle.SENSOR_OFFSET * np.cos(angle),
        Particle.SENSOR_OFFSET * np.sin(angle),
        0
        ])
    move_vector = face_to_space(central_sensor, \
        transmission_matrix(face, polyhedron))
    move_vector /= get_distance(np.asarray([0, 0, 0]), move_vector)
    move_vector *= Particle.SENSOR_OFFSET

    new_point = Particle(
        coordinates,
        coordinates + move_vector,
        face,
        polyhedron
        )
    new_point.init_sensors_from_center(polyhedron)
    particles.append(new_point)

def simulate(start_point_coordinates, initializing_face, \
        polyhedron):
    """Runs the simulation
    Parameters:
        start_point_coordinates (np.ndarray of three `int`s):
            the coordinates to initialize new particles at
        initializing_face (one-dimensional np.ndarray of `int`s):
            the polyhedron's face to initialize new particles on
            (given as an array of face's vertices' numbers)
        polyhedron (Polyhedron): the polyhedron to initialize
            new particles on
    """
    simulation_map = {}
    particles = []
    
    iteration_number = 0
    while True:
        add_particle(particles, start_point_coordinates,
            initializing_face, polyhedron)
        for particle in particles:
            particle.eat(get_map_dot(
                particle.coords,
                simulation_map
                ))
            smelled = particle.get_sensors_values(
                tuple(get_map_dot(i, simulation_map) for i in \
                    (particle.left_sensor, \
                        particle.central_sensor, \
                        particle.right_sensor)),
                iteration_number
                )
            particle.rotate(smelled)
            particle.move(get_map_dot(particle.coords, \
                simulation_map), iteration_number, polyhedron)

        yield particles, simulation_map

        iteration_number += 1

if __name__ == "__main__":
    simulation = simulate(start_point_coordinates,
        initializing_face, polyhedron)
    visualizer = Visualizer(polyhedron)

    i = 0
    while True:
        particles, simulation_map = next(simulation)
        visualizer.add_frame(particles, simulation_map)
        i += 1
        if i % 100 == 0:
            visualizer.redraw()
