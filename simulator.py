from random import randint
from multiprocessing import cpu_count
from threading import Thread, Lock

import numpy as np

from simulation_clastions import Polyhedron, TrailDot, \
    MapDot, Particle, get_distance
from visualizer import Visualizer


def chunkify(l, n_of_chunks):
    """Splits the given list into parts of approximately
    equal length
    Parameters:
        l (list): The list to be splited
        n_of_chunks (int): Number of chunks to ger
    Yielda:
        list: The chunks
    """
    chunk_length = len(l) // n_of_chunks
    for i in range(n_of_chunks - 1):
        yield l[i*chunk_length:(i+1)*chunk_length]
    yield l[(n_of_chunks-1)*chunk_length:]


class Simulator:
    """
    The object for simulation Physarum growing
    Attributes:
        polyhedron (Polyhedron): the polyhedron to initialize new particles
            on
        initializing_face (one-dimensional np.ndarray of `int`s):
            the polyhedron's face to initialize new particles on
            (presented as an array of face's vertices' numbers, 0-indexing)
        start_point_coordinates (np.ndarray of three `int`s):the coordinates
            to initialize new particles at

        particles (list of `Particle`s): particles in the simulation
        simulation_map (dict from tuple of three `int`s to MapDot):
            the simulation map
        iteration (int): the number of current simulation iteration

        map_edit_lock (threading.Lock): the mutex for editing the simulation map

        particles_per_iteration (int): the number of new particles to be created
            on each simulation iteration
        particle_random_rotate_probability (int): the 1/probability of random
            rotate on each step, for example if `random_rotate_probability` is 20,
            the probability is 1/20 = 0.05
        n_of_threads (int): the maximum number of threads to run during a step
    """
    def __init__(self, polyhedron, initializing_face, start_point_coordinates,
                n_of_new_particles_per_iteration=None,
                particle_random_rotate_probability=None, n_of_threads=None):
        """
        Initializes a Simulator object
        Parameters:
            polyhedron (Polyhedron): the polyhedron to initialize new particles
                on
            initializing_face, start_point_coordinates,
                n_of_new_particles_per_iteration, particle_random_rotate_probability,
                n_of_threads: see the class attributes
        """
        self.polyhedron = polyhedron
        self.initializing_face = initializing_face
        self.start_point_coordinates = start_point_coordinates

        self.particles = []
        self.simulation_map = {}
        self.iteration = 0
        self.map_edit_lock = Lock()

        if n_of_new_particles_per_iteration is None:
            self.particles_per_iteration = 10
        else:
            self.particles_per_iteration = n_of_new_particles_per_iteration

        if particle_random_rotate_probability is None:
            self.particle_random_rotate_probability = 15
        else:
            self.particle_random_rotate_probability = \
                    particle_random_rotate_probability

        if n_of_threads is None:
            self.n_of_threads = cpu_count()
        else:
            self.n_of_threads = n_of_threads

    def get_map_dot(self, coords):
        """
        Returns the map dot info
        Parameters:
            coords (np.ndarray of three `int`s): Coordinates of the dot
        Returns:
            MapDot: the information about the dot
        """
        coords = np.round(np.asarray(coords)).astype(int).tolist()
        coords = tuple(coords)
        if coords not in self.simulation_map.keys():
            self.simulation_map[coords] = MapDot()
        return self.simulation_map[coords]


    def add_particles(self):
        """
        Adds `particles_per_iteration` particles to the simulation
        """
        for _ in range(self.particles_per_iteration):
            angle = randint(1, 360)

            new_point = Particle(
                self.start_point_coordinates,
                angle,
                self.initializing_face,
                self.polyhedron,
                self.particle_random_rotate_probability
                )
            self.particles.append(new_point)


    def _process_particles(self, particles):
        """
        Process the given group of particles, using the simulation rules
        Parameters:
            particles (list of `Particle`s): the particles to be processed
        """
        for particle in particles:
            with self.map_edit_lock:
                particle.eat(self.get_map_dot(particle.coords))
            smelled = particle.get_sensors_values(
                tuple(self.get_map_dot(i) for i in (particle.left_sensor,
                    particle.central_sensor, particle.right_sensor)),
                self.iteration
                )
            particle.rotate(smelled)
            particle.move(self.get_map_dot(particle.coords), self.iteration,
                self.polyhedron)


    def step(self):
        """
        Executes one simulation step
        """
        self.add_particles()
        splited_particles = chunkify(self.particles, self.n_of_threads)
        threads = []
        for particles_chunk in splited_particles:
            threads.append(Thread(target=self._process_particles,
                args=(particles_chunk,)))
            threads[-1].start()
        for thread in threads:
            thread.join()

        self.iteration += 1


if __name__ == "__main__":
    from sys import stdout

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

    simulator = Simulator(polyhedron, initializing_face,
        start_point_coordinates,
        int(input("How many particles to create on each step (int): ")),
        int(input("particle_random_rotate_probability (int): ")))
    visualizer = Visualizer(polyhedron, size=3)

    frames_adding_frequency = int(input("Frames frequency adding (int): "))
    redraw_frequency = int(input("Redraw frequency (int): "))
    while True:
        print(simulator.iteration, end=' ')
        stdout.flush()
        if simulator.iteration % frames_adding_frequency == 0:
            visualizer.add_frame(simulator.particles, simulator.simulation_map)
        if(simulator.iteration % redraw_frequency == 0) and \
                (simulator.iteration != 0):
            visualizer.redraw()

        simulator.step()
