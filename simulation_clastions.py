import random

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Polyhedron():
    """
    A polyhedron for simulation
    Note: the polyhedron must be convex
    Attributes:
        vertices (np.ndarray V by 3): the polyhedron
            vertices' coordinates
        edges (np.ndarray E by 2 of int): pairs of numbers
            of vertices, connected by edges (0-indexing)
        faces (two-dimentional np.ndarray of int): the array
            of faces, where each face is given as an array of
            numbers of vertices on the face (vertices are 0-indexed).
            Note: list the vertices clockwise watching
                outside the polyhedron for each face
    """
    def __init__(self, vertices, faces):
        """
        Initializes a polyhedron
        Parameters:
            vertices, faces: see the class attributes :)
        """
        self.vertices = vertices
        self.faces = faces
        self.edges = set()
        for face in self.faces:
            for i in range(len(face) - 1):
                vertices = sorted(face[i:i+2].tolist())
                self.edges.add(tuple(vertices))
        self.edges = np.asarray(list(self.edges))


def transmission_matrix(face, polyhedron):
    """
    Calculation of transmission matrix for particle
    Parameters:
        face (one-dimentional np.ndarray): vertices defining a face
        polyhedron (Polyhedron): polyhedron we are running on
    Returns:
        np.ndarray 3 by 3 of float: transmission matrix
    """
    C = np.zeros((3, 3))
    verticeA = np.array(polyhedron.vertices[face[0]])
    verticeB = np.array(polyhedron.vertices[face[1]])
    verticeC = np.array(polyhedron.vertices[face[2]])
    C[:, 0] = verticeB - verticeA
    C[:, 1] = verticeC - verticeA
    C[0, 2] = C[1, 0]*C[2, 1] - C[2, 0]*C[1, 1]
    C[1, 2] = C[2, 0]*C[0, 1] - C[0, 0]*C[2, 1]
    C[2, 2] = C[0, 0]*C[1, 1] - C[1, 0]*C[0, 1]
    return C


def space_to_face(point, origin, trans_matrix):
    """
    Returns coordinates of the point, relative to origin in its surface
    Parameters:
        point (np.ndarray of three `float`s): the point's coordinates
        origin (np.ndarray of three `float`s): the origin's coordinates
        trans_matrix (np.ndarray 3 by 3 of float): transmission matrix
    Returns:
        np.ndarray of three `float`s: the answer
    """
    delta_p = point - origin
    return trans_matrix @ delta_p

def face_to_space(point, trans_matrix):
    """
    Returns a vector of P relative to origin in base space
    Parameters:
        point (np.ndarray of three `float`s): the point's coordinates
        trans_matrix (np.ndarray 3 by 3 of float): transmission matrix
    Returns:
        np.ndarray of three `float`s: the answer
    """
    C = np.linalg.inv(trans_matrix)
    return C @ point

def get_distance(a, b):
    """
    Get distance between two points in space
    Parameters:
        a, b (np.ndarray of three `float`s): two point's coordinates
    Returns:
        float: the answer
    """
    return np.sqrt(np.sum((a - b)**2))

def line_intersection(line1, line2):
    """
    Returns the point of two lines intersection, if they are parallel
    returns None
    Parameters:
        line1, line2 (np.ndarray 2 by 3 of int): two points on line
    Returns:
        np.ndarray of three `int`s: if they intersect
        NoneType: if they are parallel
    """
    s = np.vstack([line1[:, :2], line2[:, :2]])       # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return None
    return np.array([x/z, y/z, 0])

def is_in_segment(point, segment):
    """
    Checks if the point is in segment
    Parameters:
        point (np.ndarray of three `int`s): the point to check
        segment (np.ndarray 2 by 3 of int): two points defining a segment
    Returns:
        True if point belongs to segment
        False if point does not belong to segment
    """
    if (point[0] >= min(segment[0, 0], segment[1, 0]) and \
        point[0] <= max(segment[0, 0], segment[1, 0])) and \
       (point[1] >= min(segment[0, 1], segment[1, 1]) and \
        point[1] <= max(segment[0, 1], segment[1, 1])) and \
       (point[2] >= min(segment[0, 2], segment[1, 2]) and \
        point[2] <= max(segment[0, 2], segment[1, 2])):
        return True
    else:
        return False


class TrailDot():
    """
    A smallest section of a trail
    Attributes:
        set_moment (int or float): The simulation iteration number,
            when the trail dot was created. Note: it's either
            integer, or `float('-inf')`
    """
    def __init__(self, set_moment):
        self.set_moment = set_moment

class MapDot():
    """
    A dot in a model map: how much food does
    the dot contain, when the TrailDot was set
    Attributes:
        food (int): the number of food in this dot
        trail (TrailDot): the trail dot in this point
    """
    def __init__(self, food=0):
        self.food = food
        self.trail = TrailDot(float('-inf'))

    def __repr__(self):
        return f'<MapDot with food={self.food} and trail, ' \
            f'set at {self.trail.set_moment}>'


class Particle():
    """
    A part of a mold in the model
    Particle is also known as agent
    Attributes:
        SENSOR_ANGLE (int or float, degrees): the angle between the neighboring sensors
        ROTATION_ANGLE (int or float, degrees): angle the particle rotates at
        SENSOR_OFFSET: (int or float) distance from agent to sensor
        STEP_SIZE (int or float): agent's step size
        TRAIL_DEPTH (int or float): trail length the agent leaves

        coords (np.ndarray of three `float`s): agent's coordinates
        transmission_matrix (np.ndarray 3 by 3): transmission matrix for current face
        left_sensor (np.ndarray of three `float`s): the left sensor's coordinates
        central_sensor (np.ndarray of three `float`s): the central sensor's coordinates
        right_sensor (np.ndarray of three `float`s): the right sensor's coordinates
        face (np.ndarray of 'int's): numbres of vertices defining current agent's face

        random_rotate_probability (int): the 1/probability of random rotate on each step,
            for example if `random_rotate_probability` is 20, the probability is 1/20 = 0.05
    """
    SENSOR_ANGLE = 45
    ROTATION_ANGLE = 20
    SENSOR_OFFSET = 5
    STEP_SIZE = 1
    TRAIL_DEPTH = 255
    def __init__(self, coords, central_sensor, face, polyhedron, random_rotate_probability=None):
        """
        Initializing the particle(agent)
        Parameters:
            coords (np.ndarray of three `float`s): coordinates of agent
            central_sensor (np.ndarray of three `float`s): central sensor's coordinates
            face (np.ndarray of 'float's): vertices defining current agent's face
            polyhedron (Polyhedron): the polyhedron we are running on
            random_rotate_probability (int, default 15): the 1/probability of random
                rotate on each step, for example if `random_rotate_probability` is 20,
                the probability is 1/20 = 0.05
        """
        self.food = 255
        self.coords = np.asarray(coords).astype(float)
        self.trans_matrix = np.asarray(transmission_matrix(face, polyhedron))
        self.face = np.asarray(face)

        self.left_sensor = np.zeros(3)
        self.central_sensor = np.asarray(central_sensor)
        self.right_sensor = np.zeros(3)

        if random_rotate_probability is None:
            random_rotate_probability = 15
        self.random_rotate_probability = random_rotate_probability

    def __repr__(self):
        return f'<Particle with coords={tuple(self.coords.tolist())} ' \
            f'and food={self.food}>'

    def eat(self, map_dot):
        """
        Eat food on agent's coord
        Parameters:
            map_dot (MapDot): the dot on the map self is standing on
        """
        if map_dot.food > 0:
            map_dot.food -= 1
            self.food += 1

    def _rotate_point_angle(self, normal, radius, angle):
        """
        Rotates point with radius vector relative to agent's coordinates at the angle
        Parameters:
            normal (np.ndarray of three `float`s): perpendicular to the diven face of the agent's face
            radius (np.ndarray of three `float`s): vector from agent's coordinates to point
            angle (int or float, degrees): angle the particle rotates
        Returns:
            np.ndarray of three `float`s: new point's coordinates
        """
        return (1 - np.cos(np.radians(angle)))*np.dot(normal, radius)*normal + \
                      np.cos(np.radians(angle))*radius + \
                      np.sin(np.radians(angle))*np.cross(normal, radius) + self.coords

    def init_sensors_from_center(self, polyhedron):
        """
        Initializing left_sensor and right_sensor after full init using central_sensor
        Parameters:
            polyhedron (Polyhedron): the polyhedron we are running on
        """
        normal = np.cross(polyhedron.vertices[self.face[2]] - polyhedron.vertices[self.face[0]], \
                    polyhedron.vertices[self.face[1]] - polyhedron.vertices[self.face[0]])
        normal = normal / get_distance(normal, np.zeros((3)))
        radius = self.central_sensor - self.coords
        self.left_sensor = self._rotate_point_angle(normal, radius, self.SENSOR_ANGLE)
        self.right_sensor = self._rotate_point_angle(normal, radius, -self.SENSOR_ANGLE)

        if np.dot(normal, np.cross(self.right_sensor - self.coords, self.left_sensor - self.coords)) < 0:
            self.left_sensor, self.right_sensor = self.right_sensor, self.left_sensor

    def get_sensors_values(self, sensors_map_dots, iteration):
        """
        Get food and trail sum on each sensor
        Parameters:
            sensors_map_dots (tuple of three MapDot): map dots of sensors
            iteration (int): current simulation iteration number
        Returns:
            np.ndarray of three `int`s: the answer
        """
        trail_under_sensor = np.zeros(3)
        sensors_values = np.zeros(3)
        for i in range(3):
            if iteration - sensors_map_dots[i].trail.set_moment <= self.TRAIL_DEPTH:
                trail_under_sensor[i] = self.TRAIL_DEPTH + \
                    sensors_map_dots[i].trail.set_moment - iteration
            sensors_values[i] = sensors_map_dots[i].food + trail_under_sensor[i]
        return sensors_values

    def rotate(self, sensors_values):
        """
        Rotates the particle and its sensors at the rotation angle
        Parameters:
            sensors_values (np.ndarray of three `int`s): food and trail sum of each sensors
        """
        sensors_values = np.asarray(sensors_values)
        heading = None
        if random.randint(1, self.random_rotate_probability) == 1:
            # turn randomly
            heading = random.randint(-1, 1) * self.ROTATION_ANGLE
        else:
            if sensors_values[1] >= sensors_values[0] and sensors_values[1] >= sensors_values[2]:
                heading = 0
            elif sensors_values[1] < sensors_values[0] and sensors_values[1] < sensors_values[2] \
                 and sensors_values[0] == sensors_values[2]:
                # turn randomly
                r = random.randint(0, 1)
                if r == 0:
                    heading = -self.ROTATION_ANGLE
                else:
                    heading = self.ROTATION_ANGLE
            elif sensors_values[0] >= sensors_values[1] and sensors_values[0] >= sensors_values[2]:
                # turn left
                heading = -self.ROTATION_ANGLE
            elif sensors_values[2] >= sensors_values[1] and sensors_values[2] >= sensors_values[0]:
                # turn right
                heading = self.ROTATION_ANGLE

        normal = np.cross(self.left_sensor - self.coords, self.right_sensor - self.coords)
        normal = normal / get_distance(normal, np.zeros((1, 3)))
        radius = self.left_sensor - self.coords
        self.left_sensor = self._rotate_point_angle(normal, radius, heading)
        radius = self.central_sensor - self.coords
        self.central_sensor = self._rotate_point_angle(normal, radius, heading)
        radius = self.right_sensor - self.coords
        self.right_sensor = self._rotate_point_angle(normal, radius, heading)

    def _get_vector_move(self):
        """
        Get the vector to which the agent moves
        Returns:
            np.ndarray of three `int`s: the answer
        """
        return self.STEP_SIZE * (self.central_sensor - self.coords) / self.SENSOR_OFFSET

    def _change_face(self, edge, polyhedron):
        """
        Returns another face that edge's vertices belong to
        Parameters:
            edge (np.ndarray of two `int`s): pairs of numbers
                of vertices, connected by edges (0-indexing)
            polyhedron (Polyhedron): the polyhedron we are running on
        """
        for face in polyhedron.faces:
            if (face != self.face).any() and len(np.argwhere(face==edge[0])) > 0 \
                                 and len(np.argwhere(face==edge[1])) > 0:
                self.face = face
                break
        self.trans_matrix = transmission_matrix(self.face, polyhedron)

    def _count_moving_vector_through_edge(self, vector_move, polyhedron):
        """
        Count moving vector's direction relative to face after crossing the edge
                                                    (it's self.face after changing)
        Parameters:
            vector_move (np.ndarray of three `int`s): vector the agent moves
            polyhedron (Polyhedron): the polyhedron we are running on
        Returns:
            np.ndarray of three `int`s: the answer
        """
        normal_start = np.cross(self.left_sensor - self.coords, \
                                self.right_sensor - self.coords)
        normal_start = normal_start / get_distance(normal_start, np.zeros(3))
        normal_finish = np.cross(polyhedron.vertices[self.face[1]] - \
                                 polyhedron.vertices[self.face[0]], \
                                 polyhedron.vertices[self.face[2]] - \
                                 polyhedron.vertices[self.face[0]])
        normal_finish = normal_finish / get_distance(normal_finish, np.zeros(3))
        vector_move = vector_move / get_distance(vector_move, np.zeros(3))

        # counting angle between faces
        phi = np.arccos(np.dot(normal_start, normal_finish) / \
                    get_distance(normal_start, np.zeros(3)) / \
                    get_distance(normal_finish, np.zeros(3)))
        # counting moving vector angle
        alpha = np.arccos(np.dot(vector_move, np.cross(normal_start, normal_finish)))
        faced_vector = (normal_start + normal_finish * np.cos(phi)) * \
                        np.sin(alpha)/np.sin(phi) + \
                       (np.cross(normal_start, normal_finish)) * \
                        np.cos(alpha)/np.sin(phi)
        return faced_vector

    def _move_throught_edge(self, vector_move, edge, intersect, polyhedron):
        """
        Change agent's coordinates to another face
        Parameters:
            vector_move (np.ndarray of three `int`s): vector the agent moves
            edge (np.ndarray of two `int`s):
            intersect (np.ndarray of three `int`s):
            polyhedron (Polyhedron): the polyhedron we are running on
        """
        self._change_face(edge, polyhedron)
        # self.face and self.trans_matrix changed
        faced_vector = self._count_moving_vector_through_edge(vector_move, polyhedron)
        faced_vector = faced_vector * \
                      (self.STEP_SIZE - get_distance(intersect, np.zeros(3))) / \
                       get_distance(faced_vector, np.zeros(3))
        self.coords = self.coords + intersect + faced_vector
        self.central_sensor = self.coords + faced_vector * self.SENSOR_OFFSET / \
                                        get_distance(faced_vector, np.zeros(3))
        self.init_sensors_from_center(polyhedron)

    def _move_step_size(self, vector_move):
        """
        Moves the particle forward on step size
        Parameters:
            vector_move (np.ndarray of three `int`s): vector the agent moves
        """
        self.coords += vector_move
        self.central_sensor += vector_move
        self.left_sensor += vector_move
        self.right_sensor += vector_move

    def move(self, map_dot, iteration, polyhedron):
        """
        Moves the particle forward on step size
        Parameters:
            map_dot (MapDot): the map dot I am moving FROM (!!!)
            iteration (int): current simulation iteration number
            polyhedron (Polyhedron): the polyhedron we are running on
        """
        self.food -= 1 # Lose my energy when moving
        map_dot.trail.set_moment = iteration
        vector_move = self._get_vector_move()

        for edge in polyhedron.edges:
            # check whether agent will cross the edge or not
            line1 = np.asarray([space_to_face(polyhedron.vertices[edge[0]], \
                                    self.coords, self.trans_matrix), \
                                space_to_face(polyhedron.vertices[edge[1]], \
                                    self.coords, self.trans_matrix)])
            line2 = np.asarray([space_to_face(self.coords, \
                                    self.coords, self.trans_matrix), \
                                space_to_face(self.coords + vector_move, \
                                    self.coords, self.trans_matrix)])
            intersect = line_intersection(line1, line2)
            if intersect is not None:
                if is_in_segment(intersect, line1) and \
                        is_in_segment(intersect, line2):
                    intersect = face_to_space(intersect, self.trans_matrix)
                    self._move_throught_edge(vector_move, edge, intersect, polyhedron)
                    return

        self._move_step_size(vector_move)

    def simple_visualizing(self, ax):
        left_sensor = np.round(self.left_sensor)
        central_sensor = np.round(self.central_sensor)
        right_sensor = np.round(self.right_sensor)
        coords = np.round(self.coords)

        ax.scatter3D(xs=coords[0], ys=coords[1], zs=coords[2], color='black')
        ax.scatter3D(xs=central_sensor[0], ys=central_sensor[1], zs=central_sensor[2], color='black')
        ax.scatter3D(xs=left_sensor[0], ys=left_sensor[1], zs=left_sensor[2], color='red')
        ax.scatter3D(xs=right_sensor[0], ys=right_sensor[1], zs=right_sensor[2], color='green')
        ax.plot3D([coords[0], central_sensor[0]], [coords[1], central_sensor[1]], [coords[2], central_sensor[2]], color='black')
        ax.plot3D([coords[0], left_sensor[0]], [coords[1], left_sensor[1]], [coords[2], left_sensor[2]], color='red')
        ax.plot3D([coords[0], right_sensor[0]], [coords[1], right_sensor[1]], [coords[2], right_sensor[2]], color='green')


if __name__ == "__main__":
    triangle = Polyhedron(vertices=np.array([[0., 0, 0], [0., 10, 0], [10., 0, 0]]), edges=[], faces=[0, 1, 2])
    surface = [0, 1, 2]
    part = Particle([0., 0, 0], [5., 0, 0], surface, triangle)
    part.init_sensors_from_center(triangle)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for i in range(0, 5):
        part.simple_visualizing(ax)
        part.rotate(np.array([10., 0, 0]))
    plt.show()
