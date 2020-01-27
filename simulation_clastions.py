import random

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


_zeros = np.zeros(3)


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
            vertices, faces: see the class docs
        """
        assert(type(vertices) == np.ndarray) # To be removed
        assert(type(faces) == np.ndarray and faces.dtype == int) # To be removed

        self.vertices = vertices.astype(float)
        self.faces = faces
        self.edges = set()
        for face in self.faces:
            for i in range(len(face) - 1):
                vertices = sorted(face[i:i+2].tolist())
                self.edges.add(tuple(vertices))
        self.edges = np.asarray(list(self.edges))

EPSILON = 10**(-7)

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
    Returns the point of two lines intersection or 
            None if they are parallel
    Parameters:
        line1, line2 (np.ndarray 2 by 3 of float): two points on line
    Returns:
        np.ndarray of three `float`s: if lines intersect
        NoneType: if lines are parallel
    """
    pointA = line1[0]
    pointB = line2[0]
    direction_vectorC = (line1[1] - line1[0]) / \
            get_distance(line1[1] - line1[0], _zeros)
    direction_vectorD = (line2[1] - line2[0]) / \
            get_distance(line2[1] - line2[0], _zeros)
    vectorE = pointB - pointA
    h = np.cross(direction_vectorD, vectorE)
    k = np.cross(direction_vectorD, direction_vectorC)

    h_zeros_dist = get_distance(h, _zeros)
    k_zeros_dist = get_distance(k, _zeros)

    if (h_zeros_dist == 0 or k_zeros_dist == 0):
        return None
    else:
        l = direction_vectorC * h_zeros_dist / k_zeros_dist
        if (np.dot(h, k) / h_zeros_dist / k_zeros_dist < 90):
            return pointA + l
        else:
            return pointA - l

def is_in_segment(point, segment):
    """
    Checks if the point is in segment
    Parameters:
        point (np.ndarray of three `int`s): the point to check
        segment (np.ndarray 2 by 3 of int): two points defining a segment
    Returns:
        bool: True if point belongs to segment, False otherwise
    """
    return (get_distance(point, segment[1]) + get_distance(point, segment[0]) - \
            get_distance(segment[0], segment[1]) <= EPSILON)


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
        TRAIL_DEPTH (int or float): trail length the agent leaves

        coords (np.ndarray of three `float`s): agent's coordinates
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
    TRAIL_DEPTH = 255
    def __init__(self, coords, angle, face, polyhedron, random_rotate_probability=None):
        """
        Initializing the particle(agent)
        Parameters:
            coords (np.ndarray of three `float`s): coordinates of agent
            angle (int, degrees): direction angle of the agent
            face (np.ndarray of 'float's): vertices defining current agent's face
            polyhedron (Polyhedron): the polyhedron we are running on
            random_rotate_probability (int, default 15): the 1/probability of random
                rotate on each step, for example if `random_rotate_probability` is 20,
                the probability is 1/20 = 0.05
        """
        self.food = 255
        self.coords = np.asarray(coords).astype(float)
        self.face = np.asarray(face)

        #init of central_sensor
        normal = np.cross(polyhedron.vertices[self.face[2]] - \
                          polyhedron.vertices[self.face[0]], \
                          polyhedron.vertices[self.face[1]] - \
                          polyhedron.vertices[self.face[0]])
        normal = normal / get_distance(normal, _zeros)
        radius = polyhedron.vertices[self.face[0]] - self.coords
        radius = self.SENSOR_OFFSET * radius / \
                 get_distance(radius, _zeros)
        self.central_sensor = self._rotate_point_angle(normal, radius, angle)
        
        assert(np.zeros(3).dtype == float) # To be removed
        self.left_sensor = np.zeros(3)
        self.right_sensor = np.zeros(3)
        self._init_sensors_from_center(polyhedron)

        if random_rotate_probability is None:
            random_rotate_probability = 30
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
        angle = np.radians(angle)
        angle_cos = np.cos(angle)
        return (1 - angle_cos)*np.dot(normal, radius)*normal + \
                      angle_cos*radius + \
                      np.sin(angle)*np.cross(normal, radius) + self.coords

    def _init_sensors_from_center(self, polyhedron):
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
        assert(type(sensors_values) == np.ndarray) # To be removed
        # turn right by default:
        heading = self.ROTATION_ANGLE
        if random.randint(1, self.random_rotate_probability) == 1:
            # turn randomly:
            heading = random.randint(-1, 1) * self.ROTATION_ANGLE
        else:
            if sensors_values[1] >= sensors_values[0] and sensors_values[1] >= sensors_values[2]:
                heading = 0
            elif sensors_values[1] < sensors_values[0] and sensors_values[1] < sensors_values[2] \
                 and sensors_values[0] == sensors_values[2]:
                # turn randomly:
                if random.randint(0, 1):
                    heading = -self.ROTATION_ANGLE
            elif sensors_values[0] >= sensors_values[1] and sensors_values[0] >= sensors_values[2]:
                # turn left:
                heading = -self.ROTATION_ANGLE

        normal = np.cross(self.left_sensor - self.coords, self.right_sensor - self.coords)
        normal = normal / get_distance(normal, _zeros)
        self.left_sensor = self._rotate_point_angle(normal, (self.left_sensor - self.coords), heading)
        self.central_sensor = self._rotate_point_angle(normal, (self.central_sensor - self.coords), heading)
        self.right_sensor = self._rotate_point_angle(normal, (self.right_sensor - self.coords), heading)

    def _get_vector_move(self):
        """
        Get the vector to which the agent moves
        Returns:
            np.ndarray of three `int`s: the answer
        """
        return (self.central_sensor - self.coords) / self.SENSOR_OFFSET
        
    def _is_edge_belong_face(self, edge, face):
        """
        Returns whether the edge belongs to face
        Parameters:
            edge (np.ndarray of two `int`s):
            face (np.ndarray of 'float's): vertices defining current agent's face
        Returns:
            bool: True if the edge belongs to face, False otherwise
        """
        return (face==edge[0]).any() and (face==edge[1]).any()

    def _change_face(self, edge, polyhedron):
        """
        Returns another face that edge's vertices belong to
        Parameters:
            edge (np.ndarray of two `int`s): pairs of numbers
                of vertices, connected by edges (0-indexing)
            polyhedron (Polyhedron): the polyhedron we are running on
        """
        for face in polyhedron.faces:
            if (face != self.face).any() and self._is_edge_belong_face(edge, face):
                self.face = face
                break


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
        normal_start = normal_start / get_distance(normal_start, _zeros)
        normal_finish = np.cross(polyhedron.vertices[self.face[1]] - \
                                 polyhedron.vertices[self.face[0]], \
                                 polyhedron.vertices[self.face[2]] - \
                                 polyhedron.vertices[self.face[0]])
        normal_finish = normal_finish / get_distance(normal_finish, _zeros)
        vector_move = vector_move / get_distance(vector_move, _zeros)

        # calculating angle between faces
        phi = np.pi - np.arccos(np.dot(normal_start, normal_finish))
        phi_sin = np.sin(phi)
        # calculating moving vector angle
        alpha = np.arccos(np.dot(vector_move, np.cross(normal_start, normal_finish)))
        faced_vector = (normal_start + normal_finish * np.cos(phi)) * \
                        np.sin(alpha)/phi_sin + \
                       (np.cross(normal_start, normal_finish)) * \
                        np.cos(alpha)/phi_sin
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
                      (1 - get_distance(intersect, self.coords)) / \
                       get_distance(faced_vector, _zeros)
        self.coords = intersect + faced_vector
        self.central_sensor = self.coords + faced_vector * self.SENSOR_OFFSET / \
                                        get_distance(faced_vector, _zeros)
        self._init_sensors_from_center(polyhedron)

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
            if not self._is_edge_belong_face(edge, self.face):
                continue
            # check whether agent will cross the edge line or not
            line1 = np.asarray([polyhedron.vertices[edge[0]], \
                                polyhedron.vertices[edge[1]]])
            line2 = np.asarray([self.coords, self.coords + vector_move])
            intersect = line_intersection(line1, line2)
            if intersect is not None:
                if is_in_segment(intersect, line1) and \
                        is_in_segment(intersect, line2) and \
                        get_distance(intersect, self.coords) > EPSILON:
                    
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
    triangle = Polyhedron(vertices=np.array([[0., 0, 0], [0., 10, 0], [10., 10, 0], [0, 10., 0], 
                                [0, 0, 10.], [10, 0., 10], [10, 10., 10], [0, 10, 10.]]), 
                                faces=np.array([[0, 1, 2, 3], [0, 4, 5, 1], [4, 7, 6, 5], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]))
    face = triangle.faces[0]
    part = Particle([9.5, 5, 0], [14.5, 5, 0], face, triangle)
    part._init_sensors_from_center(triangle)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for i in range(0, 2):
        part.simple_visualizing(ax)
        
    plt.show()
