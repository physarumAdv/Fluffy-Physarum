import random

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Polyhedron():
    """
    A polyhedron for simulation
    Attributes:
        vertices (np.ndarray V by 3 of int): the polyhedron 
            vertices' coordinates
        edges (np.ndarray E by 2 of int): pairs of numbers
            of vertices, connected by edges (0-indexing)
        faces (two-dimentional np.ndarray of int): groups
            of numbers of vertices, connected by one face
            (0-indexing)
    """
    def __init__(self, vertices, edges, faces):
        """
        Initializes a polyhedron
        Parameters:
            see the class attributes :)
        """
        self.vertices = vertices
        self.edges = edges
        self.faces = faces


def transmission_matrix(face, polyhedron):
    """
    Calculation of transmission matrix for particle
    Parameters:
        surface (one-dimentional np.ndarray): vertices defining a face
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



def get_distance(a, b):
    """
    Get distance between two points in space
    Parameters:
        a, b (np.ndarray of three `int`s): two point's coordinates
    Returns:
        float: the answer 
    """
    return np.sqrt(np.sum((a - b)**2))


class Particle():
    """
    An object describing a part of a mold in the model
    Particle is also known as agent
    Attributes:
        SENSOR_ANGLE (int, degrees): the angle between the neighboring sensors
        ROTATION_ANGLE (int, degrees): angle the particle rotates at
        SENSOR_OFFSET: (int) distance from agent to sensor
        STEP_SIZE (int): agent's step size
        TRAIL_DEPTH (int): trail length the agent leaves

        coords (np.ndarray of three `int`s): agent's coordinates
        transmission_matrix (np.ndarray 3 by 3): transmission matrix for current face
        left_sensor (np.ndarray of three `int`s): the left sensor's coordinates
        central_sensor (np.ndarray of three `int`s): the central sensor's coordinates
        right_sensor (np.ndarray of three `int`s): the right sensor's coordinates
        face (np.ndarray of 'int's): vertices defining current agent's face
    """
    SENSOR_ANGLE = 45
    ROTATION_ANGLE = 20
    SENSOR_OFFSET = 5
    STEP_SIZE = 5
    TRAIL_DEPTH = 5
    def __init__(self, coords, central_sensor, face, polyhedron):
        """
        Initializing the particle(agent)
        Parameters:
            coords (np.ndarray of three `int`s): coordinates of agent
            central_sensor (np.ndarray of three `int`s): central sensor's coordinates
            face (np.ndarray of 'int's): vertices defining current agent's face
            polyhedron (Polyhedron): the polyhedron we are running on
        """
        self.food = 255
        self.coords = np.asarray(coords)
        self.trans_matrix = np.array(transmission_matrix(face, polyhedron))
        self.face = face

        self.left_sensor = np.zeros((1, 3))
        self.central_sensor = np.asarray(central_sensor)
        self.right_sensor = np.zeros((1, 3))


    def space_to_face(self, point):
        """
        Returns coordinates of the point, relative to self.coord in self.surface
        Parameters:
            point (np.ndarray of three `int`s): the point's coordinates
        Returns:
            np.ndarray of three `int`s: the answer
        """
        delta_p = point - self.coords
        return self.trans_matrix @ delta_p

    def face_to_space(self, point):
        """
        Returns coordinates of P relative to base space
        Parameters:
            point (np.ndarray of three `int`s): the point's coordinates
        Returns:
            np.ndarray of three `int`s: the answer
        """
        C = np.linalg.inv(self.trans_matrix)
        return C @ point + self.coords

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
            normal (np.ndarray of three `int`s): perpendicular to the diven face of the agent's face
            radius (np.ndarray of three `int`s): vector from agent's coordinates to point
            angle (int, degrees): angle the particle rotates
        Returns:
            np.ndarray of three `int`s: new point's coordinates
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
        normal = np.cross(polyhedron.vertices[self.face[1]] - polyhedron.vertices[self.face[0]], \
                     polyhedron.vertices[self.face[2]] - polyhedron.vertices[self.face[0]])
        normal = normal / get_distance(normal, np.zeros((3)))
        radius = self.central_sensor - self.coords
        self.left_sensor = self._rotate_point_angle(normal, radius, self.SENSOR_ANGLE)
        self.right_sensor = self._rotate_point_angle(normal, radius, -self.SENSOR_ANGLE)

        if np.dot(normal, np.cross(self.right_sensor, self.left_sensor)) > 0:
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
            if iteration - sensors_map_dots[i] <= self.TRAIL_DEPTH:
                trail_under_sensor[i] = self.TRAIL_DEPTH + sensors_map_dots[i] - iteration
            sensors_values[i] = sensors_map_dots[i].food + trail_under_sensor[i]
        return sensors_values

    def rotate(self, sensors_values):
        """
        Rotates the particle and its sensors at the rotation angle
        Parameters:
            sensors_values (np.ndarray of three `int`s): food and trail sum of each sensors
        """
        if random.randint(0, 10) == 0:
            # turn randomly
            heading += random.randint(-1, 1) * self.ROTATION_ANGLE
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

    def move(self):
        """
        Moves the particle forward on step size
        """
        vector_move = self.STEP_SIZE * (self.central_sensor - self.coords) / self.SENSOR_OFFSET
        self.coords += vector_move
        self.central_sensor += vector_move
        self.left_sensor += vector_move
        self.right_sensor += vector_move

    def simple_visualizing(self, ax):
        left_sensor = self.left_sensor.astype(int)
        central_sensor = self.central_sensor.astype(int)
        right_sensor = self.right_sensor.astype(int)
        coords = self.coords.astype(int)
        
        ax.scatter3D(xs=coords[0], ys=coords[1], zs=coords[2], color='black')
        ax.scatter3D(xs=central_sensor[0], ys=central_sensor[1], zs=central_sensor[2], color='black')
        ax.scatter3D(xs=left_sensor[0], ys=left_sensor[1], zs=left_sensor[2], color='red')
        ax.scatter3D(xs=right_sensor[0], ys=right_sensor[1], zs=right_sensor[2], color='green')
        ax.plot3D([coords[0], central_sensor[0]], [coords[1], central_sensor[1]], [coords[2], central_sensor[2]], color='black')
        ax.plot3D([coords[0], left_sensor[0]], [coords[1], left_sensor[1]], [coords[2], left_sensor[2]], color='red')
        ax.plot3D([coords[0], right_sensor[0]], [coords[1], right_sensor[1]], [coords[2], right_sensor[2]], color='green')
        self.move()
        #sleep(0.1)


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
