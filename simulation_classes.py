import random, math

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Polyhedron():
    def __init__(self, vertices, edges, faces):
        self.vertices = vertices
        self.faces = faces
        self.edges = edges


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
    verA = np.array(polyhedron.vertices[face[0]])
    verB = np.array(polyhedron.vertices[face[1]])
    verC = np.array(polyhedron.vertices[face[2]])
    C[:, 0] = verB - verA
    C[:, 1] = verC - verA
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
        polyhedron (Polyhedron): the polyhedron we are running on
    """
    SENSOR_ANGLE = 45
    ROTATION_ANGLE = 20
    SENSOR_OFFSET = 0.001
    STEP_SIZE = 0.001
    TRAIL_DEPTH = 5
    def __init__(self, coords, central_sensor, face, polyhedron):
        """
        Initializing the particle(agent)
        Parameters:
            coords (np.ndarray of three `int`s): coordinates of agent
            face (np.ndarray of 'int's): vertices defining current agent's face

        """
        self.food = 255
        self.coords = np.asarray(coords)
        self.trail = np.zeros((self.depT, 3))
        self.trans_matrix = np.array(transmission_matrix(face, polyhedron))

        self.left_sensor = 0
        self.central_sensor = central_sensor
        self.right_sensor = 0


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

    def eat(self, map_dot): # rewrite
        """
        Eat food on agent's coord
        Parameters:
            map_dot (MapDot): the dot on the map self is standing on
        """
        if FoodMap[self.coords[0], self.coords[1], self.coords[2]] > 0:
            FoodMap[self.coords[0], self.coords[1], self.coords[2]] -= 1
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
        Initializing lelf_sensor and right_sensor after full init using central_sensor
        Parameters:
            polyhedron (Polyhedron): the polyhedron we are running on
        """
        normal = np.cross(polyhedron.vertices[self.face[1]] - polyhedron.vertices[self.face[0]], \
                     polyhedron.vertices[self.face[2]] - polyhedron.vertices[self.face[0]])
        normal = normal/get_distance(normal, np.zeros((1, 3)))
        radius = self.central_sensor - self.coords
        self.left_sensor = self._rotate_point_angle(normal, radius, self.SENSOR_ANGLE)
        self.right_sensor = self._rotate_point_angle(normal, radius, -self.SENSOR_ANGLE)

        if np.dot(normal, np.cross(self.right_sensor, self.left_sensor)) > 0:
            self.left_sensor, self.right_sensor = self.right_sensor, self.left_sensor

    def get_sensors_values(self, sensors_map_dots): # rewrite
        """
        Get food and trail sum on each sensor
        Parameters:
            sensors_map_dots (np.ndarray of three MapDot): map dots of sensors
        Returns:
            np.ndarray of three `int`s: the answer
        """
        trail_map_dim = TrailMap.shape[0]
        ls = np.array([0, 0, 0])
        cs = np.array([0, 0, 0])
        rs = np.array([0, 0, 0])
        for i in range(0, 3):
            ls[i] = int((self.left_sensor[i] + 1)*(trail_map_dim-1) / 2)
            cs[i] = int((self.central_sensor[i] + 1)*(trail_map_dim-1) / 2)
            rs[i] = int((self.right_sensor[i] + 1)*(trail_map_dim-1) / 2)
        return np.array([[TrailMap[ls[0], ls[1], ls[2]], \
                          TrailMap[cs[0], cs[1], cs[2]], \
                          TrailMap[rs[0], rs[1], rs[2]]],
                         [FoodMap[ls[0], ls[1], ls[2]], \
                          FoodMap[cs[0], cs[1], cs[2]], \
                          FoodMap[rs[0], rs[1], rs[2]]]])

    def rotate(self, sensors_values): # rewrite
        """
        Rotates the particle and its sensors at the rotation angle
        Parameters:
            sensors_values (np.ndarray of three `int`s): food and trail sum of each sensors
        """
        heading = 0
        #turn according to food
        if sense[1, 0] > 0 or sense[1, 2] > 0:
            if sense[1, 0] > sense[1, 2]:
                # turn left
                heading += 5
            else:
                # turn right
                heading -= 5

        rand = random.randint(0, 15)
        if rand == 1:
            # turn randomly
            r = random.randint(0, 1)
            if r == 0:
                heading += self.RA
            else:
                heading -= self.RA
        else:
            if sense[0, 1] >= sense[0, 0] and sense[0, 1] >= sense[0, 2]:
                pass
            elif sense[0, 1] < sense[0, 0] and sense[0, 1] < sense[0, 2]:
                # turn randomly
                r = random.randint(0, 1)
                if r == 0:
                    heading += self.RA
                else:
                    heading -= self.RA
            elif sense[0, 0] >= sense[0, 1] and sense[0, 0] >= sense[0, 2]:
                # turn left
                heading += self.RA
            elif sense[0, 2] >= sense[0, 1] and sense[0, 2] >= sense[0, 0]:
                # turn right
                heading -= self.RA

        n = np.cross(self.left_sensor - self.coord, self.right_sensor - self.coord)
        n = n / get_distance(n, np.zeros((1, 3)))
        p = self.left_sensor - self.coord
        self.left_sensor = self.rotate_point_angle(n, p, heading)
        p = self.central_sensor - self.coord
        self.central_sensor = self.rotate_point_angle(n, p, heading)
        p = self.right_sensor - self.coord
        self.right_sensor = self.rotate_point_angle(n, p, heading)

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
        ax.scatter3D(xs=self.coords[0], ys=self.coords[1], zs=self.coords[2], color='black')
        ax.scatter3D(xs=self.central_sensor[0], ys=self.central_sensor[1], zs=self.central_sensor[2], color='black')
        ax.scatter3D(xs=self.left_sensor[0], ys=self.left_sensor[1], zs=self.left_sensor[2], color='red')
        ax.scatter3D(xs=self.right_sensor[0], ys=self.right_sensor[1], zs=self.right_sensor[2], color='green')
        ax.plot3D([self.coords[0], self.central_sensor[0]], [self.coords[1], self.central_sensor[1]], [self.coords[2], self.central_sensor[2]], color='black')
        ax.plot3D([self.coords[0], self.left_sensor[0]], [self.coords[1], self.left_sensor[1]], [self.coords[2], self.left_sensor[2]], color='red')
        ax.plot3D([self.coords[0], self.right_sensor[0]], [self.coords[1], self.right_sensor[1]], [self.coords[2], self.right_sensor[2]], color='green')
        self.move_all_agent_coordinates()
        #sleep(0.1)


if __name__ == "__main__":
    TrailMap = np.zeros((100, 100, 100))
    FoodMap = np.zeros((100, 100, 100))
    triangle = Polyhedron(vertices=np.array([[0, 0, 0], [0, 2., 0], [2., 0, 0]]), edges=[], faces=[0, 1, 2])
    surface = [0, 1, 2]
    part = Particle(0., 0., 0., surface, triangle.vertices, triangle.edges, triangle.faces)
    part.init_sensors_from_center()
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for i in range(0, 5):
        part.simple_visualizing(ax)
        part.rotate_all_sensors(part.sense_trail(TrailMap, FoodMap))
    plt.show()
