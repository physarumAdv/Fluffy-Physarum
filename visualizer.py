import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


class Frame():
    def __init__(self, particles):
        self.x = np.asarray([])
        self.y = np.asarray([])
        self.z = np.asarray([])
        for particle in particles:
            a_x, a_y, a_z = particle.coords
            self.x = np.append(self.x, a_x)
            self.y = np.append(self.y, a_y)
            self.z = np.append(self.z, a_z)


class Visualizer():
    def __init__(self, polyhedron, size=6):
        self.frames = []
        self.size = size
        vx, vy, vz = np.rot90(polyhedron.vertices)[::-1]
        i, j, k = [], [], []
        
        for face in polyhedron.faces:
            for a in range(1, len(face) - 1):
                i.append(face[0])
                j.append(face[a])
                k.append(face[a + 1])

        self.poly = go.Mesh3d(
            x=vx,
            y=vy,
            z=vz,
            colorbar_title='z',
            colorscale=[[0, 'grey'],
                        [0.5, 'mediumturquoise'],
                        [1, 'magenta']],
            intensity=np.random.rand(len(polyhedron.vertices)),
            i=i,
            j=j,
            k=k,
            opacity=0.4,
            name='y',
            showscale=True)

    def add_frame(self, particles, simulation_map):  # TODO add simulation_map
        self.frames.append(Frame(particles))

    def create(self, i):
        frame = self.frames[i]
        return [self.poly, go.Scatter3d(
                x=frame.x,
                y=frame.y,
                z=frame.z,
                mode='markers',
                marker=dict(size=self.size, color='yellow'))]
    
    def redraw(self):
        fig = go.Figure(data=self.create(0),
            layout=go.Layout(
                xaxis=dict(range=[0, 5], autorange=False),
                yaxis=dict(range=[0, 5], autorange=False),
            title="Physarum Polycephalum",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="grow",
                            method="animate",
                            args=[None])])]),
                frames=[go.Frame(data=self.create(i)) for i in range(len(self.frames))])
        fig.show()
        

#--------------#
if __name__ == '__main__':
    from simulation_clastions import Particle, Polyhedron

    polyhedron = Polyhedron(
        vertices=np.asarray([[0, 0, 0], [1, 0, 2], [2, 1, 0], [0, 2, 1]]),
        faces=np.asarray([[0, 1, 2], [2, 1, 3], [3, 1, 0], [0, 2, 3]]))
    
    visualizer = Visualizer(polyhedron)  # init
    p1 = Particle(np.asarray([1, 1, 1]), [], [], polyhedron)
    p2 = Particle(np.asarray([2, 2, 2]), [], [], polyhedron)
    p3 = Particle(np.asarray([0, 2, 2]), [], [], polyhedron)
    visualizer.add_frame([p1], {})  # add one frame
    visualizer.add_frame([p1, p2], {})
    visualizer.add_frame([p1, p2, p3], {})
    
    '''
    for i in range(10):  # random test
        visualiser.add_frame(np.random.rand(5, 3))  # add one frame
    '''
        
    visualizer.redraw()  # open window
