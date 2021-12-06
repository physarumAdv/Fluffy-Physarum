import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


class Frame():
    def __init__(self, particles):
        self.x = np.ndarray(0)
        self.y = np.ndarray(0)
        self.z = np.ndarray(0)

        if len(particles):
            coords = list(map(lambda p: p.coords, particles))
            coords = np.asarray(coords)
            self.x, self.y, self.z = coords.swapaxes(0, 1)


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
            colorscale=((0, 'grey'),
                        (0.5, 'mediumturquoise'),
                        (1, 'magenta')),
            intensity=np.random.rand(len(polyhedron.vertices)),
            i=i,
            j=j,
            k=k,
            opacity=0.7,
            name='y',
            showscale=True)

    def add_frame(self, particles, simulation_map):  # TODO add simulation_map
        self.frames.append(Frame(particles))

    def create(self, i):
        frame = self.frames[i]
        return (self.poly, go.Scatter3d(
                x=frame.x,
                y=frame.y,
                z=frame.z,
                mode='markers',
                marker=dict(size=self.size, color='yellow')))

    def redraw(self):
        fig = go.Figure(data=self.create(0),
            layout=go.Layout(
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(range=(0, 5), autorange=False),
                yaxis=dict(range=(0, 5), autorange=False),
            title="Physarum Polycephalum",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="grow",
                            method="animate",
                            args=[None, {"frame": {"duration": 0}}]
                            )])]),
                frames=[go.Frame(data=self.create(i)) for i in range(len(self.frames))])

        fig.layout.plot_bgcolor = '#DCDCDC'
        fig.layout.paper_bgcolor = '#fff'

        fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(showticklabels=False),
            )
        )

        fig.update_scenes(xaxis_title_text="")
        fig.update_scenes(yaxis_title_text="")
        fig.update_scenes(zaxis_title_text="")
        
        fig.show()
