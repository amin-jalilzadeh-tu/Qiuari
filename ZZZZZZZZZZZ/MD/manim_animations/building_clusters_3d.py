from manim import *
import numpy as np

class BuildingClusters3D(ThreeDScene):
    def construct(self):
        # Set up 3D view
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            z_range=[0, 3, 1],
            x_length=8,
            y_length=8,
            z_length=4,
        )
        
        # Create building clusters
        buildings = VGroup()
        colors = [RED, GREEN, BLUE, YELLOW]
        
        for i in range(4):
            for j in range(3):
                height = np.random.uniform(0.5, 2.5)
                building = Prism(dimensions=[0.5, 0.5, height])
                building.shift(
                    RIGHT * (i - 1.5) * 2 + 
                    UP * (j - 1) * 2 + 
                    OUT * height/2
                )
                building.set_fill(colors[i], opacity=0.8)
                building.set_stroke(colors[i], width=2)
                buildings.add(building)
        
        # Animate
        self.play(Create(axes))
        self.begin_ambient_camera_rotation(rate=0.2)
        self.play(
            *[Create(building) for building in buildings],
            run_time=3
        )
        self.wait(5)
        self.stop_ambient_camera_rotation()