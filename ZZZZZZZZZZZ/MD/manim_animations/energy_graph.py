from manim import *
import numpy as np

class EnergyGraph(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[0, 24, 4],
            y_range=[0, 100, 20],
            x_length=10,
            y_length=6,
            axis_config={"color": BLUE, "include_numbers": False},
        )
        
        # Create simple text labels without LaTeX
        x_label = Text("Time (Hours)", font_size=24)
        x_label.next_to(axes.x_axis, DOWN)
        y_label = Text("Energy (MW)", font_size=24)
        y_label.next_to(axes.y_axis, LEFT).rotate(PI/2)
        
        # Define energy consumption curve
        def energy_func(x):
            return 30 + 40 * np.sin(x * PI / 12 - PI/2) + 10 * np.sin(x * PI / 6)
        
        # Create the graph
        graph = axes.plot(energy_func, color=GREEN, x_range=[0, 24])
        
        # Animate
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(graph), run_time=3)
        
        # Add optimization line
        optimized_func = lambda x: 50 + 20 * np.sin(x * PI / 12 - PI/2)
        optimized_graph = axes.plot(optimized_func, color=RED, x_range=[0, 24])
        
        self.play(Create(optimized_graph))
        
        # Add labels
        original_label = Text("Original", color=GREEN, font_size=24).next_to(graph, UP)
        optimized_label = Text("Optimized", color=RED, font_size=24).next_to(optimized_graph, DOWN)
        
        self.play(Write(original_label), Write(optimized_label))