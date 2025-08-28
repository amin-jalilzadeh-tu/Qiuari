from manim import *
import numpy as np

class EnergyGridNetwork(Scene):
    def construct(self):
        # Title
        title = Text("Energy Grid Network", font_size=48, color=BLUE)
        subtitle = Text("AI-Driven Planning Visualization", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title), Write(subtitle))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Create grid nodes (substations/buildings)
        nodes = []
        node_positions = [
            [-4, 2, 0], [-2, 2, 0], [0, 2, 0], [2, 2, 0], [4, 2, 0],
            [-3, 0, 0], [-1, 0, 0], [1, 0, 0], [3, 0, 0],
            [-4, -2, 0], [-2, -2, 0], [0, -2, 0], [2, -2, 0], [4, -2, 0]
        ]
        
        # Create main substation
        main_substation = Square(side_length=0.8, color=RED, fill_opacity=1)
        main_substation.move_to([0, 0, 0])
        main_label = Text("Substation", font_size=16).next_to(main_substation, DOWN)
        
        self.play(Create(main_substation), Write(main_label))
        
        # Create buildings/nodes
        for i, pos in enumerate(node_positions):
            if pos != [0, 0, 0]:  # Skip center position
                node = Circle(radius=0.3, color=BLUE, fill_opacity=0.8)
                node.move_to(pos)
                nodes.append(node)
        
        self.play(*[Create(node) for node in nodes])
        
        # Create connections
        edges = []
        for node in nodes:
            if np.linalg.norm(node.get_center() - main_substation.get_center()) < 3:
                edge = Line(main_substation.get_center(), node.get_center(), 
                           stroke_width=2, color=YELLOW)
                edges.append(edge)
        
        self.play(*[Create(edge) for edge in edges])
        
        # Animate power flow
        power_dots = []
        for edge in edges:
            dot = Dot(color=GREEN, radius=0.1)
            dot.move_to(edge.get_start())
            power_dots.append(dot)
        
        # Create power flow animation
        self.play(*[Create(dot) for dot in power_dots])
        
        for _ in range(2):
            animations = []
            for dot, edge in zip(power_dots, edges):
                animations.append(
                    MoveAlongPath(dot, edge, rate_func=linear)
                )
            self.play(*animations, run_time=2)
            
            # Reset dots to start
            for dot, edge in zip(power_dots, edges):
                dot.move_to(edge.get_start())
        
        # Add load indicators
        load_text = Text("Peak Load Analysis", font_size=32).to_edge(UP)
        self.play(Write(load_text))
        
        # Highlight overloaded sections
        for i, node in enumerate(nodes[:5]):
            if i % 2 == 0:
                self.play(node.animate.set_color(RED).set_fill_opacity(1))
        
        overload_text = Text("Overloaded Areas", font_size=20, color=RED)
        overload_text.next_to(nodes[0], UP)
        self.play(Write(overload_text))
        
        self.wait(2)
        
        # Show AI optimization
        opt_text = Text("AI Optimization", font_size=32, color=GREEN).to_edge(UP)
        self.play(Transform(load_text, opt_text))
        
        # Add battery storage
        battery = Rectangle(width=0.6, height=0.8, color=GREEN, fill_opacity=1)
        battery.next_to(nodes[2], RIGHT)
        battery_label = Text("Battery", font_size=14).next_to(battery, RIGHT)
        
        self.play(Create(battery), Write(battery_label))
        
        # Show load balancing
        for i, node in enumerate(nodes[:5]):
            if i % 2 == 0:
                self.play(node.animate.set_color(GREEN).set_fill_opacity(0.8))
        
        self.play(FadeOut(overload_text))
        balanced_text = Text("Load Balanced", font_size=20, color=GREEN)
        balanced_text.next_to(nodes[0], UP)
        self.play(Write(balanced_text))
        
        self.wait(2)