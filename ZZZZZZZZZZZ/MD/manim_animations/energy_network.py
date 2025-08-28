from manim import *

class EnergyNetwork(Scene):
    def construct(self):
        # Create network nodes
        nodes = VGroup()
        positions = [
            UP * 2,
            UP + LEFT * 2,
            UP + RIGHT * 2,
            DOWN + LEFT * 2,
            DOWN + RIGHT * 2,
            DOWN * 2
        ]
        
        for i, pos in enumerate(positions):
            node = Circle(radius=0.3, color=BLUE_E, fill_opacity=0.8)
            node.move_to(pos)
            node_text = Text(str(i+1), font_size=20).move_to(pos)
            nodes.add(VGroup(node, node_text))
        
        # Create central hub
        hub = Circle(radius=0.5, color=RED, fill_opacity=0.8)
        hub_text = Text("Hub", font_size=20, color=WHITE).move_to(hub)
        
        # Create connections
        edges = VGroup()
        for pos in positions:
            edge = Line(ORIGIN, pos, color=GRAY)
            edges.add(edge)
        
        # Animate network construction
        self.play(Create(hub), Write(hub_text))
        self.play(*[Create(edge) for edge in edges])
        self.play(*[Create(node[0]) for node in nodes])
        self.play(*[Write(node[1]) for node in nodes])
        
        # Show energy flow
        for i in range(3):
            dots = VGroup()
            for pos in positions:
                dot = Dot(color=YELLOW, radius=0.1).move_to(pos)
                dots.add(dot)
            
            self.play(*[
                MoveAlongPath(dot, Line(pos, ORIGIN))
                for dot, pos in zip(dots, positions)
            ], run_time=1.5)
            self.remove(dots)