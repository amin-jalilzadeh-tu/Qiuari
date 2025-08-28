from manim import *

class DataFlowDiagram(Scene):
    def construct(self):
        # Create nodes
        building = Square(side_length=1.5, color=GREEN).shift(LEFT * 4)
        building_text = Text("Buildings", font_size=20).move_to(building)
        
        ai_model = Circle(radius=1, color=BLUE).shift(ORIGIN)
        ai_text = Text("AI\nModel", font_size=20).move_to(ai_model)
        
        grid = Square(side_length=1.5, color=RED).shift(RIGHT * 4)
        grid_text = Text("Power\nGrid", font_size=20).move_to(grid)
        
        # Create arrows
        arrow1 = Arrow(building.get_right(), ai_model.get_left(), color=WHITE)
        arrow2 = Arrow(ai_model.get_right(), grid.get_left(), color=WHITE)
        
        # Animate the flow
        self.play(
            Create(building), Write(building_text),
            Create(ai_model), Write(ai_text),
            Create(grid), Write(grid_text)
        )
        self.play(
            GrowArrow(arrow1),
            GrowArrow(arrow2)
        )
        
        # Show data flowing
        dot1 = Dot(color=YELLOW).move_to(building.get_right())
        dot2 = Dot(color=YELLOW).move_to(ai_model.get_right())
        
        self.play(
            MoveAlongPath(dot1, arrow1),
            run_time=2
        )
        self.play(
            MoveAlongPath(dot2, arrow2),
            run_time=2
        )