from manim import *

class IntroAnimation(Scene):
    def construct(self):
        # Create title
        title = Text("Digital Twin Framework", font_size=48)
        subtitle = Text("AI-Driven Energy Planning", font_size=36)
        
        # Position them
        title.to_edge(UP)
        subtitle.next_to(title, DOWN)
        
        # Animate
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(1)
        
        # Transform to a circle with text
        circle = Circle(radius=2, color=BLUE)
        text = Text("Energy\nOptimization", font_size=24)
        
        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            Create(circle),
            Write(text)
        )
        self.wait(1)