"""
Complete Project Overview Animation - 2 minutes
Shows the entire pipeline from SQL → KG → GNN → Results
"""

from manim import *
import numpy as np

class ProjectOverview(Scene):
    def construct(self):
        # Total duration: ~120 seconds (2 minutes)
        
        # ============= INTRO (0-10s) =============
        self.show_intro()
        
        # ============= SQL TO KG PIPELINE (10-35s) =============
        self.show_sql_to_kg_pipeline()
        
        # ============= KNOWLEDGE GRAPH STRUCTURE (35-55s) =============
        self.show_knowledge_graph()
        
        # ============= GNN MODEL ARCHITECTURE (55-80s) =============
        self.show_gnn_architecture()
        
        # ============= TRAINING & RESULTS (80-105s) =============
        self.show_training_process()
        
        # ============= FINAL RESULTS & IMPACT (105-120s) =============
        self.show_final_results()
    
    def show_intro(self):
        """Introduction with project title and overview (10s)"""
        # Main title
        title = Text("AI-Driven Energy Community Optimization", 
                    font_size=42, color=BLUE)
        subtitle = Text("From Data to Smart Grid Solutions", 
                       font_size=28, color=GRAY)
        subtitle.next_to(title, DOWN)
        
        # Project components
        components = VGroup(
            Text("SQL Database", font_size=20, color=GREEN),
            Text("→", font_size=24),
            Text("Knowledge Graph", font_size=20, color=YELLOW),
            Text("→", font_size=24),
            Text("GNN Model", font_size=20, color=RED),
            Text("→", font_size=24),
            Text("Smart Clusters", font_size=20, color=PURPLE)
        ).arrange(RIGHT, buff=0.3)
        components.next_to(subtitle, DOWN, buff=1)
        
        # Animate
        self.play(Write(title), run_time=2)
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(1)
        self.play(FadeIn(components), run_time=2)
        self.wait(2)
        self.play(FadeOut(VGroup(title, subtitle, components)))
    
    def show_sql_to_kg_pipeline(self):
        """SQL to Knowledge Graph pipeline visualization (25s)"""
        # Title
        section_title = Text("Phase 1: Data Pipeline", font_size=36, color=BLUE)
        section_title.to_edge(UP)
        self.play(Write(section_title))
        
        # SQL Database representation
        sql_db = VGroup()
        sql_box = Rectangle(width=2.5, height=3, color=GREEN)
        sql_label = Text("SQL Database", font_size=16)
        sql_label.next_to(sql_box, UP)
        
        # Tables inside SQL
        tables = VGroup(
            Text("Buildings", font_size=12),
            Text("Transformers", font_size=12),
            Text("Energy Profiles", font_size=12),
            Text("Grid Topology", font_size=12)
        ).arrange(DOWN, buff=0.2)
        tables.move_to(sql_box)
        
        sql_db.add(sql_box, sql_label, tables)
        sql_db.shift(LEFT * 4)
        
        # KG Builder
        kg_builder = VGroup()
        builder_circle = Circle(radius=1, color=YELLOW)
        builder_label = Text("KG Builder", font_size=14)
        builder_label.move_to(builder_circle)
        kg_builder.add(builder_circle, builder_label)
        
        # Neo4j Database
        neo4j = VGroup()
        neo4j_hex = RegularPolygon(6, radius=1.2, color=ORANGE)
        neo4j_label = Text("Neo4j", font_size=14)
        neo4j_label.move_to(neo4j_hex)
        neo4j.add(neo4j_hex, neo4j_label)
        neo4j.shift(RIGHT * 4)
        
        # Arrows with labels
        arrow1 = Arrow(sql_db.get_right(), kg_builder.get_left())
        arrow1_label = Text("Extract", font_size=10).next_to(arrow1, UP)
        
        arrow2 = Arrow(kg_builder.get_right(), neo4j.get_left())
        arrow2_label = Text("Transform", font_size=10).next_to(arrow2, UP)
        
        # Animate pipeline
        self.play(Create(sql_db))
        self.wait(1)
        self.play(GrowArrow(arrow1), Write(arrow1_label))
        self.play(Create(kg_builder))
        self.wait(1)
        self.play(GrowArrow(arrow2), Write(arrow2_label))
        self.play(Create(neo4j))
        
        # Show data flow animation
        data_dots = VGroup(*[Dot(color=GREEN, radius=0.08) for _ in range(5)])
        for i, dot in enumerate(data_dots):
            dot.move_to(sql_box.get_center() + UP * (0.5 - i*0.25))
        
        self.play(Create(data_dots))
        
        # Move dots through pipeline
        for dot in data_dots:
            self.play(
                MoveAlongPath(dot, Line(dot.get_center(), kg_builder.get_center())),
                run_time=0.5
            )
        
        self.play(Flash(kg_builder, color=YELLOW))
        
        for dot in data_dots:
            self.play(
                MoveAlongPath(dot, Line(kg_builder.get_center(), neo4j.get_center())),
                run_time=0.5
            )
        
        # Statistics
        stats = VGroup(
            Text("500+ Buildings", font_size=14, color=GREEN),
            Text("50+ Transformers", font_size=14, color=BLUE),
            Text("10,000+ Relationships", font_size=14, color=YELLOW)
        ).arrange(DOWN, buff=0.2).to_edge(DOWN)
        
        self.play(FadeIn(stats))
        self.wait(2)
        self.play(FadeOut(VGroup(section_title, sql_db, kg_builder, neo4j, 
                                 arrow1, arrow1_label, arrow2, arrow2_label, 
                                 data_dots, stats)))
    
    def show_knowledge_graph(self):
        """Knowledge Graph structure visualization (20s)"""
        # Title
        section_title = Text("Phase 2: Knowledge Graph", font_size=36, color=YELLOW)
        section_title.to_edge(UP)
        self.play(Write(section_title))
        
        # Create hierarchical KG structure
        # Level 1: MV Network
        mv_node = Circle(radius=0.5, color=RED, fill_opacity=0.8)
        mv_label = Text("MV", font_size=14, color=WHITE).move_to(mv_node)
        mv = VGroup(mv_node, mv_label).shift(UP * 2)
        
        # Level 2: Transformers
        transformers = VGroup()
        trans_positions = [LEFT * 3, LEFT, RIGHT, RIGHT * 3]
        for i, pos in enumerate(trans_positions):
            t_node = Square(side_length=0.6, color=ORANGE, fill_opacity=0.8)
            t_label = Text(f"T{i+1}", font_size=12).move_to(t_node)
            t = VGroup(t_node, t_label).shift(pos + UP * 0.5)
            transformers.add(t)
        
        # Level 3: Buildings
        buildings = VGroup()
        for i in range(12):
            angle = i * PI / 6
            pos = np.array([2.5 * np.cos(angle), 2.5 * np.sin(angle) - 1.5, 0])
            b_node = Circle(radius=0.25, color=BLUE, fill_opacity=0.8)
            b_label = Text(f"B{i+1}", font_size=8).move_to(b_node)
            b = VGroup(b_node, b_label).move_to(pos)
            buildings.add(b)
        
        # Create edges
        edges = VGroup()
        # MV to Transformers
        for t in transformers:
            edge = Line(mv.get_center(), t.get_center(), color=GRAY)
            edges.add(edge)
        
        # Transformers to Buildings
        for i, b in enumerate(buildings):
            t_idx = i // 3  # Each transformer connects to 3 buildings
            if t_idx < len(transformers):
                edge = Line(transformers[t_idx].get_center(), b.get_center(), color=GRAY)
                edges.add(edge)
        
        # Animate construction
        self.play(Create(edges), run_time=1)
        self.play(Create(mv))
        self.play(Create(transformers))
        self.play(Create(buildings), run_time=2)
        
        # Show relationships
        rel_text = VGroup(
            Text("FEEDS_TO", font_size=12, color=GREEN),
            Text("CONNECTS_TO", font_size=12, color=YELLOW),
            Text("SIMILAR_TO", font_size=12, color=BLUE_E)
        ).arrange(DOWN).to_edge(RIGHT)
        
        self.play(FadeIn(rel_text))
        
        # Highlight energy flow
        energy_flow = VGroup()
        for edge in edges[:4]:  # MV to transformers
            dot = Dot(color=YELLOW, radius=0.1)
            dot.move_to(edge.get_start())
            energy_flow.add(dot)
        
        self.play(Create(energy_flow))
        animations = []
        for dot, edge in zip(energy_flow, edges[:4]):
            animations.append(MoveAlongPath(dot, edge))
        self.play(*animations, run_time=2)
        
        # Show node properties
        props = VGroup(
            Text("Properties:", font_size=14, color=WHITE),
            Text("• Load profiles", font_size=12),
            Text("• Solar capacity", font_size=12),
            Text("• Grid constraints", font_size=12),
            Text("• Flexibility", font_size=12)
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT)
        
        self.play(FadeIn(props))
        self.wait(2)
        
        self.play(FadeOut(VGroup(section_title, mv, transformers, buildings, 
                                 edges, rel_text, energy_flow, props)))
    
    def show_gnn_architecture(self):
        """GNN Model Architecture visualization (25s)"""
        # Title
        section_title = Text("Phase 3: Graph Neural Network", font_size=36, color=RED)
        section_title.to_edge(UP)
        self.play(Write(section_title))
        
        # Input layer
        input_layer = VGroup()
        for i in range(5):
            node = Circle(radius=0.2, color=BLUE, fill_opacity=0.8)
            node.shift(LEFT * 5 + UP * (2 - i))
            input_layer.add(node)
        input_label = Text("Input\nFeatures", font_size=12).next_to(input_layer, DOWN)
        
        # GNN Layers
        gnn_layers = []
        layer_colors = [GREEN, YELLOW, ORANGE]
        layer_names = ["GAT Layer", "RGCN Layer", "Pool Layer"]
        
        for j, (color, name) in enumerate(zip(layer_colors, layer_names)):
            layer = VGroup()
            for i in range(4 - j):  # Decreasing nodes per layer
                node = Circle(radius=0.25, color=color, fill_opacity=0.8)
                node.shift(LEFT * (2 - j * 2) + UP * (1.5 - i * 0.8))
                layer.add(node)
            label = Text(name, font_size=10).next_to(layer, DOWN)
            layer.add(label)
            gnn_layers.append(layer)
        
        # Task heads
        task_heads = VGroup()
        tasks = ["Clustering", "Solar", "Flexibility"]
        task_colors = [PURPLE, YELLOW, GREEN]
        for i, (task, color) in enumerate(zip(tasks, task_colors)):
            head = Rectangle(width=1.2, height=0.5, color=color, fill_opacity=0.8)
            head.shift(RIGHT * 4 + UP * (1 - i))
            label = Text(task, font_size=10, color=WHITE).move_to(head)
            task_heads.add(VGroup(head, label))
        
        # Create network connections
        all_edges = VGroup()
        
        # Input to first GNN layer
        for in_node in input_layer[:-1]:  # Exclude label
            for gnn_node in gnn_layers[0][:-1]:  # Exclude label
                edge = Line(in_node.get_center(), gnn_node.get_center(), 
                          stroke_width=1, color=GRAY, stroke_opacity=0.3)
                all_edges.add(edge)
        
        # Between GNN layers
        for i in range(len(gnn_layers) - 1):
            for node1 in gnn_layers[i][:-1]:
                for node2 in gnn_layers[i+1][:-1]:
                    edge = Line(node1.get_center(), node2.get_center(),
                              stroke_width=1, color=GRAY, stroke_opacity=0.3)
                    all_edges.add(edge)
        
        # Last GNN to task heads
        for gnn_node in gnn_layers[-1][:-1]:
            for task_head in task_heads:
                edge = Line(gnn_node.get_center(), task_head[0].get_left(),
                          stroke_width=1, color=GRAY, stroke_opacity=0.3)
                all_edges.add(edge)
        
        # Animate architecture
        self.play(Create(all_edges), run_time=1)
        self.play(Create(input_layer))
        self.play(Write(input_label))
        
        for layer in gnn_layers:
            self.play(Create(layer))
        
        self.play(Create(task_heads))
        
        # Show information flow
        info_dots = VGroup()
        for node in input_layer[:-1]:
            dot = Dot(color=WHITE, radius=0.05)
            dot.move_to(node)
            info_dots.add(dot)
        
        self.play(Create(info_dots))
        
        # Propagate through network
        for layer in gnn_layers:
            new_positions = [n.get_center() for n in layer[:-1]]
            animations = []
            for i, dot in enumerate(info_dots):
                if i < len(new_positions):
                    animations.append(dot.animate.move_to(new_positions[i]))
            self.play(*animations, run_time=0.8)
            self.play(Flash(layer[0], color=layer[0].get_color()))
        
        # Model specs
        specs = VGroup(
            Text("Model Specifications:", font_size=14, color=WHITE),
            Text("• 3-layer Heterogeneous GNN", font_size=10),
            Text("• Attention mechanism", font_size=10),
            Text("• Physics constraints", font_size=10),
            Text("• Uncertainty quantification", font_size=10)
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT).shift(DOWN)
        
        self.play(FadeIn(specs))
        self.wait(2)
        
        self.play(FadeOut(VGroup(section_title, input_layer, input_label, *gnn_layers,
                                 task_heads, all_edges, info_dots, specs)))
    
    def show_training_process(self):
        """Training process and optimization (25s)"""
        # Title
        section_title = Text("Phase 4: Training & Optimization", font_size=36, color=GREEN)
        section_title.to_edge(UP)
        self.play(Write(section_title))
        
        # Loss function components
        loss_components = VGroup(
            Text("Total Loss =", font_size=20),
            Text("Clustering", font_size=16, color=BLUE),
            Text("+", font_size=16),
            Text("Complementarity", font_size=16, color=GREEN),
            Text("+", font_size=16),
            Text("Solar ROI", font_size=16, color=YELLOW),
            Text("+", font_size=16),
            Text("Physics", font_size=16, color=RED)
        ).arrange(RIGHT, buff=0.2)
        loss_components.shift(UP * 2)
        
        self.play(Write(loss_components))
        
        # Training curve
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 1, 0.2],
            x_length=8,
            y_length=4,
            axis_config={"include_numbers": False}
        ).shift(DOWN)
        
        x_label = Text("Epochs", font_size=14).next_to(axes.x_axis, DOWN)
        y_label = Text("Loss", font_size=14).next_to(axes.y_axis, LEFT).rotate(PI/2)
        
        # Create loss curve
        loss_func = lambda x: 0.8 * np.exp(-x/20) + 0.2
        loss_curve = axes.plot(loss_func, color=BLUE, x_range=[0, 100])
        
        # Validation curve
        val_func = lambda x: 0.85 * np.exp(-x/25) + 0.22
        val_curve = axes.plot(val_func, color=GREEN, x_range=[0, 100])
        
        # Animate training
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(loss_curve), Create(val_curve), run_time=3)
        
        # Add legend
        legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=BLUE), 
                   Text("Training", font_size=10)).arrange(RIGHT),
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=GREEN),
                   Text("Validation", font_size=10)).arrange(RIGHT)
        ).arrange(DOWN).next_to(axes, RIGHT)
        
        self.play(FadeIn(legend))
        
        # Show metrics
        metrics = VGroup(
            Text("Final Performance:", font_size=16, color=WHITE),
            Text("• Clustering Score: 0.92", font_size=12, color=BLUE),
            Text("• Energy Saving: 23%", font_size=12, color=GREEN),
            Text("• ROI: 185%", font_size=12, color=YELLOW),
            Text("• Grid Stability: ✓", font_size=12, color=RED)
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(RIGHT)
        
        self.play(FadeIn(metrics))
        self.wait(3)
        
        self.play(FadeOut(VGroup(section_title, loss_components, axes, x_label, 
                                 y_label, loss_curve, val_curve, legend, metrics)))
    
    def show_final_results(self):
        """Final results and impact visualization (15s)"""
        # Title
        section_title = Text("Results: Smart Energy Communities", font_size=36, color=PURPLE)
        section_title.to_edge(UP)
        self.play(Write(section_title))
        
        # Create final network visualization
        # Central community hub
        hub = Circle(radius=0.5, color=RED, fill_opacity=0.8)
        hub_label = Text("Hub", font_size=12, color=WHITE).move_to(hub)
        
        # Energy communities (clusters)
        communities = VGroup()
        community_colors = [BLUE, GREEN, YELLOW, ORANGE]
        positions = [
            UP * 2 + LEFT * 2,
            UP * 2 + RIGHT * 2,
            DOWN * 2 + LEFT * 2,
            DOWN * 2 + RIGHT * 2
        ]
        
        for i, (pos, color) in enumerate(zip(positions, community_colors)):
            # Community circle
            comm = Circle(radius=1, color=color, fill_opacity=0.3)
            comm.shift(pos)
            
            # Buildings in community
            buildings = VGroup()
            for j in range(4):
                angle = j * PI / 2
                b_pos = pos + 0.5 * np.array([np.cos(angle), np.sin(angle), 0])
                building = Square(side_length=0.2, color=color, fill_opacity=0.8)
                building.move_to(b_pos)
                buildings.add(building)
            
            label = Text(f"Community {i+1}", font_size=10).next_to(comm, DOWN)
            communities.add(VGroup(comm, buildings, label))
        
        # Connections
        connections = VGroup()
        for comm in communities:
            conn = Line(hub.get_center(), comm[0].get_center(), color=GRAY)
            connections.add(conn)
        
        # Animate final system
        self.play(Create(connections))
        self.play(Create(VGroup(hub, hub_label)))
        self.play(Create(communities), run_time=2)
        
        # Energy sharing animation
        energy_dots = VGroup()
        for comm in communities:
            dot = Dot(color=YELLOW, radius=0.1)
            dot.move_to(comm[0].get_center())
            energy_dots.add(dot)
        
        self.play(Create(energy_dots))
        
        # Show P2P energy trading
        for _ in range(2):
            animations = []
            for dot, conn in zip(energy_dots, connections):
                animations.append(MoveAlongPath(dot, conn))
            self.play(*animations, run_time=1)
            
            # Move back
            for dot, comm in zip(energy_dots, communities):
                dot.move_to(comm[0].get_center())
        
        # Final impact statistics
        impact = VGroup(
            Text("Project Impact", font_size=20, color=WHITE),
            Text("━" * 20, font_size=12),
            Text("23% Energy Savings", font_size=16, color=GREEN),
            Text("35% Peak Reduction", font_size=16, color=BLUE),
            Text("500+ Buildings Optimized", font_size=16, color=YELLOW),
            Text("€2M Annual Savings", font_size=16, color=ORANGE)
        ).arrange(DOWN).scale(0.8).to_edge(DOWN)
        
        self.play(FadeIn(impact, shift=UP))
        
        # Thank you message
        self.wait(2)
        self.play(FadeOut(VGroup(communities, hub, hub_label, connections, energy_dots)))
        
        thank_you = Text("Thank You", font_size=48, color=BLUE)
        tech_stack = Text("SQL • Neo4j • PyTorch • GNN", 
                         font_size=24, color=GRAY).next_to(thank_you, DOWN)
        
        self.play(
            Transform(section_title, thank_you),
            FadeIn(tech_stack),
            FadeOut(impact)
        )
        self.wait(2)