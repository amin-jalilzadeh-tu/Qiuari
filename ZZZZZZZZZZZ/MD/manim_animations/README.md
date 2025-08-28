# Energy System Manim Animations

This folder contains Manim animations for visualizing energy grid systems, AI-driven planning, and digital twin frameworks.

## Prerequisites

- Python 3.11+ (DDsaie environment recommended)
- Manim Community v0.19.0
- typing_extensions >= 4.15.0

## Installation

```bash
pip install --upgrade manim typing_extensions
```

## Available Animations

### 1. Intro Animation (`intro_animation.py`)
A simple introduction animation showcasing the Digital Twin Framework concept.

### 2. Data Flow Diagram (`data_flow_diagram.py`)
Visualizes the data flow between buildings, AI models, and the power grid.

### 3. Energy Graph (`energy_graph.py`)
Shows energy consumption patterns over 24 hours with original vs optimized curves.

### 4. Energy Network (`energy_network.py`)
Displays a hub-and-spoke energy distribution network with animated power flow.

### 5. Energy Grid Network (`energy_grid_network.py`)
Complex visualization of:
- Substation and building nodes
- Power distribution connections
- Peak load analysis
- AI optimization with battery storage
- Load balancing demonstration

### 6. Building Clusters 3D (`building_clusters_3d.py`)
3D visualization of building clusters with varying heights and colors.

## Running Animations

### Basic Usage

```bash
# Low quality (fast rendering)
manim -ql <filename>.py <ClassName>

# Medium quality
manim -qm <filename>.py <ClassName>

# High quality
manim -qh <filename>.py <ClassName>

# Preview after rendering
manim -pql <filename>.py <ClassName>
```

### Examples

```bash
# Run intro animation with preview
manim -pql intro_animation.py IntroAnimation

# Run energy grid network in high quality
manim -qh energy_grid_network.py EnergyGridNetwork

# Run 3D building clusters
manim -ql building_clusters_3d.py BuildingClusters3D
```

## Output Location

Rendered videos are saved in:
```
media/videos/<script_name>/<quality>/
```

Quality folders:
- `480p15/` - Low quality
- `720p30/` - Medium quality
- `1080p60/` - High quality

## Generated Videos

The following videos have been successfully generated:

1. ✅ `IntroAnimation.mp4` - Digital Twin Framework intro
2. ✅ `DataFlowDiagram.mp4` - Data flow visualization
3. ✅ `EnergyGraph.mp4` - Energy consumption graphs
4. ✅ `EnergyNetwork.mp4` - Network energy distribution
5. ✅ `EnergyGridNetwork.mp4` - Complete grid system simulation
6. ✅ `BuildingClusters3D.mp4` - 3D building visualization

## Troubleshooting

### Import Errors
If you encounter `ImportError: cannot import name 'TypeIs'`:
```bash
pip install --upgrade typing_extensions
```

### LaTeX Errors
The animations have been modified to work without LaTeX. If you want to use LaTeX features, install:
```bash
# Windows
choco install miktex

# Linux
sudo apt-get install texlive-full

# Mac
brew install --cask mactex
```

### Environment Issues
Make sure you're using the correct Python environment:
```bash
# Use DDsaie environment
"d:\New folder (2)\Anaconda\DDsaie\python.exe" -m manim -ql <script>.py <Class>
```

## Customization

You can modify the animations by editing the Python files:
- Change colors, sizes, and positions
- Adjust animation timing with `run_time` parameter
- Add new elements or animations
- Modify text labels and descriptions

## Features Demonstrated

- **Energy Grid Visualization**: Substations, buildings, power connections
- **Load Analysis**: Peak load detection and visualization
- **AI Optimization**: Automated load balancing and battery placement
- **3D Rendering**: Three-dimensional building clusters
- **Data Flow**: Information flow in smart grid systems
- **Time Series**: Energy consumption over time

## Contact

For issues or questions about these animations, please refer to the main project documentation.