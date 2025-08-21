"""
Simple inference script to test the trained GNN model
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import pandas as pd
import numpy as np
from pathlib import Path

def run_simple_inference():
    """Run a simple inference using the trained model"""
    
    print("Loading trained model...")
    
    # Load the best model
    model_path = Path("checkpoints/best_model.pth")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_loss', 'unknown')}")
    
    # Check what's in the checkpoint
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Load graph data
    graph_path = Path("processed_data/graph_data.pt")
    if graph_path.exists():
        graph_data = torch.load(graph_path, map_location='cpu', weights_only=False)
        print(f"Loaded graph with {graph_data.num_nodes} nodes")
        
        # Try to get model from checkpoint
        if 'model_state_dict' in checkpoint:
            # Need to recreate model architecture first
            from models.base_gnn import HomogeneousEnergyGNN
            model = HomogeneousEnergyGNN(
                input_dim=10,
                hidden_dim=128,
                output_dim=10,
                num_layers=3,
                dropout=0.2
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("Model loaded from state dict")
        else:
            print("Model state dict not found in checkpoint")
            return
        
        with torch.no_grad():
            # Simple forward pass
            if hasattr(graph_data, 'x') and hasattr(graph_data, 'edge_index'):
                outputs = model(graph_data.x, graph_data.edge_index)
                print(f"Generated predictions with shape: {outputs.shape}")
                
                # Show some statistics
                print(f"Mean prediction: {outputs.mean().item():.4f}")
                print(f"Std prediction: {outputs.std().item():.4f}")
                print(f"Min prediction: {outputs.min().item():.4f}")
                print(f"Max prediction: {outputs.max().item():.4f}")
                
                # Find top energy consumers (highest predicted values)
                top_k = 5
                top_values, top_indices = torch.topk(outputs.mean(dim=1), k=min(top_k, outputs.size(0)))
                
                print(f"\nTop {top_k} nodes by predicted energy:")
                for i, (idx, val) in enumerate(zip(top_indices, top_values)):
                    print(f"  Node {idx.item()}: {val.item():.4f}")
    
    # Load original building data to show details
    buildings_path = Path("mimic_data/buildings.csv")
    if buildings_path.exists():
        buildings_df = pd.read_csv(buildings_path)
        print(f"\nBuilding statistics from data:")
        print(f"  Total buildings: {len(buildings_df)}")
        if 'has_solar' in buildings_df.columns:
            print(f"  Buildings with solar: {buildings_df['has_solar'].sum()}")
        if 'has_battery' in buildings_df.columns:
            print(f"  Buildings with battery: {buildings_df['has_battery'].sum()}")
        if 'area' in buildings_df.columns:
            print(f"  Average building area: {buildings_df['area'].mean():.1f} m²")
        if 'building_function' in buildings_df.columns:
            print(f"  Building functions: {buildings_df['building_function'].value_counts().to_dict()}")
        if 'residential_type' in buildings_df.columns:
            res_types = buildings_df['residential_type'].dropna().value_counts()
            if len(res_types) > 0:
                print(f"  Residential types: {res_types.to_dict()}")

if __name__ == "__main__":
    run_simple_inference()