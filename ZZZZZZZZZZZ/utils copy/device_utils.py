"""
Utility functions for device management
"""

import torch
from torch_geometric.data import HeteroData


def move_hetero_data_to_device(data: HeteroData, device: torch.device) -> HeteroData:
    """
    Properly move all components of HeteroData to device
    
    Args:
        data: Heterogeneous graph data
        device: Target device
        
    Returns:
        Data on device
    """
    # Move node features
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x') and data[node_type].x is not None:
            data[node_type].x = data[node_type].x.to(device)
        
        # Move any other node attributes
        for attr in ['lv_group_ids', 'building_ids', 'positions']:
            if hasattr(data[node_type], attr):
                attr_value = getattr(data[node_type], attr)
                if isinstance(attr_value, torch.Tensor):
                    setattr(data[node_type], attr, attr_value.to(device))
    
    # Move edge indices
    for edge_type in data.edge_types:
        if hasattr(data[edge_type], 'edge_index') and data[edge_type].edge_index is not None:
            data[edge_type].edge_index = data[edge_type].edge_index.to(device)
        
        # Move edge attributes
        if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
            data[edge_type].edge_attr = data[edge_type].edge_attr.to(device)
    
    # Move any global attributes
    for attr in ['mv_station_id', 'lv_id_mapping', 'building_id_map']:
        if hasattr(data, attr):
            attr_value = getattr(data, attr)
            if isinstance(attr_value, torch.Tensor):
                setattr(data, attr, attr_value.to(device))
    
    return data