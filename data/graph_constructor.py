# graph_constructor.py - UPDATED FOR YOUR SCHEMA
"""
Converts Knowledge Graph data to PyTorch Geometric heterogeneous graph format.
Updated for your schema: Building → CableGroup → Transformer → Substation
"""

import torch
from torch_geometric.data import HeteroData
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GraphConstructor:
    """Constructs PyTorch Geometric graphs from your KG data."""
    # In graph_constructor.py - Update __init__ method

    def __init__(self, kg_connector):
        """
        Initialize with KG connector.
        """
        self.kg = kg_connector
        
        # Node types (including MV hierarchy)
        self.node_types = [
            'building', 
            'cable_group',  # LV groups
            'mv_station',   # NEW: MV stations
            'hv_substation',  # NEW: HV substations
            'transformer', 
            'adjacency_cluster'
        ]
        
        # Edge types including MV hierarchy relationships
        self.edge_types = [
            # Original relationships
            ('building', 'connected_to', 'cable_group'),      # CONNECTED_TO
            ('cable_group', 'connects_to', 'transformer'),    # CONNECTS_TO
            ('transformer', 'feeds_from', 'substation'),      # FEEDS_FROM (legacy)
            ('building', 'in_cluster', 'adjacency_cluster'),  # IN_ADJACENCY_CLUSTER
            
            # NEW: MV-LV hierarchy relationships
            ('cable_group', 'supplied_by', 'mv_station'),     # MV_SUPPLIES_LV
            ('mv_station', 'supplied_by', 'hv_substation'),   # HV_SUPPLIES_MV
        ]
        
        # Node ID mappings
        self.node_mappings = {}
        
    def build_hetero_graph(self, district_name: str, 
                        include_energy_sharing: bool = True,
                        include_temporal: bool = True,
                        lookback_hours: int = 24) -> HeteroData:
        """
        Build complete heterogeneous graph for a district.
        
        Args:
            district_name: District to build graph for
            include_energy_sharing: Add adjacency cluster edges
            include_temporal: Include time series features
            lookback_hours: Hours of history for temporal features
            
        Returns:
            PyTorch Geometric HeteroData object with temporal features
        """
        logger.info(f"Building graph for district {district_name}")
        logger.info(f"Temporal features: {include_temporal}, Lookback: {lookback_hours} hours")
        
        # Get topology from KG
        topology = self.kg.get_grid_topology(district_name)
        
        if not topology or not topology.get('nodes'):
            logger.error(f"No topology data found for district {district_name}")
            return HeteroData()
        
        # Create empty HeteroData
        graph = HeteroData()
        
        # Add LV group IDs if available
        if 'lv_group_ids' in topology:
            graph['building'].lv_group_ids = torch.tensor(
                topology['lv_group_ids'], dtype=torch.long
            )
        
        # Add nodes with features (including temporal if requested)
        self._add_nodes_to_graph(
            graph, 
            topology['nodes'], 
            district_name,
            include_temporal=include_temporal,
            lookback_hours=lookback_hours
        )
        
        # Add edges
        self._add_edges_to_graph(graph, topology['edges'])
        
        # Add global graph attributes
        graph.district_name = district_name
        graph.include_temporal = include_temporal
        graph.lookback_hours = lookback_hours
        
        # Calculate number of nodes per type
        graph.num_nodes_dict = {}
        for node_type in self.node_types:
            if node_type in graph.node_types:
                if hasattr(graph[node_type], 'x'):
                    graph.num_nodes_dict[node_type] = graph[node_type].x.shape[0]
                else:
                    graph.num_nodes_dict[node_type] = 0
            else:
                graph.num_nodes_dict[node_type] = 0
        
        # Add temporal info to metadata
        if include_temporal:
            temporal_info = {}
            for node_type in graph.node_types:
                if hasattr(graph[node_type], 'x_temporal'):
                    shape = graph[node_type].x_temporal.shape
                    temporal_info[node_type] = {
                        'shape': shape,
                        'n_timesteps': shape[1],
                        'n_features': shape[2]
                    }
            graph.temporal_info = temporal_info
            logger.info(f"Temporal features added for: {list(temporal_info.keys())}")
        
        logger.info(f"Graph built: {graph.num_nodes_dict}")
        
        return graph
    
    def build_subgraph_for_task(self, district_name: str, 
                               task_type: str,
                               **kwargs) -> HeteroData:
        """
        Build task-specific subgraph.
        
        Args:
            district_name: District name
            task_type: 'retrofit', 'energy_sharing', 'solar', or 'electrification'
            **kwargs: Task-specific parameters
            
        Returns:
            Task-specific HeteroData
        """
        if task_type == 'retrofit':
            return self._build_retrofit_graph(district_name, **kwargs)
        elif task_type == 'energy_sharing':
            return self._build_energy_sharing_graph(district_name, **kwargs)
        elif task_type == 'solar':
            return self._build_solar_graph(district_name, **kwargs)
        elif task_type == 'electrification':
            return self._build_electrification_graph(district_name, **kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    
    # Fix for graph_constructor.py - Replace the _add_nodes_to_graph method
    def _add_nodes_to_graph(self, graph: HeteroData, 
                        nodes: Dict[str, List],
                        district_name: str,
                        include_temporal: bool = True,
                        lookback_hours: int = 24):
        """Add nodes with features to graph including temporal data."""
        
        # Initialize ID storage separately (not as graph attributes)
        node_ids = {}
        
        # Process Buildings
        buildings = nodes.get('buildings', [])
        if buildings:
            building_features = []
            building_ids = []
            
            for i, b in enumerate(buildings):
                b_id = str(b.get('ogc_fid', f'B_{i}'))
                building_ids.append(b_id)
                
                # Energy label to numeric
                label_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1, 'Unknown': 0}
                energy_score = label_map.get(b.get('energy_label', 'Unknown'), 0)
                
                # Solar potential to numeric
                solar_map = {'high': 3, 'medium': 2, 'low': 1, 'none': 0}
                solar_score = solar_map.get(b.get('solar_potential', 'none'), 0)
                
                
                # Electrification feasibility to numeric
                electrify_map = {'immediate': 3, 'ready': 2, 'possible': 1, 'difficult': 0}
                electrify_score = electrify_map.get(b.get('electrification_feasibility', 'possible'), 1)
                
                # Extract year from age_range
                age_range = b.get('age_range', '2000 - 2010')
                if age_range and ' - ' in age_range:
                    years = age_range.split(' - ')
                    try:
                        avg_year = (int(years[0]) + int(years[1])) // 2
                    except:
                        avg_year = 2000
                else:
                    avg_year = b.get('building_year', 2000) or 2000
                age = 2024 - avg_year
                
                building_features.append([
                    float(b.get('area', 150) or 150),
                    float(energy_score),
                    float(solar_score),
                    float(electrify_score),
                    float(age),
                    float(b.get('suitable_roof_area', 100) or 100),
                    float(b.get('height', 10) or 10),
                    1.0 if b.get('has_solar', False) else 0.0,
                    1.0 if b.get('has_battery', False) else 0.0,
                    1.0 if b.get('has_heat_pump', False) else 0.0,
                    float(b.get('num_shared_walls', 0) or 0),
                    float(b.get('x', 0) or 0),
                    float(b.get('y', 0) or 0),
                    float(b.get('avg_electricity_demand_kw', 0) or 0),
                    float(b.get('avg_heating_demand_kw', 0) or 0),
                    float(b.get('peak_electricity_demand_kw', 0) or 0),
                    float(b.get('energy_intensity_kwh_m2', 0) or 0)
                ])
            
            if building_features:
                graph['building'].x = torch.tensor(building_features, dtype=torch.float)
                # Store IDs in the node_ids dict instead of as graph attribute
                node_ids['building'] = building_ids
                self.node_mappings['building'] = {id_: i for i, id_ in enumerate(building_ids)}
                
                # Add temporal features if requested
                if include_temporal:
                    logger.info(f"Fetching temporal features for {len(building_ids)} buildings...")
                    time_series_data = self.kg.get_building_time_series(
                        building_ids, 
                        lookback_hours=lookback_hours
                    )
                    
                    temporal_features = []
                    for bid in building_ids:
                        if bid in time_series_data:
                            temporal_features.append(time_series_data[bid])
                        else:
                            temporal_features.append(np.zeros((lookback_hours, 8)))
                            logger.debug(f"No temporal data for building {bid}")
                    
                    if temporal_features:
                        graph['building'].x_temporal = torch.tensor(
                            np.stack(temporal_features), 
                            dtype=torch.float
                        )
                        logger.info(f"Added temporal features: {graph['building'].x_temporal.shape}")
        
        # Process Cable Groups
        cable_groups = nodes.get('cable_groups', [])
        if cable_groups:
            cg_features = []
            cg_ids = []
            
            for i, cg in enumerate(cable_groups):
                cg_id = cg.get('group_id', f'CG_{i}')
                if cg_id is None:
                    cg_id = f'CG_{i}'
                cg_ids.append(str(cg_id))
                
                agg = self.kg.aggregate_to_cable_group(cg_id) if cg_id != f'CG_{i}' else {}
                
                cg_features.append([
                    float(cg.get('total_length_m', 500) or 500),
                    float(cg.get('segment_count', 10) or 10),
                    float(agg.get('building_count', 0) or 0),
                    float(agg.get('avg_energy_score', 4) or 4),
                    float(agg.get('total_area', 0) or 0),
                    float(agg.get('total_roof_area', 0) or 0),
                    float(agg.get('solar_count', 0) or 0),
                    float(agg.get('battery_count', 0) or 0),
                    float(agg.get('hp_count', 0) or 0),
                    float(cg.get('avg_electricity_demand_kw', 0) or 0),
                    float(cg.get('peak_electricity_demand_kw', 0) or 0),
                    float(cg.get('demand_diversity_factor', 1.0) or 1.0)
                ])
            
            if cg_features:
                graph['cable_group'].x = torch.tensor(cg_features, dtype=torch.float)
                node_ids['cable_group'] = cg_ids
                self.node_mappings['cable_group'] = {id_: i for i, id_ in enumerate(cg_ids)}
        
        # Process Transformers
        transformers = nodes.get('transformers', [])
        if transformers:
            t_features = []
            t_ids = []
            
            for i, t in enumerate(transformers):
                t_id = t.get('transformer_id', f'T_{i}')  # ← FIXED: Use transformer_id
                if t_id is None:
                    t_id = f'T_{i}'
                t_ids.append(str(t_id))
                
                # Create 8 features for transformer to match model expectations
                # Convert voltage level string to numeric
                voltage = t.get('voltage_level', 'MV')
                voltage_val = 10000.0 if voltage == 'MV' else (400.0 if voltage == 'LV' else 150000.0)
                
                t_features.append([
                    1.0,  # indicator
                    float(t.get('x', 0) or 0),
                    float(t.get('y', 0) or 0),
                    float(t.get('capacity', 1000) or 1000),  # default capacity
                    float(t.get('utilization', 0.5) or 0.5),  # default utilization
                    float(t.get('age', 10) or 10),  # default age in years
                    1.0 if t.get('is_active', True) else 0.0,  # active status
                    voltage_val  # voltage level as numeric
                ])
            
            if t_features:
                graph['transformer'].x = torch.tensor(t_features, dtype=torch.float)
                node_ids['transformer'] = t_ids
                self.node_mappings['transformer'] = {id_: i for i, id_ in enumerate(t_ids)}
        
        # Process Substations  
        substations = nodes.get('substations', [])
        if substations:
            s_features = []
            s_ids = []
            
            for i, s in enumerate(substations):
                s_id = s.get('station_id', f'S_{i}')
                if s_id is None:
                    s_id = f'S_{i}'
                s_ids.append(str(s_id))
                
                s_features.append([
                    1.0,
                    float(s.get('x', 0) or 0),
                    float(s.get('y', 0) or 0)
                ])
            
            if s_features:
                graph['substation'].x = torch.tensor(s_features, dtype=torch.float)
                node_ids['substation'] = s_ids
                self.node_mappings['substation'] = {id_: i for i, id_ in enumerate(s_ids)}
        

            # ADD THIS NEW CODE:
            else:
                # No substations found - create virtual one
                graph['substation'].x = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float)
                node_ids['substation'] = ['VIRTUAL_S0']
                self.node_mappings['substation'] = {'VIRTUAL_S0': 0}
                logger.info("Created virtual substation (no real substations found)")



        # Process Adjacency Clusters
        clusters = nodes.get('adjacency_clusters', [])
        if clusters:
            ac_features = []
            ac_ids = []
            
            sharing_map = {'LOW': 0.2, 'MEDIUM': 0.5, 'HIGH': 0.8, 'VERY_HIGH': 1.0}
            benefit_map = {'LOW': 0.2, 'MEDIUM': 0.5, 'HIGH': 0.8}
            
            for i, ac in enumerate(clusters):
                ac_id = str(ac.get('cluster_id', f'AC_{i}'))
                if ac_id is None or ac_id == 'None':
                    ac_id = f'AC_{i}'
                ac_ids.append(str(ac_id))
                
                def safe_float(value, default=0.0):
                    if value is None:
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                
                sharing_potential = sharing_map.get(
                    ac.get('energy_sharing_potential', 'LOW'), 0.2
                )
                thermal_benefit = benefit_map.get(
                    ac.get('thermal_benefit', 'LOW'), 0.2
                )
                cable_savings = benefit_map.get(
                    ac.get('cable_savings', 'LOW'), 0.2
                )
                
                ac_features.append([
                    safe_float(ac.get('member_count', 0)),
                    sharing_potential,
                    safe_float(ac.get('solar_penetration', 0)),
                    safe_float(ac.get('hp_penetration', 0)),
                    safe_float(ac.get('battery_penetration', 0)),
                    thermal_benefit,
                    cable_savings,
                    safe_float(ac.get('total_demand_kw', 0)),
                    safe_float(ac.get('export_potential_kw', 0)),
                    safe_float(ac.get('self_sufficiency_ratio', 0)),
                    safe_float(ac.get('sharing_benefit_kwh', 0))
                ])
            
            if ac_features:
                graph['adjacency_cluster'].x = torch.tensor(ac_features, dtype=torch.float)
                node_ids['adjacency_cluster'] = ac_ids
                self.node_mappings['adjacency_cluster'] = {id_: i for i, id_ in enumerate(ac_ids)}
                
                # Add temporal features for clusters if requested
                if include_temporal and ac_ids:
                    logger.info(f"Fetching temporal features for {len(ac_ids)} clusters...")
                    cluster_temporal = []
                    
                    for ac_id in ac_ids:
                        cluster_series = self.kg.get_cluster_time_series(
                            ac_id, 
                            lookback_hours=lookback_hours
                        )
                        
                        if cluster_series and len(cluster_series) >= lookback_hours:
                            cluster_array = np.array([
                                [
                                    c.get('hour', 0) / 24.0,
                                    c.get('total_demand', 0),
                                    c.get('total_solar', 0),
                                    c.get('total_export', 0),
                                    c.get('total_deficit', 0),
                                    c.get('total_surplus', 0),
                                    c.get('sharing_potential', 0)
                                ]
                                for c in cluster_series[:lookback_hours]
                            ])
                            cluster_temporal.append(cluster_array)
                        else:
                            cluster_temporal.append(np.zeros((lookback_hours, 7)))
                            logger.debug(f"No temporal data for cluster {ac_id}")
                    
                    if cluster_temporal:
                        graph['adjacency_cluster'].x_temporal = torch.tensor(
                            np.stack(cluster_temporal),
                            dtype=torch.float
                        )
                        logger.info(f"Added cluster temporal features: {graph['adjacency_cluster'].x_temporal.shape}")
        
        # Store node IDs in graph metadata (not as node attributes)
        graph.node_ids = node_ids
        # CRITICAL: Final verification that all features are tensors
        for node_type in ['building', 'cable_group', 'transformer', 'substation', 'adjacency_cluster']:
            if node_type in graph.node_types:
                # Check and convert regular features
                if hasattr(graph[node_type], 'x'):
                    features = graph[node_type].x
                    if not isinstance(features, torch.Tensor):
                        logger.warning(f"Converting {node_type} features from {type(features)} to tensor")
                        if isinstance(features, list):
                            graph[node_type].x = torch.tensor(features, dtype=torch.float)
                        elif isinstance(features, np.ndarray):
                            graph[node_type].x = torch.tensor(features, dtype=torch.float)
                        else:
                            logger.error(f"Unknown feature type for {node_type}: {type(features)}")
                    
                    # Verify the conversion worked
                    if isinstance(graph[node_type].x, torch.Tensor):
                        logger.debug(f"✓ {node_type} features are tensors: shape {graph[node_type].x.shape}")
                    else:
                        logger.error(f"✗ Failed to convert {node_type} features to tensor")
                
                # Check and convert temporal features if they exist
                if hasattr(graph[node_type], 'x_temporal'):
                    temporal = graph[node_type].x_temporal
                    if not isinstance(temporal, torch.Tensor):
                        logger.warning(f"Converting {node_type} temporal features to tensor")
                        if isinstance(temporal, np.ndarray):
                            graph[node_type].x_temporal = torch.tensor(temporal, dtype=torch.float)
                        elif isinstance(temporal, list):
                            graph[node_type].x_temporal = torch.tensor(np.array(temporal), dtype=torch.float)
                        else:
                            logger.error(f"Unknown temporal feature type for {node_type}: {type(temporal)}")



                  
    # UPDATE in graph_constructor.py - around line 420-450 in _add_edges_to_graph method

    def _add_edges_to_graph(self, graph: HeteroData, edges: Dict[str, List]):
        """Add edges to graph - FIXED to handle missing edges."""
        
        # Building to Cable Group edges
        if 'building_to_cable' in edges and edges['building_to_cable']:
            edge_index = self._create_edge_index(
                edges['building_to_cable'], 'building', 'cable_group'
            )
            if edge_index is not None and edge_index.numel() > 0:
                graph['building', 'connected_to', 'cable_group'].edge_index = edge_index
                logger.debug(f"Added {edge_index.shape[1]} building->cable_group edges")
        
        # Cable Group to Transformer edges
        if 'cable_to_transformer' in edges and edges['cable_to_transformer']:
            edge_index = self._create_edge_index(
                edges['cable_to_transformer'], 'cable_group', 'transformer'
            )
            if edge_index is not None and edge_index.numel() > 0:
                graph['cable_group', 'connects_to', 'transformer'].edge_index = edge_index
                logger.debug(f"Added {edge_index.shape[1]} cable_group->transformer edges")
        else:
            logger.warning("No cable_to_transformer edges found")
        
        # Transformer to Substation edges (might not exist)
        if 'transformer_to_substation' in edges and edges['transformer_to_substation']:
            edge_index = self._create_edge_index(
                edges['transformer_to_substation'], 'transformer', 'substation'
            )
            if edge_index is not None and edge_index.numel() > 0:
                graph['transformer', 'feeds_from', 'substation'].edge_index = edge_index
                logger.debug(f"Added {edge_index.shape[1]} transformer->substation edges")
        else:
            logger.info("No transformer_to_substation edges (substations might not exist)")
        
        # Building to Adjacency Cluster edges
        if 'building_to_cluster' in edges and edges['building_to_cluster']:
            edge_index = self._create_edge_index(
                edges['building_to_cluster'], 'building', 'adjacency_cluster'
            )
            if edge_index is not None and edge_index.numel() > 0:
                graph['building', 'in_cluster', 'adjacency_cluster'].edge_index = edge_index
                logger.debug(f"Added {edge_index.shape[1]} building->cluster edges")



    def _create_edge_index(self, edge_list: List[Dict], 
                          src_type: str, dst_type: str) -> Optional[torch.Tensor]:
        """Convert edge list to PyG edge_index tensor."""
        if not edge_list:
            return None
            
        edges = []
        src_mapping = self.node_mappings.get(src_type, {})
        dst_mapping = self.node_mappings.get(dst_type, {})
        
        for edge in edge_list:
            src_id = str(edge['src'])
            dst_id = str(edge['dst'])
            
            if src_id in src_mapping and dst_id in dst_mapping:
                edges.append([src_mapping[src_id], dst_mapping[dst_id]])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
        return None
    

    def _build_retrofit_graph(self, district_name: str, 
                            energy_labels: List[str] = ['E', 'F', 'G'],
                            age_filter: str = '19') -> HeteroData:
        """Build graph for retrofit targeting task."""
        
        # Get retrofit candidates
        candidates = self.kg.get_retrofit_candidates(district_name, energy_labels, age_filter)
        
        # Build full graph first
        graph = self.build_hetero_graph(district_name)
        
        # Add retrofit labels - FIX: Access IDs from graph.node_ids
        if hasattr(graph, 'node_ids') and 'building' in graph.node_ids:
            building_ids = graph.node_ids['building']
        else:
            # Fallback if node_ids not available
            logger.warning("No building IDs found in graph.node_ids")
            building_ids = []
        
        retrofit_labels = []
        
        for b_id in building_ids:
            # Check if building is in any retrofit candidate cable group
            is_candidate = any(
                any(b.get('ogc_fid') == int(b_id) if b_id.isdigit() else False 
                    for b in buildings)
                for buildings in candidates.values()
            )
            retrofit_labels.append(1 if is_candidate else 0)
        
        graph['building'].y = torch.tensor(retrofit_labels, dtype=torch.float)
        graph.task = 'retrofit'
        
        return graph
    
    
    def _build_energy_sharing_graph(self, district_name: str,
                                min_cluster_size: int = 3) -> HeteroData:
        """Build graph for energy sharing analysis."""
        
        # Get adjacency clusters
        clusters = self.kg.get_adjacency_clusters(district_name, min_cluster_size)
        
        # Build graph with cluster edges  
        graph = self.build_hetero_graph(district_name, include_energy_sharing=True)
        
        # Add cluster sharing potential as labels - FIX: Access IDs from graph.node_ids
        if 'adjacency_cluster' in graph.node_types and hasattr(graph, 'node_ids') and 'adjacency_cluster' in graph.node_ids:
            cluster_ids = graph.node_ids['adjacency_cluster']
            sharing_scores = []
            
            # Map string values to numeric
            potential_map = {
                'LOW': 0.2,
                'MEDIUM': 0.5, 
                'HIGH': 0.8,
                'VERY_HIGH': 1.0,
                None: 0.0
            }
            
            for c_id in cluster_ids:
                # Find cluster data
                cluster_data = None
                for c in clusters:
                    if str(c.get('cluster_id', '')) == str(c_id):
                        cluster_data = c
                        break
                
                if cluster_data:
                    # Get string value and convert to numeric
                    potential_str = cluster_data.get('sharing_potential', 'LOW')
                    potential_numeric = potential_map.get(potential_str, 0.0)
                    sharing_scores.append(potential_numeric)
                else:
                    sharing_scores.append(0.0)
            
            graph['adjacency_cluster'].y = torch.tensor(sharing_scores, dtype=torch.float)
        
        graph.task = 'energy_sharing'
        
        return graph

    
    def _build_solar_graph(self, district_name: str) -> HeteroData:
        """Build graph for solar optimization."""
        
        # Build base graph
        graph = self.build_hetero_graph(district_name)
        
        # Add solar potential as target
        building_features = graph['building'].x
        # Solar potential is feature index 2, roof area is index 5
        solar_scores = building_features[:, 2]
        roof_areas = building_features[:, 5]
        
        # Calculate solar generation potential
        solar_potential = roof_areas * solar_scores * 0.15 * 1000  # kWh/year
        
        graph['building'].y = solar_potential
        graph.task = 'solar'
        
        return graph
    
    def _build_electrification_graph(self, district_name: str) -> HeteroData:
        """Build graph for electrification planning."""
        
        # Build base graph
        graph = self.build_hetero_graph(district_name)
        
        # Add electrification readiness as target
        building_features = graph['building'].x
        # Electrification feasibility is feature index 3
        electrify_scores = building_features[:, 3]
        
        # Binary classification: ready for electrification (score >= 2)
        electrify_ready = (electrify_scores >= 2).float()
        
        graph['building'].y = electrify_ready
        graph.task = 'electrification'
        
        return graph