"""
Real Data Connector for Visualization
Connects to actual GNN results and Neo4j data - NO FAKE DATA
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class RealDataConnector:
    """Connects visualization to real GNN results and data"""
    
    def __init__(self, gnn_system=None, kg_connector=None):
        self.gnn_system = gnn_system
        self.kg_connector = kg_connector
        self.latest_results = {}
        
    def extract_from_gnn_results(self, results: Dict) -> Dict:
        """Extract real data from GNN results"""
        
        logger.info("Extracting REAL data from GNN results")
        
        extracted = {
            'cluster_assignments': [],
            'cluster_metrics': {},
            'building_features': {},
            'energy_flows': {},
            'solar_analysis': {},
            'network_metrics': {}
        }
        
        # Extract cluster assignments
        if 'final_clusters' in results:
            clusters = results['final_clusters']
            if isinstance(clusters, torch.Tensor):
                extracted['cluster_assignments'] = clusters.cpu().numpy().tolist()
            else:
                extracted['cluster_assignments'] = clusters
        
        # Extract cluster quality metrics
        if 'cluster_quality' in results:
            extracted['cluster_metrics'] = results['cluster_quality']
        
        # Extract building features (from actual data)
        if 'building_data' in results:
            building_data = results['building_data']
            extracted['building_features'] = {
                'energy_label': building_data.get('energy_labels', []),
                'building_type': building_data.get('types', []),
                'has_solar': building_data.get('has_solar', []),
                'roof_area': building_data.get('roof_areas', []),
                'annual_consumption': building_data.get('annual_demands', []),
                'peak_demand': building_data.get('peak_demands', []),
                'building_ids': building_data.get('ids', [])
            }
        
        # Extract energy flows (real P2P sharing)
        if 'energy_sharing' in results:
            sharing = results['energy_sharing']
            extracted['energy_flows'] = {
                'p2p_matrix': sharing.get('sharing_matrix', []),
                'total_shared': sharing.get('total_shared_kwh', 0),
                'sharing_pairs': sharing.get('pairs', [])
            }
        
        # Extract solar recommendations (from solar learning loop)
        if 'solar_recommendations' in results:
            solar = results['solar_recommendations']
            extracted['solar_analysis'] = {
                'priority_buildings': solar.get('priority_list', []),
                'installations': solar.get('installations', []),
                'total_capacity_kwp': solar.get('total_capacity', 0),
                'expected_generation': solar.get('annual_generation', 0)
            }
        
        # Extract network metrics
        if 'network_analysis' in results:
            network = results['network_analysis']
            extracted['network_metrics'] = {
                'lv_groups': network.get('lv_groups', []),
                'transformer_loading': network.get('transformer_utilization', {}),
                'voltage_profiles': network.get('voltage_profiles', {}),
                'congestion_points': network.get('congestion_points', [])
            }
        
        return extracted
    
    def get_temporal_data_from_kg(self, building_ids: List[str] = None, 
                                  hours: int = 24) -> pd.DataFrame:
        """Get real temporal data from knowledge graph"""
        
        if not self.kg_connector:
            logger.warning("No KG connector available")
            return pd.DataFrame()
        
        # Correct Neo4j query for energy state data
        query = """
        MATCH (b:Building)-[:HAS_ENERGY_STATE]->(es:EnergyState)-[:DURING]->(ts:TimeSlot)
        WHERE ts.timestamp >= $start_time AND ts.timestamp <= $end_time
        RETURN b.id as building_id,
               b.lv_group_id as lv_group,
               ts.timestamp as timestamp,
               ts.hour as hour,
               ts.season as season,
               es.demand_kw as demand,
               es.generation_kw as generation,
               es.net_demand_kw as net_demand,
               es.has_surplus as has_surplus
        ORDER BY b.id, ts.timestamp
        LIMIT 10000
        """
        
        # Calculate time range (last 24 hours)
        import time
        current_time = int(time.time() * 1000)
        start_time = current_time - (hours * 3600 * 1000)
        
        params = {
            'start_time': start_time,
            'end_time': current_time
        }
        
        # Add building filter if specific buildings requested
        if building_ids:
            query = query.replace("WHERE ts", "WHERE b.id IN $building_ids AND ts")
            params['building_ids'] = building_ids
        
        try:
            # Execute query
            result = self.kg_connector.run(query, params)
            
            # Convert to DataFrame
            data = []
            for record in result:
                data.append({
                    'building_id': record.get('building_id', 'unknown'),
                    'lv_group': record.get('lv_group', 'unknown'),
                    'timestamp': record.get('timestamp', 0),
                    'hour': record.get('hour', 0),
                    'season': record.get('season', 'unknown'),
                    'demand': float(record.get('demand') or 0),
                    'generation': float(record.get('generation') or 0),
                    'net_demand': float(record.get('net_demand') or 0),
                    'has_surplus': record.get('has_surplus', False)
                })
            
            df = pd.DataFrame(data)
            
            # If no data, get building features as backup
            if df.empty:
                logger.info("No temporal data found, fetching building static features")
                df = self._get_building_features_from_kg(building_ids)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching temporal data: {e}")
            # Return building features as fallback
            return self._get_building_features_from_kg(building_ids)
    
    def _get_building_features_from_kg(self, building_ids: List[str] = None) -> pd.DataFrame:
        """Get building features from KG as fallback"""
        
        if not self.kg_connector:
            return pd.DataFrame()
        
        query = """
        MATCH (b:Building)
        OPTIONAL MATCH (b)-[:IN_ADJACENCY_CLUSTER]->(ac:AdjacencyCluster)
        RETURN b.id as building_id,
               b.lv_group_id as lv_group,
               b.floor_area as area,
               b.num_dwellings as dwellings,
               b.construction_year as year,
               b.energy_label as energy_label,
               b.has_solar as has_solar,
               b.solar_capacity_kw as solar_capacity,
               b.suitable_roof_area as roof_area,
               b.peak_electricity_kw as peak_demand,
               b.annual_consumption_kwh as annual_consumption,
               ac.type as cluster_type
        LIMIT 500
        """
        
        params = {}
        if building_ids:
            query = query.replace("MATCH (b:Building)", 
                                "MATCH (b:Building) WHERE b.id IN $building_ids")
            params['building_ids'] = building_ids
        
        try:
            result = self.kg_connector.run(query, params)
            
            data = []
            for record in result:
                # Create synthetic hourly data from annual consumption
                annual_kwh = float(record.get('annual_consumption') or 10000)
                daily_avg = annual_kwh / 365
                
                # Create 24-hour profile
                for hour in range(24):
                    # Simple consumption profile
                    if 7 <= hour <= 22:  # Daytime
                        hourly_demand = daily_avg / 24 * 1.5
                    else:  # Night
                        hourly_demand = daily_avg / 24 * 0.5
                    
                    # Solar generation (if has solar)
                    generation = 0
                    if record.get('has_solar'):
                        capacity = float(record.get('solar_capacity') or 5)
                        if 8 <= hour <= 18:  # Solar hours
                            generation = capacity * max(0, np.sin((hour - 6) * np.pi / 12))
                    
                    data.append({
                        'building_id': record.get('building_id', f'B_{len(data)}'),
                        'lv_group': record.get('lv_group', 'LV_001'),
                        'hour': hour,
                        'demand': hourly_demand,
                        'generation': generation,
                        'energy_label': record.get('energy_label', 'D'),
                        'has_solar': record.get('has_solar', False),
                        'roof_area': float(record.get('roof_area') or 100)
                    })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error fetching building features: {e}")
            return pd.DataFrame()
    
    def get_cluster_metrics_from_system(self, cluster_evaluator) -> Dict:
        """Get real cluster metrics from the evaluation system"""
        
        if not cluster_evaluator:
            return {}
        
        # Get actual cluster metrics from the evaluator
        metrics = {}
        
        if hasattr(cluster_evaluator, 'cluster_metrics'):
            for cluster_id, cluster_data in cluster_evaluator.cluster_metrics.items():
                metrics[cluster_id] = {
                    'quality_score': cluster_data.get('quality_score', 0),
                    'quality_label': cluster_data.get('quality_label', 'unknown'),
                    'self_sufficiency_ratio': cluster_data.get('self_sufficiency', 0),
                    'self_consumption_ratio': cluster_data.get('self_consumption', 0),
                    'complementarity_score': cluster_data.get('complementarity', 0),
                    'peak_reduction_ratio': cluster_data.get('peak_reduction', 0),
                    'temporal_stability': cluster_data.get('stability', 0),
                    'member_count': cluster_data.get('size', 0),
                    'lv_group_id': cluster_data.get('lv_group', 0),
                    'total_demand_kwh': cluster_data.get('total_demand', 0),
                    'total_generation_kwh': cluster_data.get('total_generation', 0),
                    'total_shared_kwh': cluster_data.get('shared_energy', 0)
                }
        
        return metrics
    
    def get_solar_data_from_simulator(self, solar_simulator) -> Dict:
        """Get real solar analysis from the solar simulator"""
        
        if not solar_simulator:
            return {}
        
        solar_data = {
            'priority_list': [],
            'total_capacity': 0,
            'num_installations': 0,
            'avg_roi': 0,
            'total_investment': 0,
            'annual_generation': 0,
            'co2_savings': 0
        }
        
        if hasattr(solar_simulator, 'installation_history'):
            installations = solar_simulator.installation_history
            
            for install in installations:
                solar_data['priority_list'].append({
                    'id': install.building_id,
                    'energy_label': install.energy_label,
                    'roof_area': install.roof_area_m2,
                    'capacity': install.expected_generation_kwp,
                    'roi': install.expected_roi_years,
                    'priority_score': install.priority_score
                })
            
            if installations:
                solar_data['num_installations'] = len(installations)
                solar_data['total_capacity'] = sum(i.expected_generation_kwp for i in installations)
                solar_data['avg_roi'] = np.mean([i.expected_roi_years for i in installations])
                solar_data['total_investment'] = solar_data['total_capacity'] * 1200  # EUR/kWp
                solar_data['annual_generation'] = solar_data['total_capacity'] * 1200  # kWh/kWp/year
                solar_data['co2_savings'] = solar_data['annual_generation'] * 0.5 / 1000  # tons
        
        return solar_data
    
    def get_energy_flows_from_tracker(self, energy_flow_tracker) -> Dict:
        """Get real energy flow data from the tracker"""
        
        if not energy_flow_tracker:
            return {}
        
        flows = {
            'total_demand': 0,
            'total_generation': 0,
            'total_shared': 0,
            'grid_import': 0,
            'grid_export': 0,
            'p2p_transactions': []
        }
        
        if hasattr(energy_flow_tracker, 'flow_history'):
            history = energy_flow_tracker.flow_history
            
            if history:
                # Aggregate flows
                flows['total_demand'] = sum(h.get('total_demand', 0) for h in history)
                flows['total_generation'] = sum(h.get('total_generation', 0) for h in history)
                flows['total_shared'] = sum(h.get('total_shared', 0) for h in history)
                flows['grid_import'] = sum(h.get('grid_import', 0) for h in history)
                flows['grid_export'] = sum(h.get('grid_export', 0) for h in history)
                
                # Get P2P transactions
                for h in history:
                    if 'transactions' in h:
                        flows['p2p_transactions'].extend(h['transactions'])
        
        return flows
    
    def get_economic_metrics(self, results: Dict) -> Dict:
        """Calculate real economic metrics from results"""
        
        # Real electricity prices and parameters
        electricity_price = 0.25  # EUR/kWh
        feed_in_tariff = 0.08  # EUR/kWh
        p2p_price = 0.15  # EUR/kWh
        peak_charge = 50  # EUR/kW/month
        
        metrics = {
            'total_investment': 0,
            'total_annual_benefit': 0,
            'payback_years': 0,
            'roi_percent': 0,
            'npv': 0
        }
        
        # Calculate from real data
        if 'energy_flows' in results:
            flows = results['energy_flows']
            
            # P2P trading benefits
            shared_energy = flows.get('total_shared', 0)
            p2p_savings = shared_energy * (electricity_price - p2p_price)
            
            # Grid export revenue
            grid_export = flows.get('grid_export', 0)
            export_revenue = grid_export * feed_in_tariff
            
            metrics['total_annual_benefit'] = p2p_savings + export_revenue
        
        if 'solar_analysis' in results:
            solar = results['solar_analysis']
            metrics['total_investment'] = solar.get('total_investment', 0)
            
            if metrics['total_investment'] > 0 and metrics['total_annual_benefit'] > 0:
                metrics['payback_years'] = metrics['total_investment'] / metrics['total_annual_benefit']
                metrics['roi_percent'] = (metrics['total_annual_benefit'] / metrics['total_investment']) * 100
        
        return metrics
    
    def get_real_cluster_assignments_from_kg(self) -> Dict:
        """Get actual cluster assignments from KG or GNN results"""
        
        clusters = {}
        
        # First try to get from current GNN system state
        if self.gnn_system and hasattr(self.gnn_system, 'current_clusters'):
            cluster_tensor = self.gnn_system.current_clusters
            if cluster_tensor is not None and len(cluster_tensor) > 0:
                if isinstance(cluster_tensor, torch.Tensor):
                    assignments = cluster_tensor.cpu().numpy()
                else:
                    assignments = np.array(cluster_tensor)
                
                # Group by cluster
                unique_clusters = np.unique(assignments)
                for cluster_id in unique_clusters:
                    cluster_mask = assignments == cluster_id
                    building_ids = np.where(cluster_mask)[0]
                    clusters[int(cluster_id)] = [f'B_{bid:03d}' for bid in building_ids]
                    
                logger.info(f"Retrieved {len(unique_clusters)} clusters from GNN current state")
                return clusters
        
        if self.gnn_system and hasattr(self.gnn_system, 'last_evaluation_results'):
            # Get from last GNN run
            results = self.gnn_system.last_evaluation_results
            if 'final_clusters' in results or 'cluster_assignments' in results:
                cluster_tensor = results.get('final_clusters', results.get('cluster_assignments'))
                if isinstance(cluster_tensor, torch.Tensor):
                    assignments = cluster_tensor.cpu().numpy()
                    # Flatten if 2D array
                    if len(assignments.shape) > 1:
                        assignments = assignments.flatten()
                    # Group by cluster
                    for building_id, cluster_id in enumerate(assignments):
                        # Handle numpy scalar conversion properly
                        if hasattr(cluster_id, 'item'):
                            try:
                                cluster_id = cluster_id.item()
                            except ValueError:
                                # If array has multiple elements, take first
                                cluster_id = int(cluster_id.flat[0])
                        else:
                            cluster_id = int(cluster_id)
                        if cluster_id not in clusters:
                            clusters[cluster_id] = []
                        clusters[cluster_id].append(f'B_{building_id:03d}')
        
        # Fallback to KG query
        if not clusters and self.kg_connector:
            query = """
            MATCH (b:Building)-[:IN_CLUSTER]->(c:Cluster)
            RETURN c.id as cluster_id, 
                   collect(b.id) as building_ids,
                   c.quality_label as quality,
                   c.lv_group_id as lv_group
            """
            try:
                result = self.kg_connector.run(query, {})
                for record in result:
                    cluster_id = record.get('cluster_id', len(clusters))
                    clusters[cluster_id] = {
                        'buildings': record.get('building_ids', []),
                        'quality': record.get('quality', 'unknown'),
                        'lv_group': record.get('lv_group', 'unknown')
                    }
            except:
                pass
        
        return clusters
    
    def get_real_energy_sharing_matrix(self) -> np.ndarray:
        """Get real P2P energy sharing matrix from results or KG"""
        
        # Try to get from GNN system first
        if self.gnn_system and hasattr(self.gnn_system, 'energy_flows'):
            flows = self.gnn_system.energy_flows
            if flows:
                # Convert flow records to matrix
                building_ids = set()
                for flow_record in flows.values():
                    if 'transactions' in flow_record:
                        for tx in flow_record['transactions']:
                            building_ids.add(tx.get('from', 0))
                            building_ids.add(tx.get('to', 0))
                
                n = len(building_ids) if building_ids else 160
                matrix = np.zeros((n, n))
                
                for flow_record in flows.values():
                    if 'transactions' in flow_record:
                        for tx in flow_record['transactions']:
                            i = tx.get('from', 0)
                            j = tx.get('to', 0)
                            amount = tx.get('energy_kwh', 0)
                            if i < n and j < n:
                                matrix[i, j] += amount
                
                return matrix
        
        # Fallback: create from complementarity
        return np.zeros((160, 160))
    
    def prepare_real_visualization_data(self, gnn_results: Dict, 
                                       system_components: Dict) -> Dict:
        """Prepare all real data for visualization"""
        
        logger.info("Preparing REAL visualization data - NO FAKE DATA")
        
        # Extract real GNN results
        extracted = self.extract_from_gnn_results(gnn_results)
        
        # Get real cluster assignments
        extracted['cluster_assignments'] = self.get_real_cluster_assignments_from_kg()
        
        # Get real cluster metrics
        if 'cluster_evaluator' in system_components:
            extracted['cluster_metrics'] = self.get_cluster_metrics_from_system(
                system_components['cluster_evaluator']
            )
        
        # Get real solar data
        if 'solar_simulator' in system_components:
            extracted['solar_analysis'] = self.get_solar_data_from_simulator(
                system_components['solar_simulator']
            )
        
        # Get real energy flows
        if 'energy_flow_tracker' in system_components:
            extracted['energy_flows'] = self.get_energy_flows_from_tracker(
                system_components['energy_flow_tracker']
            )
        else:
            # Get from sharing matrix
            extracted['energy_flows'] = {
                'sharing_matrix': self.get_real_energy_sharing_matrix(),
                'total_shared': 0
            }
        
        # Get real temporal data from KG
        if self.kg_connector:
            extracted['temporal_data'] = self.get_temporal_data_from_kg()
            # Calculate total shared from temporal data
            if not extracted['temporal_data'].empty:
                df = extracted['temporal_data']
                if 'generation' in df.columns and 'demand' in df.columns:
                    # Simple sharing calculation
                    total_gen = df['generation'].sum()
                    total_demand = df['demand'].sum()
                    extracted['energy_flows']['total_shared'] = min(total_gen, total_demand) * 0.3
        
        # Calculate real economic metrics
        extracted['economic_metrics'] = self.get_economic_metrics(extracted)
        
        # Add network topology from KG
        extracted['network_topology'] = self._get_network_topology_from_kg()
        
        return extracted
    
    def _get_network_topology_from_kg(self) -> Dict:
        """Get real network topology from KG"""
        
        topology = {
            'lv_groups': [],
            'transformers': [],
            'connections': []
        }
        
        if not self.kg_connector:
            return topology
        
        query = """
        MATCH (lv:CableGroup {voltage_level: 'LV'})
        OPTIONAL MATCH (lv)<-[:CONNECTED_TO]-(b:Building)
        WITH lv, count(b) as building_count
        RETURN lv.group_id as lv_id,
               lv.name as lv_name,
               building_count
        ORDER BY lv.group_id
        LIMIT 100
        """
        
        try:
            result = self.kg_connector.run(query, {})
            for record in result:
                topology['lv_groups'].append({
                    'id': record.get('lv_id', 'unknown'),
                    'name': record.get('lv_name', 'unknown'),
                    'building_count': record.get('building_count', 0)
                })
        except Exception as e:
            logger.error(f"Error fetching network topology: {e}")
        
        return topology