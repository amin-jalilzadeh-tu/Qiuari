"""
Data Aggregation Pipeline
Consolidates data from Neo4j, GNN outputs, and simulations
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle

logger = logging.getLogger(__name__)


@dataclass
class AggregatedMetrics:
    """Consolidated metrics for reporting"""
    timestamp: str
    
    # Cluster metrics
    num_clusters: int
    avg_cluster_size: float
    cluster_stability: float
    avg_self_sufficiency: float
    avg_complementarity: float
    total_peak_reduction: float
    
    # Energy metrics
    total_demand_mwh: float
    total_generation_mwh: float
    total_shared_energy_mwh: float
    grid_import_mwh: float
    grid_export_mwh: float
    
    # Solar metrics
    num_solar_buildings: int
    total_solar_capacity_kw: float
    avg_solar_roi_years: float
    solar_coverage_percent: float
    
    # Economic metrics
    total_cost_savings_eur: float
    avg_cost_reduction_percent: float
    carbon_reduction_tons: float
    peak_charge_savings_eur: float
    
    # Network metrics
    avg_voltage_deviation: float
    transformer_utilization_percent: float
    line_loss_percent: float
    congestion_events: int
    
    # Building distribution
    energy_label_distribution: Dict[str, int]
    building_type_distribution: Dict[str, int]
    
    # LV group metrics
    num_lv_groups: int
    avg_buildings_per_lv: float
    lv_groups_with_clusters: int


class DataAggregator:
    """Aggregates data from multiple sources for reporting"""
    
    def __init__(self, results_dir: str = "results", neo4j_connector=None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.neo4j = neo4j_connector
        
        # Create subdirectories
        (self.results_dir / "data").mkdir(exist_ok=True)
        (self.results_dir / "reports").mkdir(exist_ok=True)
        (self.results_dir / "charts").mkdir(exist_ok=True)
        (self.results_dir / "dashboards").mkdir(exist_ok=True)
        
        # Cache for aggregated data
        self.cache = {}
        
    def aggregate_all_metrics(self, 
                             gnn_results: Dict,
                             cluster_metrics: Optional[Dict] = None,
                             solar_results: Optional[Dict] = None,
                             temporal_data: Optional[pd.DataFrame] = None) -> AggregatedMetrics:
        """
        Aggregate all metrics from different sources
        """
        logger.info("Starting comprehensive data aggregation...")
        
        # Extract cluster information
        cluster_data = self._aggregate_cluster_metrics(gnn_results, cluster_metrics)
        
        # Extract energy flows
        energy_data = self._aggregate_energy_metrics(gnn_results, temporal_data)
        
        # Extract solar information
        solar_data = self._aggregate_solar_metrics(solar_results, gnn_results)
        
        # Extract economic metrics
        economic_data = self._calculate_economic_metrics(
            cluster_data, energy_data, solar_data
        )
        
        # Extract network metrics
        network_data = self._aggregate_network_metrics(gnn_results)
        
        # Get building distributions
        distributions = self._get_building_distributions(gnn_results)
        
        # Get LV group statistics
        lv_stats = self._get_lv_group_statistics(gnn_results)
        
        # Create consolidated metrics
        metrics = AggregatedMetrics(
            timestamp=datetime.now().isoformat(),
            
            # Cluster metrics
            num_clusters=cluster_data.get('num_clusters', 0),
            avg_cluster_size=cluster_data.get('avg_size', 0),
            cluster_stability=cluster_data.get('stability', 0),
            avg_self_sufficiency=cluster_data.get('self_sufficiency', 0),
            avg_complementarity=cluster_data.get('complementarity', 0),
            total_peak_reduction=cluster_data.get('peak_reduction', 0),
            
            # Energy metrics
            total_demand_mwh=energy_data.get('total_demand', 0) / 1000,
            total_generation_mwh=energy_data.get('total_generation', 0) / 1000,
            total_shared_energy_mwh=energy_data.get('shared_energy', 0) / 1000,
            grid_import_mwh=energy_data.get('grid_import', 0) / 1000,
            grid_export_mwh=energy_data.get('grid_export', 0) / 1000,
            
            # Solar metrics
            num_solar_buildings=solar_data.get('num_solar', 0),
            total_solar_capacity_kw=solar_data.get('total_capacity', 0),
            avg_solar_roi_years=solar_data.get('avg_roi', 0),
            solar_coverage_percent=solar_data.get('coverage', 0),
            
            # Economic metrics
            total_cost_savings_eur=economic_data.get('total_savings', 0),
            avg_cost_reduction_percent=economic_data.get('cost_reduction', 0),
            carbon_reduction_tons=economic_data.get('carbon_reduction', 0),
            peak_charge_savings_eur=economic_data.get('peak_savings', 0),
            
            # Network metrics
            avg_voltage_deviation=network_data.get('voltage_deviation', 0),
            transformer_utilization_percent=network_data.get('transformer_util', 0),
            line_loss_percent=network_data.get('line_losses', 0),
            congestion_events=network_data.get('congestion_events', 0),
            
            # Distributions
            energy_label_distribution=distributions.get('energy_labels', {}),
            building_type_distribution=distributions.get('building_types', {}),
            
            # LV group metrics
            num_lv_groups=lv_stats.get('num_lv_groups', 0),
            avg_buildings_per_lv=lv_stats.get('avg_buildings_per_lv', 0),
            lv_groups_with_clusters=lv_stats.get('lv_with_clusters', 0)
        )
        
        # Save aggregated data
        self._save_aggregated_data(metrics)
        
        return metrics
    
    def _aggregate_cluster_metrics(self, gnn_results: Dict, 
                                  cluster_metrics: Optional[Dict]) -> Dict:
        """Extract and aggregate cluster-related metrics"""
        data = {}
        
        # From GNN results
        if 'cluster_assignments' in gnn_results:
            clusters = gnn_results['cluster_assignments']
            # Handle nested lists or tensors
            if isinstance(clusters, list):
                # Flatten if nested
                if clusters and isinstance(clusters[0], list):
                    clusters = [item for sublist in clusters for item in sublist]
                try:
                    unique_clusters = len(set(clusters))
                except TypeError:
                    unique_clusters = 0
            else:
                unique_clusters = 0
            data['num_clusters'] = unique_clusters
            
            if unique_clusters > 0:
                cluster_sizes = pd.Series(clusters).value_counts()
                data['avg_size'] = cluster_sizes.mean()
                data['min_size'] = cluster_sizes.min()
                data['max_size'] = cluster_sizes.max()
        
        # From cluster metrics
        if cluster_metrics:
            stabilities = []
            self_suffs = []
            complements = []
            peak_reds = []
            
            for cluster_id, metrics in cluster_metrics.items():
                if hasattr(metrics, 'temporal_stability'):
                    stabilities.append(metrics.temporal_stability)
                if hasattr(metrics, 'self_sufficiency_ratio'):
                    self_suffs.append(metrics.self_sufficiency_ratio)
                if hasattr(metrics, 'complementarity_score'):
                    complements.append(metrics.complementarity_score)
                if hasattr(metrics, 'peak_reduction_ratio'):
                    peak_reds.append(metrics.peak_reduction_ratio)
            
            data['stability'] = np.mean(stabilities) if stabilities else 0
            data['self_sufficiency'] = np.mean(self_suffs) if self_suffs else 0
            data['complementarity'] = np.mean(complements) if complements else 0
            data['peak_reduction'] = np.mean(peak_reds) if peak_reds else 0
        
        return data
    
    def _aggregate_energy_metrics(self, gnn_results: Dict, 
                                 temporal_data: Optional[pd.DataFrame]) -> Dict:
        """Extract and aggregate energy-related metrics"""
        data = {
            'total_demand': 0,
            'total_generation': 0,
            'shared_energy': 0,
            'grid_import': 0,
            'grid_export': 0
        }
        
        if temporal_data is not None and not temporal_data.empty:
            if 'demand' in temporal_data.columns:
                data['total_demand'] = temporal_data['demand'].sum()
            if 'generation' in temporal_data.columns:
                data['total_generation'] = temporal_data['generation'].sum()
            
            # Calculate energy sharing within clusters
            if 'cluster_id' in temporal_data.columns:
                for cluster_id in temporal_data['cluster_id'].unique():
                    cluster_data = temporal_data[temporal_data['cluster_id'] == cluster_id]
                    cluster_demand = cluster_data['demand'].sum() if 'demand' in cluster_data else 0
                    cluster_gen = cluster_data['generation'].sum() if 'generation' in cluster_data else 0
                    data['shared_energy'] += min(cluster_demand, cluster_gen)
            
            # Grid interactions
            data['grid_import'] = max(0, data['total_demand'] - data['total_generation'])
            data['grid_export'] = max(0, data['total_generation'] - data['total_demand'])
        
        # From GNN results
        if 'energy_flows' in gnn_results:
            flows = gnn_results['energy_flows']
            if isinstance(flows, dict):
                data['shared_energy'] = flows.get('total_shared', data['shared_energy'])
        
        return data
    
    def _aggregate_solar_metrics(self, solar_results: Optional[Dict], 
                                gnn_results: Dict) -> Dict:
        """Extract and aggregate solar-related metrics"""
        data = {
            'num_solar': 0,
            'total_capacity': 0,
            'avg_roi': 0,
            'coverage': 0
        }
        
        if solar_results:
            if 'installations' in solar_results:
                data['num_solar'] = len(solar_results['installations'])
            
            if 'total_capacity_kw' in solar_results:
                data['total_capacity'] = solar_results['total_capacity_kw']
            
            if 'roi_years' in solar_results:
                roi_values = solar_results['roi_years']
                if isinstance(roi_values, list) and roi_values:
                    data['avg_roi'] = np.mean(roi_values)
            
            # Coverage calculation
            if 'total_buildings' in solar_results:
                total = solar_results['total_buildings']
                if total > 0:
                    data['coverage'] = (data['num_solar'] / total) * 100
        
        return data
    
    def _calculate_economic_metrics(self, cluster_data: Dict, 
                                   energy_data: Dict, 
                                   solar_data: Dict) -> Dict:
        """Calculate economic and environmental metrics"""
        data = {}
        
        # Energy cost savings (simplified)
        electricity_price = 0.25  # EUR/kWh
        feed_in_tariff = 0.08  # EUR/kWh
        
        shared_savings = energy_data.get('shared_energy', 0) * electricity_price * 0.1  # 10% savings
        export_revenue = energy_data.get('grid_export', 0) * feed_in_tariff
        
        data['total_savings'] = shared_savings + export_revenue
        
        # Peak charge savings
        peak_reduction = cluster_data.get('peak_reduction', 0)
        peak_charge = 50  # EUR/kW/month
        avg_peak = energy_data.get('total_demand', 0) / 720  # Monthly hours
        data['peak_savings'] = peak_reduction * avg_peak * peak_charge
        
        # Cost reduction percentage
        total_cost = energy_data.get('total_demand', 0) * electricity_price
        if total_cost > 0:
            data['cost_reduction'] = (data['total_savings'] / total_cost) * 100
        else:
            data['cost_reduction'] = 0
        
        # Carbon reduction (0.5 kg CO2/kWh for grid electricity)
        renewable_energy = energy_data.get('total_generation', 0)
        data['carbon_reduction'] = (renewable_energy * 0.5) / 1000  # Convert to tons
        
        return data
    
    def _aggregate_network_metrics(self, gnn_results: Dict) -> Dict:
        """Extract network and grid metrics"""
        data = {
            'voltage_deviation': 0.02,  # 2% average
            'transformer_util': 65,  # 65% utilization
            'line_losses': 3.5,  # 3.5% losses
            'congestion_events': 0
        }
        
        # Extract from GNN results if available
        if 'network_metrics' in gnn_results:
            metrics = gnn_results['network_metrics']
            data.update(metrics)
        
        return data
    
    def _get_building_distributions(self, gnn_results: Dict) -> Dict:
        """Get distribution of building types and energy labels"""
        distributions = {
            'energy_labels': {},
            'building_types': {}
        }
        
        if 'building_features' in gnn_results:
            features = gnn_results['building_features']
            
            if 'energy_label' in features:
                labels = features['energy_label']
                if isinstance(labels, list):
                    distributions['energy_labels'] = pd.Series(labels).value_counts().to_dict()
            
            if 'building_type' in features:
                types = features['building_type']
                if isinstance(types, list):
                    distributions['building_types'] = pd.Series(types).value_counts().to_dict()
        
        return distributions
    
    def _get_lv_group_statistics(self, gnn_results: Dict) -> Dict:
        """Get LV group statistics"""
        stats = {
            'num_lv_groups': 0,
            'avg_buildings_per_lv': 0,
            'lv_with_clusters': 0
        }
        
        if 'lv_group_ids' in gnn_results:
            lv_ids = gnn_results['lv_group_ids']
            if isinstance(lv_ids, list):
                unique_lv = set(lv_ids)
                stats['num_lv_groups'] = len(unique_lv)
                
                # Count buildings per LV
                lv_counts = pd.Series(lv_ids).value_counts()
                stats['avg_buildings_per_lv'] = lv_counts.mean()
                
                # LV groups with sufficient buildings (>= 20)
                stats['lv_with_clusters'] = (lv_counts >= 20).sum()
        
        return stats
    
    def _save_aggregated_data(self, metrics: AggregatedMetrics):
        """Save aggregated metrics to files"""
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Save as JSON
        json_path = self.results_dir / "data" / f"aggregated_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        metrics_dict = convert_numpy(asdict(metrics))
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        
        # Save as pickle for Python use
        pkl_path = self.results_dir / "data" / f"aggregated_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        # Also save latest version
        latest_json = self.results_dir / "data" / "latest_metrics.json"
        with open(latest_json, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        
        logger.info(f"Aggregated metrics saved to {json_path}")
    
    def load_historical_data(self, days_back: int = 30) -> pd.DataFrame:
        """Load historical aggregated data for comparison"""
        data_files = list((self.results_dir / "data").glob("aggregated_metrics_*.json"))
        
        historical_data = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for file_path in data_files:
            try:
                # Extract date from filename
                date_str = file_path.stem.replace("aggregated_metrics_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                
                if file_date >= cutoff_date:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        historical_data.append(data)
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
        
        if historical_data:
            return pd.DataFrame(historical_data)
        else:
            return pd.DataFrame()
    
    def create_comparison_data(self, before_metrics: AggregatedMetrics, 
                              after_metrics: AggregatedMetrics) -> Dict:
        """Create before/after comparison data"""
        comparison = {}
        
        # Calculate improvements
        fields = ['avg_self_sufficiency', 'total_peak_reduction', 
                 'total_cost_savings_eur', 'carbon_reduction_tons']
        
        for field in fields:
            before_val = getattr(before_metrics, field, 0)
            after_val = getattr(after_metrics, field, 0)
            
            if before_val > 0:
                improvement = ((after_val - before_val) / before_val) * 100
            else:
                improvement = 100 if after_val > 0 else 0
            
            comparison[field] = {
                'before': before_val,
                'after': after_val,
                'improvement_percent': improvement
            }
        
        return comparison