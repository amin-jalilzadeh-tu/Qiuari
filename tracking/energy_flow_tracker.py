"""
Energy Flow Tracking Module
Detailed tracking of energy flows, transactions, and grid impacts with timestamps
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class EnergyFlowTracker:
    """
    Track detailed energy flows between buildings, communities, and grid
    """
    
    def __init__(self, output_dir: str = "results/energy_flows"):
        """Initialize energy flow tracker"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Flow records storage
        self.flow_records = []
        self.transaction_log = []
        self.grid_impact_timeline = []
        self.community_exchanges = []
        
        # Real-time tracking
        self.current_flows = {}
        self.peak_events = []
        self.congestion_events = []
        
    def record_flow(
        self,
        timestamp: datetime,
        from_id: str,
        to_id: str,
        energy_kwh: float,
        flow_type: str,  # 'p2p', 'to_grid', 'from_grid', 'community'
        cluster_id: Optional[int] = None,
        price_per_kwh: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Record a single energy flow transaction
        
        Args:
            timestamp: Time of the flow
            from_id: Source building/entity ID
            to_id: Destination building/entity ID
            energy_kwh: Amount of energy transferred
            flow_type: Type of energy flow
            cluster_id: Community cluster ID if applicable
            price_per_kwh: Price for the transaction
            metadata: Additional information
        """
        flow_record = {
            'timestamp': timestamp.isoformat(),
            'from_id': from_id,
            'to_id': to_id,
            'energy_kwh': float(energy_kwh),
            'flow_type': flow_type,
            'cluster_id': int(cluster_id) if cluster_id is not None else None,
            'price_per_kwh': float(price_per_kwh) if price_per_kwh else 0.0,
            'total_value': float(energy_kwh * (price_per_kwh or 0.0)),
            'metadata': metadata or {}
        }
        
        self.flow_records.append(flow_record)
        
        # Update current flows for real-time monitoring
        flow_key = f"{from_id}->{to_id}"
        if flow_key not in self.current_flows:
            self.current_flows[flow_key] = {
                'total_energy': 0.0,
                'total_value': 0.0,
                'count': 0
            }
        
        self.current_flows[flow_key]['total_energy'] += energy_kwh
        self.current_flows[flow_key]['total_value'] += flow_record['total_value']
        self.current_flows[flow_key]['count'] += 1
    
    def record_community_exchange(
        self,
        timestamp: datetime,
        cluster_id: int,
        generation_kwh: float,
        consumption_kwh: float,
        shared_kwh: float,
        grid_import_kwh: float,
        grid_export_kwh: float,
        self_sufficiency: float,
        participating_buildings: List[str]
    ):
        """
        Record community-level energy exchange metrics
        """
        exchange = {
            'timestamp': timestamp.isoformat(),
            'cluster_id': int(cluster_id),
            'generation_kwh': float(generation_kwh),
            'consumption_kwh': float(consumption_kwh),
            'shared_kwh': float(shared_kwh),
            'grid_import_kwh': float(grid_import_kwh),
            'grid_export_kwh': float(grid_export_kwh),
            'self_sufficiency': float(self_sufficiency),
            'n_buildings': len(participating_buildings),
            'building_ids': participating_buildings
        }
        
        self.community_exchanges.append(exchange)
        
    def record_grid_impact(
        self,
        timestamp: datetime,
        transformer_id: str,
        load_kw: float,
        capacity_kw: float,
        voltage_pu: float,
        losses_kw: float,
        congestion_level: float  # 0-1 scale
    ):
        """
        Record grid impact metrics at transformer level
        """
        impact = {
            'timestamp': timestamp.isoformat(),
            'transformer_id': transformer_id,
            'load_kw': float(load_kw),
            'capacity_kw': float(capacity_kw),
            'utilization': float(load_kw / capacity_kw) if capacity_kw > 0 else 0.0,
            'voltage_pu': float(voltage_pu),
            'losses_kw': float(losses_kw),
            'congestion_level': float(congestion_level)
        }
        
        self.grid_impact_timeline.append(impact)
        
        # Track congestion events
        if congestion_level > 0.8:
            self.congestion_events.append({
                'timestamp': timestamp.isoformat(),
                'transformer_id': transformer_id,
                'severity': congestion_level,
                'load_kw': load_kw
            })
    
    def analyze_flow_patterns(self, cluster_assignments: torch.Tensor) -> Dict:
        """
        Analyze energy flow patterns and identify opportunities
        """
        if not self.flow_records:
            return {}
        
        df = pd.DataFrame(self.flow_records)
        
        analysis = {
            'total_flows': len(df),
            'total_energy_transferred': df['energy_kwh'].sum(),
            'total_value_created': df['total_value'].sum(),
            'flow_types': df['flow_type'].value_counts().to_dict(),
            'average_transaction_size': df['energy_kwh'].mean(),
            'peak_hour_flows': self._identify_peak_hours(df),
            'most_active_pairs': self._get_active_pairs(df),
            'community_performance': self._analyze_communities(df)
        }
        
        return analysis
    
    def _identify_peak_hours(self, df: pd.DataFrame) -> List[int]:
        """Identify hours with most energy flows"""
        if 'timestamp' not in df.columns or df.empty:
            return []
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_flows = df.groupby('hour')['energy_kwh'].sum()
        
        # Get top 3 peak hours
        peak_hours = hourly_flows.nlargest(3).index.tolist()
        return peak_hours
    
    def _get_active_pairs(self, df: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """Get most active building pairs for energy trading"""
        if df.empty:
            return []
        
        # Group by building pairs
        pairs = df.groupby(['from_id', 'to_id']).agg({
            'energy_kwh': 'sum',
            'total_value': 'sum',
            'flow_type': 'first'
        }).reset_index()
        
        # Sort by energy transferred
        pairs = pairs.nlargest(top_n, 'energy_kwh')
        
        return pairs.to_dict('records')
    
    def _analyze_communities(self, df: pd.DataFrame) -> Dict:
        """Analyze performance by community"""
        if 'cluster_id' not in df.columns or df.empty:
            return {}
        
        community_stats = {}
        for cluster_id in df['cluster_id'].dropna().unique():
            cluster_df = df[df['cluster_id'] == cluster_id]
            
            community_stats[int(cluster_id)] = {
                'total_shared': cluster_df[cluster_df['flow_type'] == 'community']['energy_kwh'].sum(),
                'n_transactions': len(cluster_df),
                'avg_transaction': cluster_df['energy_kwh'].mean(),
                'total_value': cluster_df['total_value'].sum()
            }
        
        return community_stats
    
    def generate_flow_report(self, save_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive energy flow report
        """
        report = {
            'summary': {
                'total_transactions': len(self.flow_records),
                'total_energy_transferred_mwh': sum(f['energy_kwh'] for f in self.flow_records) / 1000,
                'total_value_created': sum(f['total_value'] for f in self.flow_records),
                'n_congestion_events': len(self.congestion_events),
                'n_communities_tracked': len(set(e['cluster_id'] for e in self.community_exchanges))
            },
            'flow_analysis': self.analyze_flow_patterns(None),
            'grid_impacts': self._summarize_grid_impacts(),
            'community_performance': self._summarize_communities(),
            'peak_events': self.peak_events[-10:],  # Last 10 peak events
            'recommendations': self._generate_recommendations()
        }
        
        if save_path:
            save_path = Path(save_path)
        else:
            save_path = self.output_dir / f'flow_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Energy flow report saved to {save_path}")
        return report
    
    def _summarize_grid_impacts(self) -> Dict:
        """Summarize grid impact metrics"""
        if not self.grid_impact_timeline:
            return {}
        
        df = pd.DataFrame(self.grid_impact_timeline)
        
        return {
            'avg_utilization': df['utilization'].mean(),
            'peak_utilization': df['utilization'].max(),
            'total_losses_kwh': df['losses_kw'].sum() * 0.25,  # Assuming 15-min intervals
            'avg_voltage_pu': df['voltage_pu'].mean(),
            'congestion_hours': len(df[df['congestion_level'] > 0.8]),
            'transformers_monitored': df['transformer_id'].nunique()
        }
    
    def _summarize_communities(self) -> Dict:
        """Summarize community exchange performance"""
        if not self.community_exchanges:
            return {}
        
        df = pd.DataFrame(self.community_exchanges)
        
        summary = {}
        for cluster_id in df['cluster_id'].unique():
            cluster_df = df[df['cluster_id'] == cluster_id]
            
            summary[int(cluster_id)] = {
                'avg_self_sufficiency': cluster_df['self_sufficiency'].mean(),
                'total_shared_mwh': cluster_df['shared_kwh'].sum() / 1000,
                'avg_buildings': cluster_df['n_buildings'].mean(),
                'peak_generation': cluster_df['generation_kwh'].max(),
                'peak_consumption': cluster_df['consumption_kwh'].max()
            }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on flow analysis"""
        recommendations = []
        
        # Check congestion
        if len(self.congestion_events) > 10:
            recommendations.append(f"High congestion detected ({len(self.congestion_events)} events) - consider battery storage or demand response")
        
        # Check flow efficiency
        if self.flow_records:
            df = pd.DataFrame(self.flow_records)
            grid_flows = df[df['flow_type'].isin(['to_grid', 'from_grid'])]
            if len(grid_flows) > len(df) * 0.5:
                recommendations.append("Over 50% of flows involve grid - increase local sharing")
        
        # Check community performance
        if self.community_exchanges:
            df = pd.DataFrame(self.community_exchanges)
            low_performers = df[df['self_sufficiency'] < 0.3]['cluster_id'].unique()
            if len(low_performers) > 0:
                recommendations.append(f"Communities {low_performers.tolist()} have low self-sufficiency - consider solar additions")
        
        if not recommendations:
            recommendations.append("System operating efficiently - monitor for seasonal changes")
        
        return recommendations
    
    def visualize_flows(self, cluster_assignments: torch.Tensor, save_path: Optional[str] = None):
        """
        Create flow visualization (Sankey diagram data)
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        import matplotlib.patches as mpatches
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Energy Flow Visualization', fontsize=16, fontweight='bold')
        
        # Left plot: Flow distribution by type
        if self.flow_records:
            df = pd.DataFrame(self.flow_records)
            flow_types = df.groupby('flow_type')['energy_kwh'].sum()
            
            colors = {
                'p2p': '#2ecc71',
                'community': '#3498db', 
                'to_grid': '#e74c3c',
                'from_grid': '#f39c12'
            }
            
            ax1.pie(flow_types.values, labels=flow_types.index, autopct='%1.1f%%',
                   colors=[colors.get(ft, '#95a5a6') for ft in flow_types.index])
            ax1.set_title('Energy Flow Distribution by Type')
        
        # Right plot: Hourly flow pattern
        if self.flow_records:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hourly = df.groupby('hour')['energy_kwh'].sum()
            
            ax2.bar(hourly.index, hourly.values, color='#3498db', edgecolor='navy')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Energy Transferred (kWh)')
            ax2.set_title('Hourly Energy Flow Pattern')
            ax2.grid(True, alpha=0.3)
            
            # Mark peak hours
            peak_hours = hourly.nlargest(3).index
            for ph in peak_hours:
                ax2.axvline(x=ph, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        else:
            save_path = self.output_dir / f'flow_visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.close()
        logger.info(f"Flow visualization saved to {save_path}")