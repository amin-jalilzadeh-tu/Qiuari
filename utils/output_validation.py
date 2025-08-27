"""
Comprehensive output validation and structured reporting for Energy GNN.
Validates predictions against physics constraints and generates detailed reports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class PhysicsValidator:
    """
    Validates GNN outputs against physical constraints of energy systems.
    """
    
    def __init__(self, tolerance: float = 1e-3):
        """
        Initialize physics validator.
        
        Args:
            tolerance: Numerical tolerance for constraint violations
        """
        self.tolerance = tolerance
        self.violation_history = []
        
    def validate_energy_balance(self, generation: torch.Tensor,
                               consumption: torch.Tensor,
                               storage_charge: torch.Tensor,
                               storage_discharge: torch.Tensor,
                               grid_import: torch.Tensor,
                               grid_export: torch.Tensor) -> Dict[str, Any]:
        """
        Validate energy balance equation.
        Generation + Import + Discharge = Consumption + Export + Charge
        
        Returns:
            Validation results with violation details
        """
        # Calculate balance
        supply = generation + grid_import + storage_discharge
        demand = consumption + grid_export + storage_charge
        
        # Check balance
        imbalance = torch.abs(supply - demand)
        violations = imbalance > self.tolerance
        
        return {
            'valid': not violations.any(),
            'violation_ratio': violations.float().mean().item(),
            'max_imbalance': imbalance.max().item(),
            'mean_imbalance': imbalance.mean().item(),
            'violation_indices': torch.where(violations)[0].tolist()
        }
    
    def validate_power_flow(self, power_flows: torch.Tensor,
                          line_capacities: torch.Tensor,
                          voltage_limits: Tuple[float, float] = (0.95, 1.05)) -> Dict[str, Any]:
        """
        Validate power flow constraints.
        
        Args:
            power_flows: Power flow through lines [E]
            line_capacities: Maximum capacity of lines [E]
            voltage_limits: Min and max voltage p.u.
            
        Returns:
            Validation results
        """
        # Check line capacity constraints
        capacity_violations = torch.abs(power_flows) > line_capacities
        
        # Approximate voltage drop (simplified)
        voltage_drop = torch.abs(power_flows) / (line_capacities + 1e-6) * 0.1
        voltage = 1.0 - voltage_drop
        voltage_violations = (voltage < voltage_limits[0]) | (voltage > voltage_limits[1])
        
        return {
            'valid': not (capacity_violations.any() or voltage_violations.any()),
            'capacity_violations': capacity_violations.sum().item(),
            'voltage_violations': voltage_violations.sum().item(),
            'max_loading': (torch.abs(power_flows) / line_capacities).max().item(),
            'mean_loading': (torch.abs(power_flows) / line_capacities).mean().item()
        }
    
    def validate_storage_constraints(self, soc: torch.Tensor,
                                   charge_rate: torch.Tensor,
                                   discharge_rate: torch.Tensor,
                                   capacity: torch.Tensor,
                                   max_charge_rate: torch.Tensor,
                                   max_discharge_rate: torch.Tensor) -> Dict[str, Any]:
        """
        Validate battery storage constraints.
        
        Returns:
            Validation results for storage operations
        """
        violations = {}
        
        # SOC limits (0-100%)
        soc_violations = (soc < 0) | (soc > capacity)
        violations['soc_out_of_bounds'] = soc_violations.sum().item()
        
        # Charge/discharge rate limits
        charge_violations = charge_rate > max_charge_rate
        discharge_violations = discharge_rate > max_discharge_rate
        violations['charge_rate_exceeded'] = charge_violations.sum().item()
        violations['discharge_rate_exceeded'] = discharge_violations.sum().item()
        
        # Simultaneous charge/discharge check
        simultaneous = (charge_rate > 0) & (discharge_rate > 0)
        violations['simultaneous_charge_discharge'] = simultaneous.sum().item()
        
        # Energy conservation in storage
        energy_change = charge_rate - discharge_rate
        soc_change = torch.diff(soc, dim=-1) if soc.dim() > 1 else torch.tensor([0.0])
        conservation_error = torch.abs(soc_change - energy_change[:-1] if energy_change.dim() > 1 else torch.tensor([0.0]))
        violations['energy_conservation_error'] = conservation_error.mean().item()
        
        return {
            'valid': all(v == 0 for v in violations.values() if isinstance(v, int)),
            'violations': violations,
            'total_violations': sum(v for v in violations.values() if isinstance(v, int))
        }
    
    def validate_transformer_loading(self, transformer_loads: torch.Tensor,
                                   transformer_capacities: torch.Tensor,
                                   max_loading: float = 1.2) -> Dict[str, Any]:
        """
        Validate transformer loading constraints.
        
        Returns:
            Validation results for transformer operations
        """
        # Calculate loading percentage
        loading = transformer_loads / (transformer_capacities + 1e-6)
        
        # Check overloading
        overloaded = loading > max_loading
        critical = loading > 1.0
        
        return {
            'valid': not overloaded.any(),
            'overloaded_transformers': overloaded.sum().item(),
            'critical_transformers': critical.sum().item(),
            'max_loading': loading.max().item(),
            'mean_loading': loading.mean().item(),
            'loading_distribution': {
                'below_50': (loading < 0.5).sum().item(),
                '50_to_80': ((loading >= 0.5) & (loading < 0.8)).sum().item(),
                '80_to_100': ((loading >= 0.8) & (loading < 1.0)).sum().item(),
                'above_100': (loading >= 1.0).sum().item()
            }
        }
    
    def comprehensive_validation(self, predictions: Dict[str, torch.Tensor],
                               constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive validation of all predictions.
        
        Args:
            predictions: Model predictions
            constraints: System constraints
            
        Returns:
            Complete validation report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_valid': True,
            'checks': {}
        }
        
        # Energy balance check
        if all(k in predictions for k in ['generation', 'consumption']):
            balance_check = self.validate_energy_balance(
                generation=predictions.get('generation', torch.zeros(1)),
                consumption=predictions.get('consumption', torch.zeros(1)),
                storage_charge=predictions.get('storage_charge', torch.zeros(1)),
                storage_discharge=predictions.get('storage_discharge', torch.zeros(1)),
                grid_import=predictions.get('grid_import', torch.zeros(1)),
                grid_export=predictions.get('grid_export', torch.zeros(1))
            )
            report['checks']['energy_balance'] = balance_check
            report['overall_valid'] &= balance_check['valid']
        
        # Power flow check
        if 'power_flows' in predictions:
            flow_check = self.validate_power_flow(
                power_flows=predictions['power_flows'],
                line_capacities=constraints.get('line_capacities', torch.ones_like(predictions['power_flows']) * 100)
            )
            report['checks']['power_flow'] = flow_check
            report['overall_valid'] &= flow_check['valid']
        
        # Storage check
        if 'battery_soc' in predictions:
            storage_check = self.validate_storage_constraints(
                soc=predictions['battery_soc'],
                charge_rate=predictions.get('battery_charge', torch.zeros_like(predictions['battery_soc'])),
                discharge_rate=predictions.get('battery_discharge', torch.zeros_like(predictions['battery_soc'])),
                capacity=constraints.get('battery_capacity', torch.ones_like(predictions['battery_soc']) * 100),
                max_charge_rate=constraints.get('max_charge_rate', torch.ones_like(predictions['battery_soc']) * 10),
                max_discharge_rate=constraints.get('max_discharge_rate', torch.ones_like(predictions['battery_soc']) * 10)
            )
            report['checks']['storage'] = storage_check
            report['overall_valid'] &= storage_check['valid']
        
        # Transformer loading check
        if 'transformer_loads' in predictions:
            transformer_check = self.validate_transformer_loading(
                transformer_loads=predictions['transformer_loads'],
                transformer_capacities=constraints.get('transformer_capacities', torch.ones_like(predictions['transformer_loads']) * 630)
            )
            report['checks']['transformers'] = transformer_check
            report['overall_valid'] &= transformer_check['valid']
        
        # Store in history
        self.violation_history.append(report)
        
        return report


class StructuredReportGenerator:
    """
    Generates comprehensive structured reports from GNN outputs.
    """
    
    def __init__(self, output_dir: str = 'reports'):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.report_template = {
            'metadata': {},
            'predictions': {},
            'confidence': {},
            'explanations': {},
            'validation': {},
            'recommendations': {},
            'visualizations': []
        }
        
    def generate_comprehensive_report(self, model_outputs: Dict[str, Any],
                                     validation_results: Dict[str, Any],
                                     experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report from model outputs.
        
        Args:
            model_outputs: All model predictions and intermediate outputs
            validation_results: Physics validation results
            experiment_config: Experiment configuration
            
        Returns:
            Structured report dictionary
        """
        report = self.report_template.copy()
        
        # Metadata
        report['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'experiment_id': experiment_config.get('experiment_id', 'unknown'),
            'model_version': experiment_config.get('model_version', 'v1.0'),
            'data_source': experiment_config.get('data_source', 'unknown')
        }
        
        # Process predictions
        report['predictions'] = self._process_predictions(model_outputs)
        
        # Add confidence scores
        report['confidence'] = self._extract_confidence(model_outputs)
        
        # Add explanations
        report['explanations'] = self._generate_explanations(model_outputs)
        
        # Add validation
        report['validation'] = validation_results
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(
            model_outputs, validation_results
        )
        
        # Create visualizations
        report['visualizations'] = self._create_visualizations(model_outputs)
        
        # Calculate summary statistics
        report['summary'] = self._calculate_summary(report)
        
        return report
    
    def _process_predictions(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure predictions."""
        predictions = {}
        
        # Clustering predictions
        if 'clustering_cluster_assignments' in outputs:
            clusters = outputs['clustering_cluster_assignments']
            predictions['clusters'] = {
                'assignments': clusters.tolist() if torch.is_tensor(clusters) else clusters,
                'num_clusters': len(torch.unique(clusters)) if torch.is_tensor(clusters) else len(set(clusters)),
                'cluster_sizes': self._get_cluster_sizes(clusters)
            }
        
        # Energy predictions
        if 'energy_predictions' in outputs:
            energy = outputs['energy_predictions']
            predictions['energy'] = {
                'peak_demand': float(energy.max()) if torch.is_tensor(energy) else max(energy),
                'average_demand': float(energy.mean()) if torch.is_tensor(energy) else np.mean(energy),
                'total_consumption': float(energy.sum()) if torch.is_tensor(energy) else sum(energy)
            }
        
        # Intervention predictions
        if 'intervention_values' in outputs:
            interventions = outputs['intervention_values']
            predictions['interventions'] = {
                'recommended_nodes': torch.topk(interventions, k=10).indices.tolist() if torch.is_tensor(interventions) else sorted(range(len(interventions)), key=lambda i: interventions[i], reverse=True)[:10],
                'expected_impact': float(interventions.max()) if torch.is_tensor(interventions) else max(interventions)
            }
        
        return predictions
    
    def _extract_confidence(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract confidence scores from outputs."""
        confidence = {}
        
        if 'confidence_scores' in outputs:
            conf = outputs['confidence_scores']
            confidence['overall'] = float(conf.mean()) if torch.is_tensor(conf) else np.mean(conf)
            confidence['min'] = float(conf.min()) if torch.is_tensor(conf) else min(conf)
            confidence['max'] = float(conf.max()) if torch.is_tensor(conf) else max(conf)
            confidence['std'] = float(conf.std()) if torch.is_tensor(conf) else np.std(conf)
        
        if 'uncertainty' in outputs:
            uncertainty = outputs['uncertainty']
            confidence['uncertainty'] = {
                'epistemic': float(uncertainty.get('epistemic', 0)),
                'aleatoric': float(uncertainty.get('aleatoric', 0)),
                'total': float(uncertainty.get('total', 0))
            }
        
        return confidence
    
    def _generate_explanations(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanations from model outputs."""
        explanations = {}
        
        # Feature importance
        if 'feature_importance' in outputs:
            importance = outputs['feature_importance']
            explanations['feature_importance'] = {
                'top_features': self._get_top_features(importance),
                'feature_scores': importance.tolist() if torch.is_tensor(importance) else importance
            }
        
        # Attention weights
        if 'attention_weights' in outputs:
            attention = outputs['attention_weights']
            explanations['attention'] = {
                'mean_attention': float(attention.mean()) if torch.is_tensor(attention) else np.mean(attention),
                'attention_distribution': self._get_attention_distribution(attention)
            }
        
        # Decision path
        if 'decision_path' in outputs:
            explanations['decision_path'] = outputs['decision_path']
        
        return explanations
    
    def _generate_recommendations(self, outputs: Dict[str, Any],
                                 validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations."""
        recommendations = {
            'immediate_actions': [],
            'medium_term': [],
            'long_term': [],
            'warnings': []
        }
        
        # Check validation results
        if not validation.get('overall_valid', True):
            recommendations['warnings'].append({
                'type': 'physics_violation',
                'message': 'Model predictions violate physical constraints',
                'severity': 'high'
            })
        
        # Energy balance recommendations
        if 'energy_balance' in validation.get('checks', {}):
            balance = validation['checks']['energy_balance']
            if balance.get('mean_imbalance', 0) > 0.1:
                recommendations['immediate_actions'].append({
                    'action': 'balance_correction',
                    'description': 'Adjust generation or storage to balance supply and demand',
                    'priority': 'high',
                    'expected_impact': f"Reduce imbalance by {balance.get('mean_imbalance', 0):.2f} kW"
                })
        
        # Transformer loading recommendations
        if 'transformers' in validation.get('checks', {}):
            transformers = validation['checks']['transformers']
            if transformers.get('overloaded_transformers', 0) > 0:
                recommendations['immediate_actions'].append({
                    'action': 'load_shedding',
                    'description': f"Reduce load on {transformers.get('overloaded_transformers', 0)} overloaded transformers",
                    'priority': 'critical',
                    'affected_transformers': transformers.get('overloaded_transformers', 0)
                })
        
        # Clustering recommendations
        if 'clusters' in outputs:
            clusters = outputs['clusters']
            if isinstance(clusters, dict) and clusters.get('num_clusters', 0) > 20:
                recommendations['medium_term'].append({
                    'action': 'cluster_optimization',
                    'description': 'Consider consolidating energy communities for better management',
                    'current_clusters': clusters['num_clusters'],
                    'recommended_clusters': 10-15
                })
        
        # Solar recommendations
        if 'solar_suitability' in outputs:
            solar = outputs['solar_suitability']
            if torch.is_tensor(solar):
                top_candidates = torch.topk(solar, k=5).indices.tolist()
            else:
                top_candidates = sorted(range(len(solar)), key=lambda i: solar[i], reverse=True)[:5]
            
            recommendations['long_term'].append({
                'action': 'solar_deployment',
                'description': 'Deploy solar panels on high-suitability buildings',
                'candidate_buildings': top_candidates,
                'expected_generation': 'TBD based on roof area'
            })
        
        return recommendations
    
    def _create_visualizations(self, outputs: Dict[str, Any]) -> List[str]:
        """Create and save visualizations."""
        viz_paths = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Cluster visualization
        if 'clustering_cluster_assignments' in outputs:
            path = self._visualize_clusters(
                outputs['clustering_cluster_assignments'],
                self.output_dir / f'clusters_{timestamp}.png'
            )
            viz_paths.append(str(path))
        
        # Energy profile visualization
        if 'energy_profiles' in outputs:
            path = self._visualize_energy_profiles(
                outputs['energy_profiles'],
                self.output_dir / f'energy_{timestamp}.png'
            )
            viz_paths.append(str(path))
        
        # Network impact visualization
        if 'network_impacts' in outputs:
            path = self._visualize_network_impacts(
                outputs['network_impacts'],
                self.output_dir / f'network_{timestamp}.png'
            )
            viz_paths.append(str(path))
        
        return viz_paths
    
    def _calculate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        summary = {
            'total_predictions': 0,
            'valid_predictions': 0,
            'confidence_level': 'unknown',
            'key_findings': [],
            'risk_level': 'low'
        }
        
        # Count predictions
        if 'predictions' in report:
            for category in report['predictions'].values():
                if isinstance(category, dict):
                    summary['total_predictions'] += len(category)
        
        # Validation summary
        if 'validation' in report:
            if report['validation'].get('overall_valid'):
                summary['valid_predictions'] = summary['total_predictions']
            else:
                # Count valid checks
                checks = report['validation'].get('checks', {})
                valid_checks = sum(1 for check in checks.values() if check.get('valid'))
                summary['valid_predictions'] = int(summary['total_predictions'] * valid_checks / max(len(checks), 1))
        
        # Confidence summary
        if 'confidence' in report:
            overall_conf = report['confidence'].get('overall', 0)
            if overall_conf > 0.8:
                summary['confidence_level'] = 'high'
            elif overall_conf > 0.6:
                summary['confidence_level'] = 'medium'
            else:
                summary['confidence_level'] = 'low'
        
        # Risk assessment
        warnings = report.get('recommendations', {}).get('warnings', [])
        if any(w.get('severity') == 'critical' for w in warnings):
            summary['risk_level'] = 'critical'
        elif any(w.get('severity') == 'high' for w in warnings):
            summary['risk_level'] = 'high'
        elif warnings:
            summary['risk_level'] = 'medium'
        
        # Key findings
        if 'predictions' in report and 'clusters' in report['predictions']:
            summary['key_findings'].append(
                f"Identified {report['predictions']['clusters']['num_clusters']} energy communities"
            )
        
        if 'validation' in report and 'checks' in report['validation']:
            if 'transformers' in report['validation']['checks']:
                trans = report['validation']['checks']['transformers']
                if trans['overloaded_transformers'] > 0:
                    summary['key_findings'].append(
                        f"{trans['overloaded_transformers']} transformers are overloaded"
                    )
        
        return summary
    
    def _get_cluster_sizes(self, clusters: Union[torch.Tensor, List]) -> Dict[int, int]:
        """Get size of each cluster."""
        if torch.is_tensor(clusters):
            unique, counts = torch.unique(clusters, return_counts=True)
            return {int(u): int(c) for u, c in zip(unique, counts)}
        else:
            from collections import Counter
            return dict(Counter(clusters))
    
    def _get_top_features(self, importance: Union[torch.Tensor, np.ndarray],
                         k: int = 5) -> List[Tuple[int, float]]:
        """Get top-k important features."""
        if torch.is_tensor(importance):
            values, indices = torch.topk(importance, k=min(k, len(importance)))
            return [(int(i), float(v)) for i, v in zip(indices, values)]
        else:
            importance = np.array(importance)
            indices = np.argsort(importance)[-k:][::-1]
            return [(int(i), float(importance[i])) for i in indices]
    
    def _get_attention_distribution(self, attention: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """Get attention weight distribution statistics."""
        if torch.is_tensor(attention):
            attention = attention.detach().cpu().numpy()
        
        attention = np.array(attention).flatten()
        
        return {
            'mean': float(np.mean(attention)),
            'std': float(np.std(attention)),
            'min': float(np.min(attention)),
            'max': float(np.max(attention)),
            'median': float(np.median(attention))
        }
    
    def _visualize_clusters(self, clusters: Union[torch.Tensor, np.ndarray],
                          save_path: Path) -> Path:
        """Visualize cluster assignments."""
        if torch.is_tensor(clusters):
            clusters = clusters.detach().cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        unique_clusters = np.unique(clusters)
        cluster_sizes = [np.sum(clusters == c) for c in unique_clusters]
        
        plt.bar(unique_clusters, cluster_sizes)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Buildings')
        plt.title('Cluster Size Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _visualize_energy_profiles(self, profiles: Union[torch.Tensor, np.ndarray],
                                  save_path: Path) -> Path:
        """Visualize energy profiles."""
        if torch.is_tensor(profiles):
            profiles = profiles.detach().cpu().numpy()
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Average profile
        if profiles.ndim > 1:
            avg_profile = np.mean(profiles, axis=0)
            std_profile = np.std(profiles, axis=0)
        else:
            avg_profile = profiles
            std_profile = np.zeros_like(profiles)
        
        x = np.arange(len(avg_profile))
        
        axes[0].plot(x, avg_profile, 'b-', label='Average')
        axes[0].fill_between(x, avg_profile - std_profile, avg_profile + std_profile,
                            alpha=0.3, label='±1 STD')
        axes[0].set_xlabel('Time (15-min intervals)')
        axes[0].set_ylabel('Power (kW)')
        axes[0].set_title('Average Energy Profile')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Heatmap of all profiles
        if profiles.ndim > 1:
            im = axes[1].imshow(profiles[:min(50, len(profiles))], aspect='auto', cmap='YlOrRd')
            axes[1].set_xlabel('Time (15-min intervals)')
            axes[1].set_ylabel('Building Index')
            axes[1].set_title('Individual Energy Profiles (First 50)')
            plt.colorbar(im, ax=axes[1], label='Power (kW)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _visualize_network_impacts(self, impacts: Union[torch.Tensor, np.ndarray],
                                  save_path: Path) -> Path:
        """Visualize network impact distribution."""
        if torch.is_tensor(impacts):
            impacts = impacts.detach().cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        
        if impacts.ndim > 1:
            # Assume it's a hop-wise impact matrix
            hop_impacts = [impacts[:, i].mean() for i in range(impacts.shape[1])]
            hops = list(range(1, len(hop_impacts) + 1))
            
            plt.bar(hops, hop_impacts)
            plt.xlabel('Hop Distance')
            plt.ylabel('Average Impact')
            plt.title('Network Impact by Hop Distance')
        else:
            # Single impact vector
            plt.hist(impacts, bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel('Network Impact Score')
            plt.ylabel('Frequency')
            plt.title('Network Impact Distribution')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def save_report(self, report: Dict[str, Any], format: str = 'json') -> Path:
        """
        Save report to file.
        
        Args:
            report: Report dictionary
            format: Output format ('json', 'html', 'pdf')
            
        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            path = self.output_dir / f'report_{timestamp}.json'
            with open(path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif format == 'html':
            path = self.output_dir / f'report_{timestamp}.html'
            html_content = self._generate_html_report(report)
            with open(path, 'w') as f:
                f.write(html_content)
        
        elif format == 'pdf':
            # Requires additional libraries like reportlab
            raise NotImplementedError("PDF export not yet implemented")
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Report saved to {path}")
        return path
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Energy GNN Report - {report['metadata']['timestamp']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; }}
                .summary {{ background: #ecf0f1; padding: 15px; border-radius: 5px; }}
                .valid {{ color: green; }}
                .invalid {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #34495e; color: white; }}
                .warning {{ background-color: #f39c12; color: white; padding: 10px; border-radius: 5px; }}
                .recommendation {{ background-color: #3498db; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Energy GNN Analysis Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Experiment ID:</strong> {report['metadata'].get('experiment_id', 'N/A')}</p>
                <p><strong>Timestamp:</strong> {report['metadata']['timestamp']}</p>
                <p><strong>Overall Valid:</strong> <span class="{'valid' if report.get('validation', {}).get('overall_valid') else 'invalid'}">
                    {report.get('validation', {}).get('overall_valid', 'Unknown')}</span></p>
                <p><strong>Confidence Level:</strong> {report.get('summary', {}).get('confidence_level', 'Unknown')}</p>
                <p><strong>Risk Level:</strong> {report.get('summary', {}).get('risk_level', 'Unknown')}</p>
            </div>
            
            <h2>Key Findings</h2>
            <ul>
                {"".join(f"<li>{finding}</li>" for finding in report.get('summary', {}).get('key_findings', []))}
            </ul>
            
            <h2>Recommendations</h2>
            {"".join(f'<div class="recommendation"><strong>{rec["action"]}:</strong> {rec["description"]}</div>' 
                     for rec in report.get('recommendations', {}).get('immediate_actions', []))}
            
            <h2>Warnings</h2>
            {"".join(f'<div class="warning"><strong>{w["type"]}:</strong> {w["message"]}</div>' 
                     for w in report.get('recommendations', {}).get('warnings', []))}
        </body>
        </html>
        """
        return html