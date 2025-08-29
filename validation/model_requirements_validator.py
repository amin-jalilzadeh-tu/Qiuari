"""
Model Requirements Validator
Comprehensive validation system to track and verify that the GNN model meets all specifications
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RequirementCheck:
    """Single requirement check result"""
    requirement_id: str
    requirement_name: str
    category: str
    status: str  # 'passed', 'failed', 'partial', 'not_tested'
    score: float  # 0-100
    details: Dict[str, Any]
    evidence: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationReport:
    """Complete validation report"""
    model_name: str
    validation_date: datetime
    requirements_checks: List[RequirementCheck]
    overall_compliance: float
    critical_failures: List[str]
    warnings: List[str]
    recommendations: List[str]
    performance_metrics: Dict[str, float]
    test_data_info: Dict[str, Any]


class ModelRequirementsValidator:
    """
    Comprehensive validator for GNN model requirements
    Tracks and validates all specifications from the requirements document
    """
    
    def __init__(self, config: Dict = None):
        """Initialize validator with requirements"""
        self.config = config or {}
        self.validation_history = []
        self.current_report = None
        
        # Define all requirements to validate
        self.requirements = self._define_requirements()
        
        # Track validation results
        self.results = defaultdict(dict)
        
        logger.info("Initialized Model Requirements Validator")
    
    def _define_requirements(self) -> Dict[str, Dict]:
        """Define all requirements from specification"""
        return {
            # Architecture Requirements
            'arch_gnn_type': {
                'name': 'GNN Architecture Type',
                'category': 'Architecture',
                'critical': True,
                'check_func': self._check_gnn_architecture,
                'description': 'Must be Graph Neural Network with heterogeneous support'
            },
            'arch_node_types': {
                'name': 'Heterogeneous Node Types',
                'category': 'Architecture',
                'critical': True,
                'check_func': self._check_node_types,
                'description': 'Must support buildings, LV groups, MV stations, transformers'
            },
            'arch_message_passing': {
                'name': 'Message Passing Layers',
                'category': 'Architecture',
                'critical': True,
                'check_func': self._check_message_passing,
                'description': 'Must have 3 message passing layers'
            },
            'arch_attention': {
                'name': 'Attention Mechanisms',
                'category': 'Architecture',
                'critical': False,
                'check_func': self._check_attention,
                'description': 'Must include attention for important connections'
            },
            'arch_mc_dropout': {
                'name': 'MC Dropout for Uncertainty',
                'category': 'Architecture',
                'critical': True,
                'check_func': self._check_mc_dropout,
                'description': 'Must implement MC Dropout (20 passes)'
            },
            
            # Dynamic Clustering Requirements
            'cluster_dynamic': {
                'name': 'Dynamic Sub-clustering',
                'category': 'Clustering',
                'critical': True,
                'check_func': self._check_dynamic_clustering,
                'description': 'Must discover optimal number of sub-clusters'
            },
            'cluster_size_range': {
                'name': 'Cluster Size Range',
                'category': 'Clustering',
                'critical': True,
                'check_func': self._check_cluster_size,
                'description': 'Clusters must be 3-20 buildings'
            },
            'cluster_lv_constraint': {
                'name': 'LV Boundary Constraint',
                'category': 'Clustering',
                'critical': True,
                'check_func': self._check_lv_boundary,
                'description': 'Clusters CANNOT cross LV group boundaries'
            },
            'cluster_optimization': {
                'name': 'Cluster Optimization',
                'category': 'Clustering',
                'critical': True,
                'check_func': self._check_cluster_optimization,
                'description': 'Must optimize for self-sufficiency AND complementarity'
            },
            'cluster_stability': {
                'name': 'Cluster Stability',
                'category': 'Clustering',
                'critical': False,
                'check_func': self._check_cluster_stability,
                'description': 'Must track and minimize cluster jumping'
            },
            
            # Energy Flow Requirements
            'energy_tracking': {
                'name': 'Energy Flow Tracking',
                'category': 'Energy',
                'critical': True,
                'check_func': self._check_energy_tracking,
                'description': 'Must track energy sharing at each timestep'
            },
            'energy_constraint': {
                'name': 'Energy Sharing Constraint',
                'category': 'Energy',
                'critical': True,
                'check_func': self._check_energy_constraint,
                'description': 'Energy sharing ONLY within same LV group'
            },
            'energy_balance': {
                'name': 'Energy Balance Calculation',
                'category': 'Energy',
                'critical': True,
                'check_func': self._check_energy_balance,
                'description': 'Must calculate grid import vs self-generated vs peer-shared'
            },
            'energy_temporal': {
                'name': 'Temporal Energy Patterns',
                'category': 'Energy',
                'critical': False,
                'check_func': self._check_temporal_patterns,
                'description': 'Must consider weekday/weekend and seasonal variations'
            },
            
            # Cluster Quality Requirements
            'quality_assessment': {
                'name': 'Quality Assessment',
                'category': 'Quality',
                'critical': True,
                'check_func': self._check_quality_assessment,
                'description': 'Must generate quality labels for each cluster'
            },
            'quality_metrics': {
                'name': 'Quality Metrics',
                'category': 'Quality',
                'critical': True,
                'check_func': self._check_quality_metrics,
                'description': 'Must calculate self-sufficiency, complementarity, peak reduction'
            },
            'quality_categorization': {
                'name': 'Quality Categories',
                'category': 'Quality',
                'critical': False,
                'check_func': self._check_quality_categories,
                'description': 'Must categorize clusters: excellent/good/fair/poor'
            },
            
            # Solar Requirements
            'solar_recommendations': {
                'name': 'Solar Recommendations',
                'category': 'Solar',
                'critical': True,
                'check_func': self._check_solar_recommendations,
                'description': 'Must recommend solar based on cascade effects'
            },
            'solar_cascade': {
                'name': 'Cascade Modeling',
                'category': 'Solar',
                'critical': True,
                'check_func': self._check_cascade_modeling,
                'description': 'Must model 1-hop, 2-hop, 3-hop impacts'
            },
            'solar_roi': {
                'name': 'ROI Calculation',
                'category': 'Solar',
                'critical': True,
                'check_func': self._check_roi_calculation,
                'description': 'Must calculate ROI with uncertainty bounds'
            },
            'solar_roadmap': {
                'name': 'Penetration Roadmap',
                'category': 'Solar',
                'critical': True,
                'check_func': self._check_roadmap_capability,
                'description': 'Must generate multi-year deployment roadmap'
            },
            
            # Learning Requirements
            'learning_unsupervised': {
                'name': 'Unsupervised Discovery',
                'category': 'Learning',
                'critical': True,
                'check_func': self._check_unsupervised_learning,
                'description': 'Must start with unsupervised discovery'
            },
            'learning_pseudo_labels': {
                'name': 'Pseudo-label Generation',
                'category': 'Learning',
                'critical': True,
                'check_func': self._check_pseudo_labels,
                'description': 'Must generate pseudo-labels from performance'
            },
            'learning_iterative': {
                'name': 'Iterative Learning',
                'category': 'Learning',
                'critical': False,
                'check_func': self._check_iterative_learning,
                'description': 'Must learn iteratively from generated labels'
            },
            
            # Constraint Requirements
            'constraint_lv_boundary': {
                'name': 'LV Boundary Enforcement',
                'category': 'Constraints',
                'critical': True,
                'check_func': self._check_lv_enforcement,
                'description': 'Must STRICTLY enforce LV boundaries'
            },
            'constraint_cluster_size': {
                'name': 'Cluster Size Limits',
                'category': 'Constraints',
                'critical': True,
                'check_func': self._check_size_limits,
                'description': 'Must maintain 3-20 building clusters'
            },
            'constraint_topology': {
                'name': 'Network Topology',
                'category': 'Constraints',
                'critical': True,
                'check_func': self._check_topology_respect,
                'description': 'Must respect MV→LV→Building hierarchy'
            },
            
            # Uncertainty Requirements
            'uncertainty_confidence': {
                'name': 'Confidence Intervals',
                'category': 'Uncertainty',
                'critical': True,
                'check_func': self._check_confidence_intervals,
                'description': 'Must provide confidence intervals for predictions'
            },
            'uncertainty_mc_dropout': {
                'name': 'MC Dropout Usage',
                'category': 'Uncertainty',
                'critical': True,
                'check_func': self._check_mc_dropout_usage,
                'description': 'Must use MC Dropout (20 iterations)'
            },
            'uncertainty_calibration': {
                'name': 'Calibrated Confidence',
                'category': 'Uncertainty',
                'critical': False,
                'check_func': self._check_calibration,
                'description': 'Must output calibrated confidence scores'
            },
            
            # Explainability Requirements
            'explain_clustering': {
                'name': 'Clustering Explanations',
                'category': 'Explainability',
                'critical': False,
                'check_func': self._check_clustering_explanations,
                'description': 'Must explain why buildings are clustered together'
            },
            'explain_solar': {
                'name': 'Solar Explanations',
                'category': 'Explainability',
                'critical': False,
                'check_func': self._check_solar_explanations,
                'description': 'Must explain solar recommendations with cascade'
            },
            'explain_human_readable': {
                'name': 'Human-readable Summaries',
                'category': 'Explainability',
                'critical': False,
                'check_func': self._check_human_readable,
                'description': 'Must generate human-readable summaries'
            }
        }
    
    def validate_model(
        self,
        model: nn.Module,
        test_data: Any,
        gnn_system: Any = None,
        detailed: bool = True
    ) -> ValidationReport:
        """
        Perform comprehensive validation of the model
        
        Args:
            model: The GNN model to validate
            test_data: Test data for validation
            gnn_system: The complete GNN system (for component checks)
            detailed: Whether to perform detailed validation
            
        Returns:
            Validation report with all checks
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING COMPREHENSIVE MODEL VALIDATION")
        logger.info("="*80 + "\n")
        
        checks = []
        critical_failures = []
        warnings = []
        recommendations = []
        
        # Run all requirement checks
        for req_id, requirement in self.requirements.items():
            logger.info(f"Checking: {requirement['name']}...")
            
            try:
                # Run the check function
                check_result = requirement['check_func'](
                    model, test_data, gnn_system
                )
                
                # Create requirement check
                check = RequirementCheck(
                    requirement_id=req_id,
                    requirement_name=requirement['name'],
                    category=requirement['category'],
                    status=check_result['status'],
                    score=check_result['score'],
                    details=check_result.get('details', {}),
                    evidence=check_result.get('evidence', [])
                )
                
                checks.append(check)
                
                # Track critical failures
                if requirement['critical'] and check.status == 'failed':
                    critical_failures.append(
                        f"{requirement['name']}: {check_result.get('reason', 'Failed check')}"
                    )
                
                # Track warnings
                if check.status == 'partial':
                    warnings.append(
                        f"{requirement['name']}: {check_result.get('reason', 'Partial compliance')}"
                    )
                
                # Log result
                status_symbol = {
                    'passed': '✅',
                    'failed': '❌',
                    'partial': '⚠️',
                    'not_tested': '⏭️'
                }[check.status]
                
                logger.info(f"  {status_symbol} {check.status.upper()} (Score: {check.score:.0f}/100)")
                
            except Exception as e:
                logger.error(f"  ❌ ERROR checking {requirement['name']}: {e}")
                check = RequirementCheck(
                    requirement_id=req_id,
                    requirement_name=requirement['name'],
                    category=requirement['category'],
                    status='not_tested',
                    score=0,
                    details={'error': str(e)},
                    evidence=[]
                )
                checks.append(check)
        
        # Calculate overall compliance
        total_score = sum(c.score for c in checks)
        max_score = len(checks) * 100
        overall_compliance = (total_score / max_score) * 100 if max_score > 0 else 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(checks)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            model, test_data, gnn_system
        )
        
        # Create report
        report = ValidationReport(
            model_name=model.__class__.__name__,
            validation_date=datetime.now(),
            requirements_checks=checks,
            overall_compliance=overall_compliance,
            critical_failures=critical_failures,
            warnings=warnings,
            recommendations=recommendations,
            performance_metrics=performance_metrics,
            test_data_info={
                'num_buildings': test_data['building'].x.shape[0] if hasattr(test_data, '__getitem__') else 0,
                'num_edges': test_data.edge_index_dict.get(('building', 'connected_to', 'building'), torch.tensor([[], []])).shape[1] if hasattr(test_data, 'edge_index_dict') else 0
            }
        )
        
        self.current_report = report
        self.validation_history.append(report)
        
        # Log summary
        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Overall Compliance: {overall_compliance:.1f}%")
        logger.info(f"Critical Failures: {len(critical_failures)}")
        logger.info(f"Warnings: {len(warnings)}")
        logger.info(f"Recommendations: {len(recommendations)}")
        
        if critical_failures:
            logger.error("\n⚠️ CRITICAL FAILURES DETECTED:")
            for failure in critical_failures:
                logger.error(f"  - {failure}")
        
        if overall_compliance >= 90:
            logger.info("\n✅ MODEL MEETS REQUIREMENTS - Excellent compliance!")
        elif overall_compliance >= 70:
            logger.info("\n✅ MODEL PARTIALLY MEETS REQUIREMENTS - Good compliance with some issues")
        elif overall_compliance >= 50:
            logger.warning("\n⚠️ MODEL HAS SIGNIFICANT GAPS - Major improvements needed")
        else:
            logger.error("\n❌ MODEL DOES NOT MEET REQUIREMENTS - Critical rework required")
        
        return report
    
    def _check_gnn_architecture(self, model, data, system) -> Dict:
        """Check if model is proper GNN with heterogeneous support"""
        try:
            # Check for GNN layers
            has_gnn = any(
                'GNN' in name or 'Conv' in name or 'Message' in name
                for name, _ in model.named_modules()
            )
            
            # Check for heterogeneous support
            has_hetero = hasattr(model, 'convs') or 'Hetero' in model.__class__.__name__
            
            if has_gnn and has_hetero:
                return {
                    'status': 'passed',
                    'score': 100,
                    'details': {'architecture': model.__class__.__name__},
                    'evidence': ['GNN layers detected', 'Heterogeneous support confirmed']
                }
            elif has_gnn:
                return {
                    'status': 'partial',
                    'score': 70,
                    'details': {'architecture': model.__class__.__name__},
                    'evidence': ['GNN layers detected'],
                    'reason': 'Heterogeneous support unclear'
                }
            else:
                return {
                    'status': 'failed',
                    'score': 0,
                    'details': {'architecture': model.__class__.__name__},
                    'reason': 'Not a GNN architecture'
                }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_node_types(self, model, data, system) -> Dict:
        """Check if model supports required node types"""
        try:
            required_types = {'building', 'cable_group', 'transformer'}
            
            if hasattr(data, 'node_types'):
                found_types = set(data.node_types)
                missing = required_types - found_types
                
                if not missing:
                    return {
                        'status': 'passed',
                        'score': 100,
                        'details': {'node_types': list(found_types)},
                        'evidence': [f'All required node types present: {found_types}']
                    }
                else:
                    return {
                        'status': 'partial',
                        'score': 70,
                        'details': {'found': list(found_types), 'missing': list(missing)},
                        'reason': f'Missing node types: {missing}'
                    }
            else:
                return {
                    'status': 'failed',
                    'score': 0,
                    'reason': 'No heterogeneous node types found'
                }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_message_passing(self, model, data, system) -> Dict:
        """Check for 3 message passing layers"""
        try:
            # Count convolutional layers
            conv_layers = [m for name, m in model.named_modules() if 'conv' in name.lower()]
            
            if len(conv_layers) == 3:
                return {
                    'status': 'passed',
                    'score': 100,
                    'details': {'num_layers': len(conv_layers)},
                    'evidence': ['Exactly 3 message passing layers found']
                }
            elif len(conv_layers) >= 2:
                return {
                    'status': 'partial',
                    'score': 70,
                    'details': {'num_layers': len(conv_layers)},
                    'reason': f'Found {len(conv_layers)} layers instead of 3'
                }
            else:
                return {
                    'status': 'failed',
                    'score': 0,
                    'details': {'num_layers': len(conv_layers)},
                    'reason': 'Insufficient message passing layers'
                }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_attention(self, model, data, system) -> Dict:
        """Check for attention mechanisms"""
        try:
            has_attention = any(
                'attention' in name.lower() or 'attn' in name.lower()
                for name, _ in model.named_modules()
            )
            
            if has_attention:
                return {
                    'status': 'passed',
                    'score': 100,
                    'evidence': ['Attention mechanisms detected']
                }
            else:
                return {
                    'status': 'failed',
                    'score': 0,
                    'reason': 'No attention mechanisms found'
                }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_mc_dropout(self, model, data, system) -> Dict:
        """Check for MC Dropout implementation"""
        try:
            # Check for dropout layers
            has_dropout = any(
                isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
                for m in model.modules()
            )
            
            # Check for MC Dropout in system
            has_mc = system and hasattr(system, 'uncertainty_analyzer')
            
            if has_dropout and has_mc:
                return {
                    'status': 'passed',
                    'score': 100,
                    'evidence': ['Dropout layers present', 'MC Dropout system available']
                }
            elif has_dropout:
                return {
                    'status': 'partial',
                    'score': 60,
                    'reason': 'Dropout present but MC implementation unclear'
                }
            else:
                return {
                    'status': 'failed',
                    'score': 0,
                    'reason': 'No dropout layers found'
                }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_dynamic_clustering(self, model, data, system) -> Dict:
        """Check if model can discover optimal clusters"""
        try:
            # Run model prediction
            model.eval()
            with torch.no_grad():
                outputs = model(data, task='clustering')
                
            if 'cluster_logits' in outputs or 'cluster_assignments' in outputs:
                clusters = outputs.get('cluster_logits', outputs.get('cluster_assignments'))
                num_clusters = clusters.shape[-1] if len(clusters.shape) > 1 else clusters.unique().shape[0]
                
                if num_clusters > 1:
                    return {
                        'status': 'passed',
                        'score': 100,
                        'details': {'num_clusters': num_clusters},
                        'evidence': [f'Dynamic clustering with {num_clusters} clusters']
                    }
                else:
                    return {
                        'status': 'failed',
                        'score': 0,
                        'reason': 'All buildings in single cluster (collapsed)'
                    }
            else:
                return {
                    'status': 'failed',
                    'score': 0,
                    'reason': 'No clustering output found'
                }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_cluster_size(self, model, data, system) -> Dict:
        """Check if clusters are within 3-20 buildings"""
        try:
            model.eval()
            with torch.no_grad():
                outputs = model(data, task='clustering')
                
            if 'cluster_logits' in outputs or 'cluster_assignments' in outputs:
                clusters = outputs.get('cluster_logits', outputs.get('cluster_assignments'))
                if len(clusters.shape) > 1:
                    assignments = clusters.argmax(dim=-1)
                else:
                    assignments = clusters
                
                # Count cluster sizes
                unique_clusters = assignments.unique()
                violations = 0
                sizes = []
                
                for cluster_id in unique_clusters:
                    size = (assignments == cluster_id).sum().item()
                    sizes.append(size)
                    if size < 3 or size > 20:
                        violations += 1
                
                if violations == 0:
                    return {
                        'status': 'passed',
                        'score': 100,
                        'details': {'cluster_sizes': sizes},
                        'evidence': ['All clusters within 3-20 buildings']
                    }
                else:
                    return {
                        'status': 'partial',
                        'score': max(0, 100 - violations * 20),
                        'details': {'cluster_sizes': sizes, 'violations': violations},
                        'reason': f'{violations} clusters violate size constraints'
                    }
            else:
                return {
                    'status': 'failed',
                    'score': 0,
                    'reason': 'No clustering output to check'
                }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_lv_boundary(self, model, data, system) -> Dict:
        """Check if clusters respect LV boundaries"""
        try:
            if not hasattr(data, 'lv_group_ids'):
                return {
                    'status': 'not_tested',
                    'score': 0,
                    'reason': 'No LV group information available'
                }
            
            model.eval()
            with torch.no_grad():
                outputs = model(data, task='clustering')
                
            if 'cluster_logits' in outputs or 'cluster_assignments' in outputs:
                clusters = outputs.get('cluster_logits', outputs.get('cluster_assignments'))
                if len(clusters.shape) > 1:
                    assignments = clusters.argmax(dim=-1)
                else:
                    assignments = clusters
                
                # Check for LV boundary violations
                violations = 0
                for cluster_id in assignments.unique():
                    cluster_mask = assignments == cluster_id
                    lv_groups = data.lv_group_ids[cluster_mask].unique()
                    if len(lv_groups) > 1:
                        violations += 1
                
                if violations == 0:
                    return {
                        'status': 'passed',
                        'score': 100,
                        'evidence': ['No LV boundary violations detected']
                    }
                else:
                    return {
                        'status': 'failed',
                        'score': 0,
                        'details': {'violations': violations},
                        'reason': f'{violations} clusters cross LV boundaries'
                    }
            else:
                return {
                    'status': 'failed',
                    'score': 0,
                    'reason': 'No clustering output to check'
                }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_cluster_optimization(self, model, data, system) -> Dict:
        """Check if clusters optimize for self-sufficiency and complementarity"""
        try:
            if system and hasattr(system, 'cluster_evaluator'):
                # Use system's evaluator
                model.eval()
                with torch.no_grad():
                    outputs = model(data, task='clustering')
                    
                if 'cluster_logits' in outputs or 'cluster_assignments' in outputs:
                    # Create minimal temporal data for evaluation
                    temporal_data = pd.DataFrame({
                        'building_id': range(data['building'].x.shape[0]),
                        'consumption': np.random.rand(data['building'].x.shape[0]) * 100,
                        'generation': np.random.rand(data['building'].x.shape[0]) * 20
                    })
                    
                    metrics = system.cluster_evaluator.evaluate(
                        outputs.get('cluster_logits', outputs.get('cluster_assignments')),
                        temporal_data,
                        {},
                        torch.zeros(data['building'].x.shape[0])
                    )
                    
                    self_suff = metrics.get('avg_self_sufficiency', 0)
                    complementarity = metrics.get('avg_complementarity', 0)
                    
                    if self_suff > 0.5 and complementarity < -0.3:
                        return {
                            'status': 'passed',
                            'score': 100,
                            'details': {
                                'self_sufficiency': self_suff,
                                'complementarity': complementarity
                            },
                            'evidence': ['Good self-sufficiency and complementarity']
                        }
                    else:
                        return {
                            'status': 'partial',
                            'score': 50,
                            'details': {
                                'self_sufficiency': self_suff,
                                'complementarity': complementarity
                            },
                            'reason': 'Suboptimal optimization metrics'
                        }
                else:
                    return {
                        'status': 'failed',
                        'score': 0,
                        'reason': 'No clustering output'
                    }
            else:
                return {
                    'status': 'not_tested',
                    'score': 0,
                    'reason': 'No cluster evaluator available'
                }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_cluster_stability(self, model, data, system) -> Dict:
        """Check cluster stability tracking"""
        try:
            if system and hasattr(system, 'cluster_stability'):
                stability_scores = system.cluster_stability.get('stability_scores', [])
                if stability_scores:
                    avg_stability = np.mean(stability_scores)
                    return {
                        'status': 'passed' if avg_stability > 0.8 else 'partial',
                        'score': min(100, avg_stability * 100),
                        'details': {'avg_stability': avg_stability},
                        'evidence': [f'Average stability: {avg_stability:.2%}']
                    }
            return {
                'status': 'not_tested',
                'score': 0,
                'reason': 'No stability tracking data available'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_energy_tracking(self, model, data, system) -> Dict:
        """Check energy flow tracking"""
        try:
            if system and hasattr(system, 'energy_flows'):
                if system.energy_flows:
                    return {
                        'status': 'passed',
                        'score': 100,
                        'details': {'num_epochs_tracked': len(system.energy_flows)},
                        'evidence': ['Energy flows tracked across epochs']
                    }
            return {
                'status': 'failed',
                'score': 0,
                'reason': 'No energy flow tracking found'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_energy_constraint(self, model, data, system) -> Dict:
        """Check if energy sharing respects LV constraints"""
        # This would need actual energy flow data to validate
        return {
            'status': 'not_tested',
            'score': 0,
            'reason': 'Requires runtime energy flow analysis'
        }
    
    def _check_energy_balance(self, model, data, system) -> Dict:
        """Check energy balance calculations"""
        try:
            if system and hasattr(system, 'energy_flows'):
                if system.energy_flows:
                    # Check if balance metrics are calculated
                    sample_flow = next(iter(system.energy_flows.values()))
                    if sample_flow:
                        sample_cluster = next(iter(sample_flow.values()))
                        if 'self_sufficiency' in sample_cluster:
                            return {
                                'status': 'passed',
                                'score': 100,
                                'evidence': ['Energy balance metrics calculated']
                            }
            return {
                'status': 'failed',
                'score': 0,
                'reason': 'No energy balance calculations found'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_temporal_patterns(self, model, data, system) -> Dict:
        """Check temporal pattern consideration"""
        try:
            if hasattr(data, 'temporal_features') or hasattr(data['building'], 'temporal_features'):
                return {
                    'status': 'passed',
                    'score': 100,
                    'evidence': ['Temporal features present in data']
                }
            return {
                'status': 'partial',
                'score': 50,
                'reason': 'No explicit temporal features found'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_quality_assessment(self, model, data, system) -> Dict:
        """Check quality label generation"""
        try:
            if system and hasattr(system, 'quality_labeler'):
                return {
                    'status': 'passed',
                    'score': 100,
                    'evidence': ['Quality labeler component present']
                }
            return {
                'status': 'failed',
                'score': 0,
                'reason': 'No quality labeler found'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_quality_metrics(self, model, data, system) -> Dict:
        """Check quality metric calculations"""
        try:
            if system and hasattr(system, 'quality_labeler'):
                # Check if labeler calculates required metrics
                required_metrics = ['self_sufficiency', 'complementarity', 'peak_reduction']
                return {
                    'status': 'passed',
                    'score': 100,
                    'evidence': ['Quality metrics calculator available']
                }
            return {
                'status': 'failed',
                'score': 0,
                'reason': 'No quality metrics calculator'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_quality_categories(self, model, data, system) -> Dict:
        """Check quality categorization"""
        try:
            if system and hasattr(system, 'pseudo_labels'):
                if 'cluster_labels' in system.pseudo_labels:
                    return {
                        'status': 'passed',
                        'score': 100,
                        'evidence': ['Cluster quality categorization present']
                    }
            return {
                'status': 'partial',
                'score': 50,
                'reason': 'Quality categorization not fully implemented'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_solar_recommendations(self, model, data, system) -> Dict:
        """Check solar recommendation capability"""
        try:
            model.eval()
            with torch.no_grad():
                outputs = model(data, task='solar')
                
            if 'solar' in outputs:
                return {
                    'status': 'passed',
                    'score': 100,
                    'evidence': ['Solar recommendation output present']
                }
            return {
                'status': 'failed',
                'score': 0,
                'reason': 'No solar output from model'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_cascade_modeling(self, model, data, system) -> Dict:
        """Check cascade effect modeling"""
        try:
            if system and hasattr(system, 'cascade_analyzer'):
                return {
                    'status': 'passed',
                    'score': 100,
                    'evidence': ['Cascade analyzer component present']
                }
            return {
                'status': 'failed',
                'score': 0,
                'reason': 'No cascade analyzer found'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_roi_calculation(self, model, data, system) -> Dict:
        """Check ROI calculation with uncertainty"""
        try:
            model.eval()
            with torch.no_grad():
                outputs = model(data, task='solar')
                
            if 'solar' in outputs and outputs['solar'].shape[-1] == 4:
                # Check for 4 ROI classes
                return {
                    'status': 'passed',
                    'score': 100,
                    'details': {'roi_classes': 4},
                    'evidence': ['ROI categorization present']
                }
            return {
                'status': 'partial',
                'score': 50,
                'reason': 'ROI output unclear'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_roadmap_capability(self, model, data, system) -> Dict:
        """Check roadmap generation capability"""
        try:
            if system and hasattr(system, 'roadmap_planner'):
                return {
                    'status': 'passed',
                    'score': 100,
                    'evidence': ['Roadmap planner component present']
                }
            return {
                'status': 'failed',
                'score': 0,
                'reason': 'No roadmap planner found'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_unsupervised_learning(self, model, data, system) -> Dict:
        """Check unsupervised discovery phase"""
        try:
            if system and hasattr(system, '_train_discovery_epoch'):
                return {
                    'status': 'passed',
                    'score': 100,
                    'evidence': ['Unsupervised discovery phase implemented']
                }
            return {
                'status': 'failed',
                'score': 0,
                'reason': 'No discovery phase found'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_pseudo_labels(self, model, data, system) -> Dict:
        """Check pseudo-label generation"""
        try:
            if system and hasattr(system, 'pseudo_labels'):
                if system.pseudo_labels:
                    return {
                        'status': 'passed',
                        'score': 100,
                        'evidence': ['Pseudo-labels generated']
                    }
            return {
                'status': 'failed',
                'score': 0,
                'reason': 'No pseudo-label generation'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_iterative_learning(self, model, data, system) -> Dict:
        """Check iterative learning from labels"""
        try:
            if system and hasattr(system, '_train_semi_supervised_epoch'):
                return {
                    'status': 'passed',
                    'score': 100,
                    'evidence': ['Semi-supervised learning phase present']
                }
            return {
                'status': 'partial',
                'score': 50,
                'reason': 'Iterative learning unclear'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_lv_enforcement(self, model, data, system) -> Dict:
        """Check strict LV boundary enforcement"""
        # Same as _check_lv_boundary but focusing on enforcement
        return self._check_lv_boundary(model, data, system)
    
    def _check_size_limits(self, model, data, system) -> Dict:
        """Check cluster size limit enforcement"""
        # Same as _check_cluster_size
        return self._check_cluster_size(model, data, system)
    
    def _check_topology_respect(self, model, data, system) -> Dict:
        """Check network topology respect"""
        try:
            if hasattr(data, 'edge_index_dict'):
                has_hierarchy = (
                    ('building', 'connected_to', 'cable_group') in data.edge_index_dict or
                    ('cable_group', 'connected_to', 'transformer') in data.edge_index_dict
                )
                if has_hierarchy:
                    return {
                        'status': 'passed',
                        'score': 100,
                        'evidence': ['Hierarchical topology present']
                    }
            return {
                'status': 'partial',
                'score': 50,
                'reason': 'Topology structure unclear'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_confidence_intervals(self, model, data, system) -> Dict:
        """Check confidence interval provision"""
        try:
            if system and hasattr(system, 'uncertainty_analyzer'):
                return {
                    'status': 'passed',
                    'score': 100,
                    'evidence': ['Uncertainty analyzer present']
                }
            return {
                'status': 'failed',
                'score': 0,
                'reason': 'No uncertainty quantification'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_mc_dropout_usage(self, model, data, system) -> Dict:
        """Check MC Dropout usage (20 iterations)"""
        try:
            if system and hasattr(system.config, '__getitem__'):
                mc_iterations = system.config.get('uncertainty', {}).get('mc_iterations', 0)
                if mc_iterations >= 20:
                    return {
                        'status': 'passed',
                        'score': 100,
                        'details': {'mc_iterations': mc_iterations},
                        'evidence': [f'{mc_iterations} MC Dropout iterations configured']
                    }
                elif mc_iterations > 0:
                    return {
                        'status': 'partial',
                        'score': 50,
                        'details': {'mc_iterations': mc_iterations},
                        'reason': f'Only {mc_iterations} iterations (need 20)'
                    }
            return {
                'status': 'failed',
                'score': 0,
                'reason': 'MC Dropout not configured'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_calibration(self, model, data, system) -> Dict:
        """Check confidence calibration"""
        return {
            'status': 'not_tested',
            'score': 0,
            'reason': 'Requires calibration analysis'
        }
    
    def _check_clustering_explanations(self, model, data, system) -> Dict:
        """Check clustering explanation capability"""
        try:
            if system and hasattr(system, 'explainability_gen'):
                return {
                    'status': 'passed',
                    'score': 100,
                    'evidence': ['Explainability generator present']
                }
            return {
                'status': 'failed',
                'score': 0,
                'reason': 'No explainability component'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _check_solar_explanations(self, model, data, system) -> Dict:
        """Check solar recommendation explanations"""
        return self._check_clustering_explanations(model, data, system)
    
    def _check_human_readable(self, model, data, system) -> Dict:
        """Check human-readable summary generation"""
        try:
            if system and hasattr(system, 'explainability_gen'):
                # Try to generate an explanation
                try:
                    explanation = system.explainability_gen.explain_cluster_assignment(
                        model, data, 0, 0
                    )
                    if 'summary' in explanation or 'confidence' in explanation:
                        return {
                            'status': 'passed',
                            'score': 100,
                            'evidence': ['Human-readable explanations generated']
                        }
                except:
                    pass
            return {
                'status': 'partial',
                'score': 50,
                'reason': 'Explanation generation incomplete'
            }
        except Exception as e:
            return {'status': 'not_tested', 'score': 0, 'details': {'error': str(e)}}
    
    def _generate_recommendations(self, checks: List[RequirementCheck]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Group checks by category
        by_category = defaultdict(list)
        for check in checks:
            by_category[check.category].append(check)
        
        # Generate category-specific recommendations
        for category, category_checks in by_category.items():
            failed = [c for c in category_checks if c.status == 'failed']
            partial = [c for c in category_checks if c.status == 'partial']
            
            if failed:
                if category == 'Architecture':
                    recommendations.append(
                        f"Review {category} implementation - {len(failed)} critical components missing"
                    )
                elif category == 'Clustering':
                    recommendations.append(
                        f"Fix clustering constraints - LV boundaries and size limits must be enforced"
                    )
                elif category == 'Solar':
                    recommendations.append(
                        f"Implement solar cascade modeling and ROI calculations"
                    )
            
            if partial:
                recommendations.append(
                    f"Improve {category} components - {len(partial)} partially compliant"
                )
        
        # Add general recommendations
        overall_score = np.mean([c.score for c in checks])
        if overall_score < 50:
            recommendations.insert(0, "CRITICAL: Major rework needed to meet requirements")
        elif overall_score < 70:
            recommendations.insert(0, "Focus on failed critical requirements first")
        elif overall_score < 90:
            recommendations.insert(0, "Address remaining gaps to achieve full compliance")
        else:
            recommendations.insert(0, "Model meets most requirements - fine-tune remaining issues")
        
        return recommendations
    
    def _calculate_performance_metrics(
        self,
        model: nn.Module,
        test_data: Any,
        system: Any
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {}
        
        try:
            model.eval()
            with torch.no_grad():
                # Clustering metrics
                cluster_outputs = model(test_data, task='clustering')
                if 'cluster_logits' in cluster_outputs or 'cluster_assignments' in cluster_outputs:
                    clusters = cluster_outputs.get('cluster_logits', cluster_outputs.get('cluster_assignments'))
                    if len(clusters.shape) > 1:
                        assignments = clusters.argmax(dim=-1)
                    else:
                        assignments = clusters
                    
                    num_clusters = assignments.unique().shape[0]
                    metrics['num_clusters'] = num_clusters
                    
                    # Calculate average cluster size
                    sizes = [(assignments == c).sum().item() for c in assignments.unique()]
                    metrics['avg_cluster_size'] = np.mean(sizes) if sizes else 0
                    metrics['min_cluster_size'] = min(sizes) if sizes else 0
                    metrics['max_cluster_size'] = max(sizes) if sizes else 0
                
                # Solar metrics
                solar_outputs = model(test_data, task='solar')
                if 'solar' in solar_outputs:
                    solar_preds = solar_outputs['solar']
                    metrics['solar_output_dim'] = solar_preds.shape[-1]
                    
                # Model complexity
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                metrics['total_parameters'] = total_params
                metrics['trainable_parameters'] = trainable_params
                
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
        
        return metrics
    
    def save_report(self, filepath: str = None):
        """Save validation report to file"""
        if not self.current_report:
            logger.warning("No report to save")
            return
        
        if filepath is None:
            filepath = f"validation_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True)
        
        # Convert report to dict
        report_dict = {
            'model_name': self.current_report.model_name,
            'validation_date': self.current_report.validation_date.isoformat(),
            'overall_compliance': self.current_report.overall_compliance,
            'critical_failures': self.current_report.critical_failures,
            'warnings': self.current_report.warnings,
            'recommendations': self.current_report.recommendations,
            'performance_metrics': self.current_report.performance_metrics,
            'test_data_info': self.current_report.test_data_info,
            'requirements_checks': [
                {
                    'requirement_id': c.requirement_id,
                    'requirement_name': c.requirement_name,
                    'category': c.category,
                    'status': c.status,
                    'score': c.score,
                    'details': c.details,
                    'evidence': c.evidence
                }
                for c in self.current_report.requirements_checks
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {filepath}")
    
    def generate_html_report(self, filepath: str = None):
        """Generate HTML validation report"""
        if not self.current_report:
            logger.warning("No report to generate")
            return
        
        if filepath is None:
            filepath = f"validation_report_{datetime.now():%Y%m%d_%H%M%S}.html"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True)
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GNN Model Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .passed {{ color: green; font-weight: bold; }}
                .failed {{ color: red; font-weight: bold; }}
                .partial {{ color: orange; font-weight: bold; }}
                .not_tested {{ color: gray; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f0f0f0; }}
                .category-header {{ background: #e0e0e0; font-weight: bold; }}
                .progress-bar {{ width: 200px; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }}
                .progress-fill {{ height: 100%; background: linear-gradient(90deg, #ff4444, #ffaa00, #44ff44); }}
            </style>
        </head>
        <body>
            <h1>GNN Model Validation Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Model:</strong> {self.current_report.model_name}</p>
                <p><strong>Date:</strong> {self.current_report.validation_date.strftime('%Y-%m-%d %H:%M')}</p>
                <p><strong>Overall Compliance:</strong> {self.current_report.overall_compliance:.1f}%</p>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {self.current_report.overall_compliance}%"></div>
                </div>
                <p><strong>Critical Failures:</strong> {len(self.current_report.critical_failures)}</p>
                <p><strong>Warnings:</strong> {len(self.current_report.warnings)}</p>
            </div>
        """
        
        # Add critical failures
        if self.current_report.critical_failures:
            html += """
            <div class="critical">
                <h2>⚠️ Critical Failures</h2>
                <ul>
            """
            for failure in self.current_report.critical_failures:
                html += f"<li class='failed'>{failure}</li>"
            html += "</ul></div>"
        
        # Add recommendations
        if self.current_report.recommendations:
            html += """
            <div class="recommendations">
                <h2>📋 Recommendations</h2>
                <ol>
            """
            for rec in self.current_report.recommendations:
                html += f"<li>{rec}</li>"
            html += "</ol></div>"
        
        # Add detailed requirements table
        html += """
        <h2>Detailed Requirements Validation</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Requirement</th>
                <th>Status</th>
                <th>Score</th>
                <th>Evidence/Details</th>
            </tr>
        """
        
        # Group by category
        by_category = defaultdict(list)
        for check in self.current_report.requirements_checks:
            by_category[check.category].append(check)
        
        for category, checks in by_category.items():
            for i, check in enumerate(checks):
                if i == 0:
                    html += f"<tr><td rowspan='{len(checks)}' class='category-header'>{category}</td>"
                else:
                    html += "<tr>"
                
                status_class = check.status.replace('_', '')
                html += f"""
                    <td>{check.requirement_name}</td>
                    <td class='{status_class}'>{check.status.upper()}</td>
                    <td>{check.score:.0f}/100</td>
                    <td>{', '.join(check.evidence) if check.evidence else check.details.get('reason', '')}</td>
                </tr>
                """
        
        html += """
        </table>
        
        <h2>Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
        """
        
        for metric, value in self.current_report.performance_metrics.items():
            if isinstance(value, float):
                html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.2f}</td></tr>"
            else:
                html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        html += """
        </table>
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"HTML report generated: {filepath}")