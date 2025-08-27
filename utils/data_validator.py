"""
Data validation utilities for Energy GNN System
Validates Neo4j data, dimensions, and data flow
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from neo4j import GraphDatabase
import logging
from datetime import datetime
import warnings

from utils.constants import (
    MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE,
    MAX_INPUT_DIM, MIN_INPUT_DIM,
    MAX_BUILDING_HEIGHT_M, MAX_BUILDING_AREA_M2,
    MAX_CONSUMPTION_KWH_DAY, MAX_GENERATION_KW,
    ERR_NO_NEO4J_CONNECTION, ERR_NO_BUILDINGS,
    ERR_DIMENSION_MISMATCH, WARN_FEW_SAMPLES
)

logger = logging.getLogger(__name__)


class Neo4jDataValidator:
    """
    Validates Neo4j knowledge graph data completeness and consistency
    """
    
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection for validation"""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.connected = True
            logger.info("Neo4j validator connected successfully")
        except Exception as e:
            self.connected = False
            self.driver = None
            logger.error(f"{ERR_NO_NEO4J_CONNECTION}: {e}")
    
    def validate_connection(self) -> Dict[str, Any]:
        """Validate Neo4j connection and database status"""
        validation = {
            'connected': False,
            'database_name': None,
            'neo4j_version': None,
            'errors': []
        }
        
        if not self.connected or not self.driver:
            validation['errors'].append(ERR_NO_NEO4J_CONNECTION)
            return validation
        
        try:
            with self.driver.session() as session:
                # Check connection
                result = session.run("CALL dbms.components()")
                components = result.single()
                if components:
                    validation['connected'] = True
                    validation['neo4j_version'] = components.get('versions', ['unknown'])[0]
                
                # Get database name
                result = session.run("CALL db.info()")
                db_info = result.single()
                if db_info:
                    validation['database_name'] = db_info.get('name', 'neo4j')
                    
        except Exception as e:
            validation['errors'].append(f"Connection test failed: {e}")
        
        return validation
    
    def validate_schema(self) -> Dict[str, Any]:
        """Validate expected node types and relationships exist"""
        validation = {
            'valid': True,
            'node_types': {},
            'relationship_types': {},
            'missing_types': [],
            'indexes': [],
            'constraints': []
        }
        
        if not self.connected:
            validation['valid'] = False
            return validation
        
        try:
            with self.driver.session() as session:
                # Check node types
                expected_nodes = ['Building', 'CableGroup', 'Transformer', 
                                 'Substation', 'AdjacencyCluster', 'TimeSlot']
                
                for node_type in expected_nodes:
                    query = f"MATCH (n:{node_type}) RETURN count(n) as count"
                    result = session.run(query)
                    count = result.single()['count']
                    validation['node_types'][node_type] = count
                    
                    if count == 0:
                        validation['missing_types'].append(node_type)
                        if node_type in ['Building', 'CableGroup']:
                            validation['valid'] = False
                
                # Check relationship types
                expected_rels = ['CONNECTED_TO', 'BELONGS_TO', 'SUPPLIES', 
                                'ADJACENT_TO', 'HAS_STATE', 'DURING']
                
                for rel_type in expected_rels:
                    query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
                    result = session.run(query)
                    count = result.single()['count']
                    validation['relationship_types'][rel_type] = count
                
                # Check indexes
                result = session.run("SHOW INDEXES")
                for record in result:
                    validation['indexes'].append(record['name'])
                
                # Check constraints
                result = session.run("SHOW CONSTRAINTS")
                for record in result:
                    validation['constraints'].append(record['name'])
                    
        except Exception as e:
            validation['valid'] = False
            validation['error'] = str(e)
        
        return validation
    
    def validate_buildings(self) -> Dict[str, Any]:
        """Validate building data completeness"""
        validation = {
            'valid': True,
            'total_buildings': 0,
            'missing_fields': {},
            'value_ranges': {},
            'orphaned_buildings': 0,
            'warnings': []
        }
        
        if not self.connected:
            validation['valid'] = False
            return validation
        
        try:
            with self.driver.session() as session:
                # Count buildings
                result = session.run("MATCH (b:Building) RETURN count(b) as count")
                validation['total_buildings'] = result.single()['count']
                
                if validation['total_buildings'] == 0:
                    validation['valid'] = False
                    validation['warnings'].append(ERR_NO_BUILDINGS)
                    return validation
                
                # Check required fields
                required_fields = ['ogc_fid', 'x', 'y']
                for field in required_fields:
                    query = f"""
                    MATCH (b:Building)
                    WHERE b.{field} IS NULL
                    RETURN count(b) as count
                    """
                    result = session.run(query)
                    missing = result.single()['count']
                    if missing > 0:
                        validation['missing_fields'][field] = missing
                        if field in ['ogc_fid', 'x', 'y']:
                            validation['valid'] = False
                
                # Check value ranges
                range_checks = [
                    ('height', 0, MAX_BUILDING_HEIGHT_M),
                    ('area', 0, MAX_BUILDING_AREA_M2),
                    ('annual_consumption_kwh', 0, MAX_CONSUMPTION_KWH_DAY * 365)
                ]
                
                for field, min_val, max_val in range_checks:
                    query = f"""
                    MATCH (b:Building)
                    WHERE b.{field} IS NOT NULL
                    RETURN min(b.{field}) as min_val, 
                           max(b.{field}) as max_val,
                           avg(b.{field}) as avg_val
                    """
                    result = session.run(query)
                    stats = result.single()
                    if stats:
                        validation['value_ranges'][field] = {
                            'min': stats['min_val'],
                            'max': stats['max_val'],
                            'avg': stats['avg_val']
                        }
                        
                        if stats['max_val'] and stats['max_val'] > max_val:
                            validation['warnings'].append(
                                f"{field} exceeds expected maximum: {stats['max_val']} > {max_val}"
                            )
                
                # Check orphaned buildings (not connected to cable groups)
                query = """
                MATCH (b:Building)
                WHERE NOT EXISTS((b)-[:CONNECTED_TO]->(:CableGroup))
                RETURN count(b) as count
                """
                result = session.run(query)
                validation['orphaned_buildings'] = result.single()['count']
                
                if validation['orphaned_buildings'] > validation['total_buildings'] * 0.1:
                    validation['warnings'].append(
                        f"High number of orphaned buildings: {validation['orphaned_buildings']}"
                    )
                    
        except Exception as e:
            validation['valid'] = False
            validation['error'] = str(e)
        
        return validation
    
    def validate_cable_groups(self) -> Dict[str, Any]:
        """Validate cable group data and connectivity"""
        validation = {
            'valid': True,
            'total_groups': 0,
            'group_sizes': {},
            'invalid_groups': [],
            'disconnected_groups': 0
        }
        
        if not self.connected:
            validation['valid'] = False
            return validation
        
        try:
            with self.driver.session() as session:
                # Count cable groups
                query = "MATCH (cg:CableGroup) RETURN count(cg) as count"
                result = session.run(query)
                validation['total_groups'] = result.single()['count']
                
                # Check group sizes
                query = """
                MATCH (b:Building)-[:CONNECTED_TO]->(cg:CableGroup)
                WITH cg, count(b) as size
                RETURN cg.group_id as group_id, size
                ORDER BY size DESC
                """
                result = session.run(query)
                
                for record in result:
                    group_id = record['group_id']
                    size = record['size']
                    validation['group_sizes'][group_id] = size
                    
                    if size < MIN_CLUSTER_SIZE or size > MAX_CLUSTER_SIZE:
                        validation['invalid_groups'].append({
                            'group_id': group_id,
                            'size': size,
                            'issue': 'size_violation'
                        })
                
                # Check disconnected groups
                query = """
                MATCH (cg:CableGroup)
                WHERE NOT EXISTS((cg)-[:CONNECTS_TO]->(:Transformer))
                RETURN count(cg) as count
                """
                result = session.run(query)
                validation['disconnected_groups'] = result.single()['count']
                
        except Exception as e:
            validation['valid'] = False
            validation['error'] = str(e)
        
        return validation
    
    def validate_temporal_data(self) -> Dict[str, Any]:
        """Validate temporal energy data if available"""
        validation = {
            'has_temporal': False,
            'time_slots': 0,
            'energy_states': 0,
            'time_range': None
        }
        
        if not self.connected:
            return validation
        
        try:
            with self.driver.session() as session:
                # Check for time slots
                query = "MATCH (ts:TimeSlot) RETURN count(ts) as count"
                result = session.run(query)
                validation['time_slots'] = result.single()['count']
                
                if validation['time_slots'] > 0:
                    validation['has_temporal'] = True
                    
                    # Get time range
                    query = """
                    MATCH (ts:TimeSlot)
                    RETURN min(ts.timestamp) as start_time,
                           max(ts.timestamp) as end_time
                    """
                    result = session.run(query)
                    time_range = result.single()
                    if time_range:
                        validation['time_range'] = {
                            'start': time_range['start_time'],
                            'end': time_range['end_time']
                        }
                
                # Check energy states
                query = "MATCH (es:EnergyState) RETURN count(es) as count"
                result = session.run(query)
                validation['energy_states'] = result.single()['count']
                
        except Exception as e:
            validation['error'] = str(e)
        
        return validation
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        report_lines = [
            "=" * 60,
            "NEO4J DATA VALIDATION REPORT",
            f"Generated: {datetime.now()}",
            "=" * 60,
            ""
        ]
        
        # Connection validation
        conn_val = self.validate_connection()
        report_lines.append("CONNECTION STATUS:")
        report_lines.append(f"  Connected: {conn_val['connected']}")
        report_lines.append(f"  Database: {conn_val.get('database_name', 'N/A')}")
        report_lines.append(f"  Version: {conn_val.get('neo4j_version', 'N/A')}")
        if conn_val.get('errors'):
            report_lines.append(f"  Errors: {', '.join(conn_val['errors'])}")
        report_lines.append("")
        
        if not conn_val['connected']:
            report_lines.append("Cannot continue validation - no connection")
            return "\n".join(report_lines)
        
        # Schema validation
        schema_val = self.validate_schema()
        report_lines.append("SCHEMA VALIDATION:")
        report_lines.append(f"  Valid: {schema_val['valid']}")
        report_lines.append("  Node Types:")
        for node_type, count in schema_val['node_types'].items():
            status = "✓" if count > 0 else "✗"
            report_lines.append(f"    {status} {node_type}: {count}")
        report_lines.append("  Relationships:")
        for rel_type, count in schema_val['relationship_types'].items():
            report_lines.append(f"    - {rel_type}: {count}")
        report_lines.append(f"  Indexes: {len(schema_val['indexes'])}")
        report_lines.append(f"  Constraints: {len(schema_val['constraints'])}")
        report_lines.append("")
        
        # Building validation
        building_val = self.validate_buildings()
        report_lines.append("BUILDING DATA:")
        report_lines.append(f"  Total Buildings: {building_val['total_buildings']}")
        report_lines.append(f"  Orphaned: {building_val['orphaned_buildings']}")
        if building_val['missing_fields']:
            report_lines.append("  Missing Fields:")
            for field, count in building_val['missing_fields'].items():
                report_lines.append(f"    - {field}: {count} buildings")
        if building_val['warnings']:
            report_lines.append("  Warnings:")
            for warning in building_val['warnings']:
                report_lines.append(f"    ! {warning}")
        report_lines.append("")
        
        # Cable group validation
        cable_val = self.validate_cable_groups()
        report_lines.append("CABLE GROUPS:")
        report_lines.append(f"  Total Groups: {cable_val['total_groups']}")
        report_lines.append(f"  Disconnected: {cable_val['disconnected_groups']}")
        if cable_val['invalid_groups']:
            report_lines.append(f"  Invalid Groups: {len(cable_val['invalid_groups'])}")
        report_lines.append("")
        
        # Temporal data
        temporal_val = self.validate_temporal_data()
        report_lines.append("TEMPORAL DATA:")
        report_lines.append(f"  Available: {temporal_val['has_temporal']}")
        if temporal_val['has_temporal']:
            report_lines.append(f"  Time Slots: {temporal_val['time_slots']}")
            report_lines.append(f"  Energy States: {temporal_val['energy_states']}")
        report_lines.append("")
        
        # Overall status
        report_lines.append("=" * 60)
        overall_valid = (
            conn_val['connected'] and 
            schema_val['valid'] and 
            building_val['valid'] and
            building_val['total_buildings'] > 0
        )
        status = "✓ READY FOR TRAINING" if overall_valid else "✗ VALIDATION FAILED"
        report_lines.append(f"OVERALL STATUS: {status}")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()


class DimensionValidator:
    """
    Validates tensor dimensions throughout the model pipeline
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with expected dimensions from config"""
        self.input_dim = config.get('input_dim', None)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.edge_dim = config.get('edge_dim', 3)
        self.num_clusters = config.get('num_clusters', 20)
        self.temporal_dim = config.get('temporal_dim', 8)
        
    def validate_input(self, x: torch.Tensor, name: str = "input") -> bool:
        """Validate input tensor dimensions"""
        if x.dim() < 2:
            raise ValueError(f"{name} must be at least 2D, got {x.dim()}D")
        
        feature_dim = x.shape[-1]
        
        if feature_dim < MIN_INPUT_DIM or feature_dim > MAX_INPUT_DIM:
            warnings.warn(
                f"{name} feature dimension {feature_dim} outside expected range "
                f"[{MIN_INPUT_DIM}, {MAX_INPUT_DIM}]"
            )
        
        # Update input_dim if not set
        if self.input_dim is None:
            self.input_dim = feature_dim
            logger.info(f"Set input_dim to {self.input_dim} based on {name}")
        elif self.input_dim != feature_dim:
            raise ValueError(
                ERR_DIMENSION_MISMATCH.format(expected=self.input_dim, actual=feature_dim)
            )
        
        return True
    
    def validate_edge_index(self, edge_index: torch.Tensor, num_nodes: int) -> bool:
        """Validate edge index tensor"""
        if edge_index.dim() != 2:
            raise ValueError(f"edge_index must be 2D, got {edge_index.dim()}D")
        
        if edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}")
        
        if edge_index.max() >= num_nodes:
            raise ValueError(
                f"edge_index contains node {edge_index.max()} but only {num_nodes} nodes exist"
            )
        
        if edge_index.min() < 0:
            raise ValueError(f"edge_index contains negative values")
        
        return True
    
    def validate_edge_attr(self, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> bool:
        """Validate edge attribute tensor"""
        if edge_attr.dim() != 2:
            raise ValueError(f"edge_attr must be 2D, got {edge_attr.dim()}D")
        
        num_edges = edge_index.shape[1]
        if edge_attr.shape[0] != num_edges:
            raise ValueError(
                f"edge_attr has {edge_attr.shape[0]} edges but edge_index has {num_edges}"
            )
        
        if edge_attr.shape[1] != self.edge_dim:
            warnings.warn(
                f"edge_attr dimension {edge_attr.shape[1]} != expected {self.edge_dim}"
            )
            self.edge_dim = edge_attr.shape[1]
        
        return True
    
    def validate_output(self, output: Dict[str, torch.Tensor]) -> Dict[str, bool]:
        """Validate model output dimensions"""
        validation = {}
        
        for key, tensor in output.items():
            valid = True
            
            if 'cluster' in key and 'probs' in key:
                # Cluster probabilities should be [N, K]
                if tensor.dim() != 2:
                    valid = False
                    logger.error(f"{key} should be 2D, got {tensor.dim()}D")
                if tensor.shape[1] != self.num_clusters:
                    warnings.warn(
                        f"{key} has {tensor.shape[1]} clusters, expected {self.num_clusters}"
                    )
            
            elif 'embedding' in key:
                # Embeddings should be [N, hidden_dim]
                if tensor.dim() != 2:
                    valid = False
                    logger.error(f"{key} should be 2D, got {tensor.dim()}D")
                if tensor.shape[1] != self.hidden_dim:
                    warnings.warn(
                        f"{key} has dim {tensor.shape[1]}, expected {self.hidden_dim}"
                    )
            
            validation[key] = valid
        
        return validation
    
    def validate_batch(self, batch: Any) -> bool:
        """Validate a batch of data"""
        required_attrs = ['x', 'edge_index']
        
        for attr in required_attrs:
            if not hasattr(batch, attr):
                raise ValueError(f"Batch missing required attribute: {attr}")
        
        # Validate node features
        self.validate_input(batch.x, "batch.x")
        
        # Validate edge index
        self.validate_edge_index(batch.edge_index, batch.x.shape[0])
        
        # Validate edge attributes if present
        if hasattr(batch, 'edge_attr'):
            self.validate_edge_attr(batch.edge_attr, batch.edge_index)
        
        # Check batch consistency
        if hasattr(batch, 'batch'):
            if batch.batch.max() >= batch.num_graphs:
                raise ValueError("Batch indices exceed number of graphs")
        
        return True


def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate training configuration"""
    validation = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'adjusted': {}
    }
    
    # Check required fields
    required = ['training', 'model', 'data']
    for field in required:
        if field not in config:
            validation['errors'].append(f"Missing required config section: {field}")
            validation['valid'] = False
    
    # Validate training parameters
    if 'training' in config:
        train_cfg = config['training']
        
        # Check learning rate
        lr = train_cfg.get('learning_rate', 0.001)
        if lr > 0.1:
            validation['warnings'].append(f"Learning rate {lr} seems high")
        if lr < 1e-6:
            validation['warnings'].append(f"Learning rate {lr} seems very low")
        
        # Check batch size
        batch_size = train_cfg.get('batch_size', 32)
        if batch_size > 256:
            validation['warnings'].append(f"Batch size {batch_size} may cause memory issues")
    
    # Validate model parameters
    if 'model' in config:
        model_cfg = config['model']
        
        # Check hidden dimensions
        hidden_dim = model_cfg.get('hidden_dim', 128)
        if hidden_dim not in [32, 64, 128, 256, 512]:
            validation['warnings'].append(
                f"Hidden dim {hidden_dim} is non-standard, consider power of 2"
            )
        
        # Check number of layers
        num_layers = model_cfg.get('num_layers', 4)
        if num_layers > 10:
            validation['warnings'].append(f"Deep network with {num_layers} layers may be hard to train")
    
    return validation


def test_validators():
    """Test validation functions"""
    print("Testing validators...")
    
    # Test dimension validator
    config = {'input_dim': 17, 'hidden_dim': 128}
    dim_val = DimensionValidator(config)
    
    # Test valid input
    x = torch.randn(100, 17)
    assert dim_val.validate_input(x, "test_input")
    
    # Test edge index
    edge_index = torch.randint(0, 100, (2, 500))
    assert dim_val.validate_edge_index(edge_index, 100)
    
    # Test config validation
    test_config = {
        'training': {'learning_rate': 0.001, 'batch_size': 32},
        'model': {'hidden_dim': 128, 'num_layers': 4},
        'data': {}
    }
    val_result = validate_training_config(test_config)
    assert val_result['valid']
    
    print("✓ All validation tests passed")


if __name__ == "__main__":
    # Test validators
    test_validators()
    
    # Test Neo4j validator (requires connection)
    try:
        validator = Neo4jDataValidator(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        report = validator.generate_validation_report()
        print(report)
        validator.close()
    except Exception as e:
        print(f"Neo4j validation skipped: {e}")