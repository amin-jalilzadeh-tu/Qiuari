"""
Comprehensive test suite for Energy GNN System
Run with: python -m pytest test_all.py -v
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.constants import *
from utils.feature_mapping import feature_mapper
from utils.data_validator import DimensionValidator, Neo4jDataValidator
from models.base_gnn import create_gnn_model
from training.loss_functions import EnergyBalanceLoss

class TestConstants:
    """Test that constants are properly defined"""
    
    def test_constants_exist(self):
        assert DROPOUT_RATE > 0
        assert DEFAULT_HIDDEN_DIM > 0
        assert MIN_CLUSTER_SIZE > 0
        assert MAX_CLUSTER_SIZE > MIN_CLUSTER_SIZE

class TestFeatureMapping:
    """Test feature mapper for Dutch data"""
    
    def test_feature_mapper(self):
        # Create sample Dutch data
        sample_data = pd.DataFrame({
            'ogc_fid': ['TEST001'],
            'bouwjaar': [1990],
            'gem_hoogte': [10.0],
            'vloeroppervlakte1': [100.0],
            'x': [150000],
            'y': [450000]
        })
        
        # Test mapping
        features = feature_mapper.get_feature_vector(sample_data)
        assert features is not None
        assert features.shape[0] == 1
        assert features.shape[1] > 0
    
    def test_validation(self):
        sample_data = pd.DataFrame({
            'ogc_fid': ['TEST001'],
            'x': [150000],
            'y': [450000]
        })
        
        validation = feature_mapper.validate_data(sample_data)
        assert validation['valid'] == True
        assert validation['row_count'] == 1

class TestDimensionValidator:
    """Test dimension validation"""
    
    def test_input_validation(self):
        config = {'input_dim': None, 'hidden_dim': 128}
        validator = DimensionValidator(config)
        
        # Test dynamic dimension detection
        x = torch.randn(10, 25)
        assert validator.validate_input(x, "test")
        assert validator.input_dim == 25
        
        # Test dimension mismatch detection
        x2 = torch.randn(10, 30)
        with pytest.raises(ValueError):
            validator.validate_input(x2, "test2")
    
    def test_edge_validation(self):
        config = {'edge_dim': 3}
        validator = DimensionValidator(config)
        
        edge_index = torch.randint(0, 10, (2, 20))
        assert validator.validate_edge_index(edge_index, 10)
        
        # Test invalid edge index
        bad_edge = torch.randint(0, 20, (2, 20))
        with pytest.raises(ValueError):
            validator.validate_edge_index(bad_edge, 10)

class TestLossFunctions:
    """Test loss function fixes"""
    
    def test_edge_index_conversion(self):
        # Test the TODO fix
        loss_fn = EnergyBalanceLoss()
        
        # Create test data
        energy_sharing = torch.randn(10, 10)
        cluster_assignments = torch.randint(0, 3, (10,))
        positions = torch.randn(10, 2)
        
        # Should not crash
        loss, components = loss_fn(
            energy_sharing=energy_sharing,
            cluster_assignments=cluster_assignments,
            node_positions=positions
        )
        
        assert loss is not None
        assert 'balance' in components

class TestModelCreation:
    """Test dynamic model creation"""
    
    def test_dynamic_dimensions(self):
        # Test with None input_dim
        config = {
            'input_dim': None,
            'hidden_dim': 64,
            'num_layers': 2
        }
        
        # Should not crash
        model = create_gnn_model(config)
        assert model is not None
    
    def test_auto_detection(self):
        config = {
            'input_dim': None,
            'hidden_dim': 64
        }
        
        model = create_gnn_model(config)
        
        # First forward pass should set dimensions
        x = torch.randn(10, 30)
        edge_index = torch.randint(0, 10, (2, 20))
        
        # This would normally crash with old code
        # but now should auto-detect
        try:
            output = model(x, edge_index)
            assert True
        except:
            assert False, "Auto-detection failed"

class TestErrorHandling:
    """Test error handling improvements"""
    
    def test_neo4j_connection_error(self):
        # Test graceful failure
        try:
            validator = Neo4jDataValidator(
                uri="bolt://invalid:7687",
                user="test",
                password="test"
            )
            assert not validator.connected
        except:
            assert False, "Should handle connection error gracefully"
    
    def test_dimension_mismatch_error(self):
        validator = DimensionValidator({'input_dim': 10})
        
        x = torch.randn(5, 20)
        with pytest.raises(ValueError) as exc_info:
            validator.validate_input(x, "test")
        
        assert "expected 10, got 20" in str(exc_info.value).lower()

def test_all():
    """Run all tests"""
    pytest.main([__file__, '-v'])

if __name__ == "__main__":
    test_all()
