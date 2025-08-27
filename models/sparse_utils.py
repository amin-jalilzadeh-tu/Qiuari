"""
Utility module to handle optional torch-sparse dependency.
"""

import warnings

# Try to import SparseTensor, handle Windows DLL issues
SPARSE_AVAILABLE = False
SparseTensor = None

try:
    import torch_sparse
    SparseTensor = torch_sparse.SparseTensor
    SPARSE_AVAILABLE = True
except (ImportError, OSError) as e:
    warnings.warn(f"torch-sparse not available: {e}. Some features will be disabled.")
    
    # Create a dummy SparseTensor class for compatibility
    class SparseTensor:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("SparseTensor requires torch-sparse to be installed")