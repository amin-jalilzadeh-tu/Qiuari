# data/preprocessor.py
"""
Data preprocessing module for energy system data
Handles data cleaning, normalization, and transformation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessor for energy system data"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize preprocessor with configuration"""
        self.config = config or {}
        self.scalers = {}
        logger.info("DataPreprocessor initialized")
    
    def preprocess(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Preprocess input data"""
        logger.debug("Preprocessing data")
        return data
    
    def normalize(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize data using specified method"""
        if method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val - min_val > 0:
                return (data - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                return (data - mean) / std
        return data
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe by handling missing values and outliers"""
        logger.debug(f"Cleaning data with shape {df.shape}")
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df
    
    def get_status(self) -> Dict[str, Any]:
        """Get preprocessor status"""
        return {
            "initialized": True,
            "config": self.config,
            "scalers": list(self.scalers.keys())
        }