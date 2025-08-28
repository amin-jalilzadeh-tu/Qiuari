"""
Feature mapping for Dutch building data (Alliander/PDOK)
Maps actual database columns to expected feature names
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DutchBuildingFeatureMapper:
    """
    Maps Dutch building data columns to standardized feature names
    Handles BAG (Basisregistratie Adressen en Gebouwen) and Alliander data
    """
    
    def __init__(self):
        # Column mapping from Dutch names to standard names
        self.column_mapping = {
            # Geometric features
            'area': ['area', 'vloeroppervlakte1', 'vloeroppervlakte2', 'opp_adresseerbaarobject_m2'],
            'height': ['gem_hoogte', 'b3_h_maaiveld', 'height'],
            'volume': ['volume', 'b3_volume_lod13', 'b3_volume_lod22'],
            'perimeter': ['perimeter'],
            
            # Building characteristics
            'build_year': ['bouwjaar', 'year', 'building_year'],
            'building_type': ['gebouwtypen', 'building_type', 'typeadresseerbaarobject', 'housing_type'],
            'building_function': ['building_function', 'function', 'woonfunctie'],
            'residential_type': ['residential_type', 'woningtype'],
            'non_residential_type': ['non_residential_type'],
            'floors': ['gem_bouwlagen', 'floors'],
            'age_range': ['age_range'],
            
            # Energy features
            'energy_label': ['meestvoorkomendelabel', 'elabels', 'energy_label'],
            'annual_electricity': ['annual_electricity_kwh', 'annual_electricity'],
            'annual_heating': ['annual_heating_kwh', 'annual_heating'],
            'peak_electricity': ['peak_electricity_demand_kw', 'peak_electricity'],
            'peak_heating': ['peak_heating_demand_kw', 'peak_heating'],
            'wwr': ['average_wwr'],  # Window-to-wall ratio
            
            # Roof features (for solar potential)
            'roof_flat': ['b3_opp_dak_plat', 'roof_flat'],
            'roof_slanted': ['b3_opp_dak_schuin', 'roof_slanted'],
            'roof_area': ['suitable_roof_area', 'roof_area'],
            'roof_type': ['b3_dak_type'],
            
            # Wall features (for insulation)
            'wall_area': ['b3_opp_buitenmuur'],
            'ground_area': ['b3_opp_grond'],
            'shared_wall': ['b3_opp_scheidingsmuur'],
            
            # Location features
            'x_coord': ['x', 'x_coord'],
            'y_coord': ['y', 'y_coord'],
            'lon': ['lon'],
            'lat': ['lat'],
            'postcode': ['postcode', 'meestvoorkomendepostcode'],
            'district': ['wijknaam', 'district', 'district_name'],
            'neighborhood': ['buurtnaam', 'neighborhood', 'neighborhood_name'],
            
            # Identifiers
            'building_id': ['ogc_fid', 'pand_identificatie', 'pand_id', 'id', 'building_id'],
            'address_id': ['nummeraanduiding_id'],
            
            # Grid connection (from Alliander)
            'cable_group': ['cable_group_id', 'lv_group'],
            'transformer': ['transformer_id'],
            
            # Orientation (for solar)
            'orientation': ['building_orientation', 'building_orientation_cardinal', 'orientation'],
            'north_facade': ['north_facade_length', 'north_side', 'north_facade'],
            'south_facade': ['south_facade_length', 'south_side', 'south_facade'],
            'east_facade': ['east_facade_length', 'east_side', 'east_facade'],
            'west_facade': ['west_facade_length', 'west_side', 'west_facade'],
            
            # Shared walls
            'north_shared': ['north_shared_length', 'north_shared'],
            'south_shared': ['south_shared_length', 'south_shared'],
            'east_shared': ['east_shared_length', 'east_shared'],
            'west_shared': ['west_shared_length', 'west_shared'],
            'num_shared_walls': ['num_shared_walls'],
            'total_shared_length': ['total_shared_length']
        }
        
        # Feature engineering functions
        self.feature_engineers = {
            'age': self._calculate_age,
            'solar_potential': self._estimate_solar_potential,
            'insulation_quality': self._estimate_insulation,
            'consumption_estimate': self._estimate_consumption,
            # Removed building_compactness - redundant with real consumption data
            'shared_wall_ratio': self._calculate_shared_wall_ratio
        }
        
        # Value mappings
        self.energy_label_map = {
            'A': 7, 'B': 6, 'C': 5, 'D': 4,
            'E': 3, 'F': 2, 'G': 1,
            None: 0, 'unknown': 0, '': 0
        }
        
        self.building_type_map = {
            'woonfunctie': 1,
            'bijeenkomstfunctie': 2,
            'celfunctie': 3,
            'gezondheidszorgfunctie': 4,
            'industriefunctie': 5,
            'kantoorfunctie': 6,
            'logiesfunctie': 7,
            'onderwijsfunctie': 8,
            'sportfunctie': 9,
            'winkelfunctie': 10,
            'overige_gebruiksfunctie': 11
        }
        
    def find_column(self, df: pd.DataFrame, feature_name: str) -> Optional[str]:
        """
        Find the actual column name in dataframe for a feature
        """
        possible_names = self.column_mapping.get(feature_name, [feature_name])
        
        for col_name in possible_names:
            if col_name in df.columns:
                return col_name
        
        # Try case-insensitive match
        df_cols_lower = {col.lower(): col for col in df.columns}
        for col_name in possible_names:
            if col_name.lower() in df_cols_lower:
                return df_cols_lower[col_name.lower()]
        
        return None
    
    def map_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map Dutch building data to standardized features
        Returns dataframe with standardized column names
        """
        mapped_df = pd.DataFrame()
        feature_report = []
        
        # Map basic features
        for standard_name, possible_names in self.column_mapping.items():
            actual_col = self.find_column(df, standard_name)
            if actual_col:
                # Special handling for energy labels
                if standard_name == 'energy_label':
                    mapped_df[standard_name] = df[actual_col].map(self.energy_label_map).fillna(0)
                else:
                    # Try to convert to numeric, keep as is if it fails
                    values = df[actual_col]
                    try:
                        mapped_df[standard_name] = pd.to_numeric(values, errors='coerce').fillna(0)
                    except:
                        mapped_df[standard_name] = values
                feature_report.append(f"✓ {standard_name} <- {actual_col}")
            else:
                feature_report.append(f"✗ {standard_name} (not found)")
        
        # Engineer additional features
        for feature_name, engineer_func in self.feature_engineers.items():
            try:
                mapped_df[feature_name] = engineer_func(df)
                feature_report.append(f"✓ {feature_name} (engineered)")
            except Exception as e:
                feature_report.append(f"✗ {feature_name} (engineering failed: {e})")
        
        # Log mapping report
        logger.info("Feature mapping report:\n" + "\n".join(feature_report))
        
        return mapped_df
    
    def _calculate_age(self, df: pd.DataFrame) -> pd.Series:
        """Calculate building age from build year"""
        year_col = self.find_column(df, 'build_year')
        if year_col:
            current_year = datetime.now().year
            build_years = pd.to_numeric(df[year_col], errors='coerce')
            ages = current_year - build_years
            # Cap age at 200 years and handle invalid values
            ages = ages.clip(0, 200).fillna(50)  # Default to 50 years if unknown
            return ages
        return pd.Series([50] * len(df))  # Default age
    
    def _estimate_solar_potential(self, df: pd.DataFrame) -> pd.Series:
        """Estimate solar potential based on roof area and orientation"""
        potential = pd.Series([0.5] * len(df))  # Default medium potential
        
        # Check roof area
        roof_flat = self.find_column(df, 'roof_flat')
        roof_slanted = self.find_column(df, 'roof_slanted')
        
        if roof_flat and roof_slanted:
            total_roof = df[roof_flat].fillna(0) + df[roof_slanted].fillna(0)
            # Normalize to 0-1 scale (assuming max 200m² roof)
            potential = (total_roof / 200).clip(0, 1)
        elif roof_flat:
            potential = (df[roof_flat].fillna(0) / 200).clip(0, 1)
        
        # Adjust for orientation if available
        south_col = self.find_column(df, 'south_facade')
        if south_col:
            south_bonus = (df[south_col].fillna(0) / df[south_col].max()) * 0.2
            potential = (potential + south_bonus).clip(0, 1)
        
        return potential
    
    def _estimate_insulation(self, df: pd.DataFrame) -> pd.Series:
        """Estimate insulation quality based on age and energy label"""
        insulation = pd.Series([0.5] * len(df))  # Default medium insulation
        
        # Use energy label if available
        label_col = self.find_column(df, 'energy_label')
        if label_col:
            labels = df[label_col].map(self.energy_label_map).fillna(0)
            insulation = labels / 7.0  # Normalize to 0-1
        else:
            # Estimate from age
            age_col = self.find_column(df, 'build_year')
            if age_col:
                years = pd.to_numeric(df[age_col], errors='coerce').fillna(1970)
                # Newer buildings have better insulation
                insulation = ((years - 1900) / 120).clip(0, 1)
        
        return insulation
    
    def _estimate_consumption(self, df: pd.DataFrame) -> pd.Series:
        """Use REAL consumption data instead of estimating!"""
        # First try to use REAL annual consumption data
        annual_col = self.find_column(df, 'annual_electricity')
        if annual_col:
            # Convert annual kWh to daily average
            return df[annual_col].fillna(0) / 365.0
        
        # If no real data, try peak demand * hours
        peak_col = self.find_column(df, 'peak_electricity')
        if peak_col:
            # Rough conversion: peak * 8 hours average usage
            return df[peak_col].fillna(0) * 8
        
        # Only estimate if NO real data available
        area_col = self.find_column(df, 'area')
        if area_col:
            area = df[area_col].fillna(100)
            # Last resort estimate: 150 kWh/m²/year -> daily
            consumption = (area * 150 / 365).clip(10, 1000)
        else:
            consumption = pd.Series([50] * len(df))  # Default fallback
        
        return consumption
    
    def _calculate_compactness(self, df: pd.DataFrame) -> pd.Series:
        """Calculate building compactness (volume/surface ratio)"""
        volume_col = self.find_column(df, 'volume')
        area_col = self.find_column(df, 'area')
        height_col = self.find_column(df, 'height')
        
        # If volume exists, use it
        if volume_col and area_col:
            volume = df[volume_col].fillna(1)
            area = df[area_col].fillna(1)
            height = df[height_col].fillna(3) if height_col else 3
        # Otherwise calculate volume from area * height
        elif area_col and height_col:
            area = df[area_col].fillna(100)
            height = df[height_col].fillna(3)
            volume = area * height  # Approximate volume
        else:
            return pd.Series([0.5] * len(df))
            
        # Estimate surface area (2*floor + 4*walls)
        surface = 2 * area + 4 * np.sqrt(area) * height
        compactness = volume / (surface + 1e-6)
        return compactness.clip(0, 10) / 10  # Normalize
    
    def _calculate_shared_wall_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ratio of shared walls (for heat loss estimation)"""
        shared_col = self.find_column(df, 'shared_wall')
        wall_col = self.find_column(df, 'wall_area')
        
        if shared_col and wall_col:
            shared = df[shared_col].fillna(0)
            total = df[wall_col].fillna(1)
            ratio = shared / (total + shared + 1e-6)
            return ratio.clip(0, 1)
        
        return pd.Series([0.2] * len(df))  # Default 20% shared
    
    def get_feature_vector(self, df: pd.DataFrame, 
                          features: Optional[List[str]] = None) -> np.ndarray:
        """
        Get feature vector for model input
        """
        mapped = self.map_features(df)
        
        if features is None:
            # Use all available features from KG
            features = [
                # Basic geometric features
                'area', 'height',
                # Building characteristics
                'age', 'energy_label',
                # Location (if available)
                'x_coord', 'y_coord',
                # Roof features for solar
                'roof_area', 'roof_flat', 'roof_slanted',
                # Facade features
                'north_facade', 'south_facade', 'east_facade', 'west_facade',
                # Shared walls
                'num_shared_walls', 'total_shared_length',
                # Engineered features using REAL data
                'solar_potential', 'insulation_quality',
                'consumption_estimate',  # Now uses real annual_electricity_kwh!
                'shared_wall_ratio'
            ]
        
        # Create feature matrix
        feature_matrix = []
        for feat in features:
            if feat in mapped.columns:
                # Ensure numeric conversion
                values = pd.to_numeric(mapped[feat], errors='coerce').fillna(0).values
            else:
                logger.warning(f"Feature {feat} not available, using zeros")
                values = np.zeros(len(df))
            feature_matrix.append(values)
        
        # Ensure float32 type for PyTorch compatibility
        return np.array(feature_matrix, dtype=np.float32).T
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names"""
        return list(self.column_mapping.keys()) + list(self.feature_engineers.keys())
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate that dataframe has minimum required features
        """
        validation = {
            'valid': True,
            'missing_critical': [],
            'missing_optional': [],
            'available_features': [],
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
        # Critical features
        critical = ['building_id', 'x_coord', 'y_coord']
        for feat in critical:
            if not self.find_column(df, feat):
                validation['missing_critical'].append(feat)
                validation['valid'] = False
        
        # Optional but important features
        optional = ['area', 'height', 'build_year', 'energy_label']
        for feat in optional:
            if not self.find_column(df, feat):
                validation['missing_optional'].append(feat)
        
        # List available features
        for feat in self.column_mapping.keys():
            if self.find_column(df, feat):
                validation['available_features'].append(feat)
        
        return validation


# Singleton instance
feature_mapper = DutchBuildingFeatureMapper()


def test_mapper():
    """Test the feature mapper with sample data"""
    # Create sample Dutch building data
    sample_data = pd.DataFrame({
        'ogc_fid': ['FAA0000100', 'FAA0000101'],
        'bouwjaar': [1985, 2010],
        'gem_hoogte': [10.5, 15.0],
        'vloeroppervlakte1': [120, 200],
        'b3_opp_dak_plat': [60, 100],
        'meestvoorkomendelabel': ['C', 'A'],
        'x': [155000, 155100],
        'y': [463000, 463100],
        'wijknaam': ['Centrum', 'Centrum']
    })
    
    # Test mapping
    mapper = DutchBuildingFeatureMapper()
    mapped = mapper.map_features(sample_data)
    print("Mapped columns:", mapped.columns.tolist())
    
    # Test feature vector
    features = mapper.get_feature_vector(sample_data)
    print("Feature shape:", features.shape)
    
    # Test validation
    validation = mapper.validate_data(sample_data)
    print("Validation:", validation)


if __name__ == "__main__":
    test_mapper()