# models/temporal_layers.py
"""
Temporal processing layers for time-aware energy pattern analysis
Processes consumption histories and creates dynamic hourly embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ConsumptionPatternExtractor(nn.Module):
    """Extracts key features from consumption time series"""
    
    def __init__(self, input_dim: int = 8, output_dim: int = 32):
        super().__init__()
        
        # For processing raw consumption data
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Statistical feature processor
        self.stats_processor = nn.Sequential(
            nn.Linear(5, 16),  # 5 stats: mean, std, max, min, range
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Pattern classifier (flat, single-peak, double-peak, variable)
        self.pattern_classifier = nn.Sequential(
            nn.Linear(output_dim + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, consumption_history):
        """
        Args:
            consumption_history: [batch_size, num_buildings, timesteps, features]
        
        Returns:
            pattern_features: [batch_size, num_buildings, output_dim]
            pattern_type: [batch_size, num_buildings, 4]
        """
        B, N, T, F = consumption_history.shape
        
        # Flatten for processing
        consumption_flat = consumption_history.reshape(B * N, T, F)
        
        # Extract features from each timestep
        timestep_features = []
        for t in range(T):
            feat = self.feature_extractor(consumption_flat[:, t, :])
            timestep_features.append(feat)
        
        timestep_features = torch.stack(timestep_features, dim=1)  # [B*N, T, output_dim]
        
        # Aggregate over time
        pattern_features = timestep_features.mean(dim=1)  # [B*N, output_dim]
        
        # Compute statistical features
        consumption_values = consumption_flat[:, :, 0]  # Assuming first feature is consumption
        stats = torch.stack([
            consumption_values.mean(dim=1),
            consumption_values.std(dim=1),
            consumption_values.max(dim=1)[0],
            consumption_values.min(dim=1)[0],
            consumption_values.max(dim=1)[0] - consumption_values.min(dim=1)[0]  # range
        ], dim=1)
        
        stats_features = self.stats_processor(stats)
        
        # Combine and classify pattern
        combined = torch.cat([pattern_features, stats_features], dim=-1)
        pattern_type = self.pattern_classifier(combined)
        
        # Reshape back
        pattern_features = pattern_features.reshape(B, N, -1)
        pattern_type = pattern_type.reshape(B, N, -1)
        
        return pattern_features, pattern_type


class TemporalSequenceEncoder(nn.Module):
    """GRU-based encoder for temporal sequences"""
    
    def __init__(self, 
                 input_dim: int = 32,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 bidirectional: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, sequence_features, lengths=None):
        """
        Args:
            sequence_features: [batch_size, num_buildings, timesteps, features]
            lengths: Optional sequence lengths for packing
        
        Returns:
            temporal_encoding: [batch_size, num_buildings, hidden_dim]
        """
        B, N, T, F = sequence_features.shape
        
        # Reshape for GRU
        sequence_flat = sequence_features.reshape(B * N, T, F)
        
        # Process through GRU
        gru_out, hidden = self.gru(sequence_flat)
        
        # Use last hidden state (combine forward and backward if bidirectional)
        if self.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            final_hidden = torch.cat([hidden_forward, hidden_backward], dim=-1)
        else:
            final_hidden = hidden[-1, :, :]
        
        # Project to output dimension
        temporal_encoding = self.output_projection(final_hidden)
        
        # Reshape back
        temporal_encoding = temporal_encoding.reshape(B, N, -1)
        
        return temporal_encoding


class HourlyEmbeddingSpecializer(nn.Module):
    """Creates hour-specific embeddings for dynamic behavior"""
    
    def __init__(self, embed_dim: int = 128, num_hours: int = 24):
        super().__init__()
        
        self.num_hours = num_hours
        self.embed_dim = embed_dim
        
        # Hour-specific transformations
        self.hourly_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_hours)
        ])
        
        # Hour embeddings
        self.hour_embeddings = nn.Embedding(num_hours, embed_dim)
        
        # Combination layer
        self.combiner = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
    def forward(self, embeddings, current_hour=None):
        """
        Args:
            embeddings: [batch_size, num_buildings, embed_dim]
            current_hour: Integer hour (0-23) or None for all hours
        
        Returns:
            hourly_embeddings: [batch_size, num_buildings, embed_dim] or
                              [batch_size, num_buildings, num_hours, embed_dim]
        """
        B, N, D = embeddings.shape
        device = embeddings.device
        
        if current_hour is not None:
            # Single hour
            hour_proj = self.hourly_projections[current_hour]
            hour_emb = self.hour_embeddings(torch.tensor([current_hour], device=device))
            hour_emb = hour_emb.expand(B, N, D)
            
            # Transform embeddings for this hour
            hourly = hour_proj(embeddings)
            
            # Combine with hour embedding
            combined = torch.cat([hourly, hour_emb], dim=-1)
            output = self.combiner(combined)
            
            return output
        else:
            # All hours
            all_hourly = []
            for h in range(self.num_hours):
                hour_proj = self.hourly_projections[h]
                hour_emb = self.hour_embeddings(torch.tensor([h], device=device))
                hour_emb = hour_emb.expand(B, N, D)
                
                # Transform for this hour
                hourly = hour_proj(embeddings)
                
                # Combine
                combined = torch.cat([hourly, hour_emb], dim=-1)
                output = self.combiner(combined)
                all_hourly.append(output)
            
            # Stack all hours
            all_hourly = torch.stack(all_hourly, dim=2)  # [B, N, 24, D]
            
            return all_hourly

class SeasonalWeekdayAdapter(nn.Module):
    """FIXED: Adapts embeddings based on season and weekday/weekend"""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        
        # Season embeddings (4 seasons)
        self.season_embeddings = nn.Embedding(4, 32)
        
        # Weekday embeddings (weekday=0, weekend=1)
        self.weekday_embeddings = nn.Embedding(2, 16)
        
        # Adaptation network
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim + 48, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, embeddings, season, is_weekend):
        """
        FIXED: Handle scalar season and is_weekend inputs properly
        """
        B, N, D = embeddings.shape
        device = embeddings.device
        
        # FIX: Handle scalar inputs
        if season.dim() == 0:
            season = season.unsqueeze(0)
        if is_weekend.dim() == 0:
            is_weekend = is_weekend.unsqueeze(0)
        
        # Get embeddings
        season_emb = self.season_embeddings(season)  # [1, 32] or [B, 32]
        weekday_emb = self.weekday_embeddings(is_weekend.long())  # [1, 16] or [B, 16]
        
        # FIX: Proper expansion
        if season_emb.dim() == 2:
            if season_emb.shape[0] == 1:
                season_emb = season_emb.expand(B, -1)  # [B, 32]
        season_emb = season_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, 32]
        
        if weekday_emb.dim() == 2:
            if weekday_emb.shape[0] == 1:
                weekday_emb = weekday_emb.expand(B, -1)  # [B, 16]
        weekday_emb = weekday_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, 16]
        
        # Concatenate and adapt
        combined = torch.cat([embeddings, season_emb, weekday_emb], dim=-1)
        adapted = self.adapter(combined)
        
        # Residual connection
        output = embeddings + 0.5 * adapted
        
        return output


class TemporalProcessor(nn.Module):
    """FIXED: Main temporal processing module"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        embed_dim = config.get('hidden_dim', 128)
        temporal_dim = 64
        num_hours = 24
        
        # Components
        self.pattern_extractor = ConsumptionPatternExtractor(
            input_dim=8,  # From your EnergyState features
            output_dim=32
        )
        
        self.sequence_encoder = TemporalSequenceEncoder(
            input_dim=32,  # Takes pattern features, not raw consumption
            hidden_dim=temporal_dim,
            num_layers=2,
            bidirectional=True
        )
        
        self.hourly_specializer = HourlyEmbeddingSpecializer(
            embed_dim=embed_dim,
            num_hours=num_hours
        )
        
        self.seasonal_adapter = SeasonalWeekdayAdapter(embed_dim)
        
        self.complementarity_scorer = TemporalComplementarityScorer(
            embed_dim=embed_dim,
            temporal_dim=temporal_dim
        )
        
        self.consumption_predictor = ConsumptionPredictor(
            embed_dim=embed_dim,
            temporal_dim=temporal_dim,
            prediction_horizon=num_hours
        )
        
        self.peak_identifier = PeakHourIdentifier(
            temporal_dim=temporal_dim,
            num_hours=num_hours
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim + temporal_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        logger.info("Initialized TemporalProcessor with all components")
    
    def forward(self, 
                embeddings_dict: Dict,
                temporal_data: Optional[Dict] = None,
                current_hour: Optional[int] = None,
                return_all_hours: bool = False):
        """
        FIXED: Process temporal information with correct dimensions
        """
        
        # Get building embeddings
        building_embeddings = embeddings_dict.get('building')
        if building_embeddings is None:
            raise ValueError("Building embeddings not found")
        
        # Add batch dimension if needed
        if building_embeddings.dim() == 2:
            building_embeddings = building_embeddings.unsqueeze(0)
        
        B, N, D = building_embeddings.shape
        device = building_embeddings.device
        
        # Process temporal data if provided
        if temporal_data is not None and 'consumption_history' in temporal_data:
            consumption_history = temporal_data['consumption_history']
            
            # Extract consumption patterns (8-dim → 32-dim)
            pattern_features, pattern_types = self.pattern_extractor(consumption_history)
            
            # FIX: Create proper sequence for encoder
            # The sequence encoder expects [B*N, T, 32] but pattern_features is [B, N, 32]
            # We need to create a temporal sequence from pattern features
            pattern_sequence = pattern_features.unsqueeze(2).expand(B, N, 24, -1)
            
            # Encode temporal sequences
            temporal_encoding = self.sequence_encoder(pattern_sequence)
            
        else:
            # Create dummy temporal features if not provided
            temporal_encoding = torch.randn(B, N, 64, device=device)
            pattern_features = torch.randn(B, N, 32, device=device)
        
        # Seasonal and weekday adaptation
        if temporal_data is not None:
            season = temporal_data.get('season', torch.tensor(0, device=device))
            is_weekend = temporal_data.get('is_weekend', torch.tensor(False, device=device))
        else:
            season = torch.tensor(0, device=device)
            is_weekend = torch.tensor(False, device=device)
        
        # FIX: Pass to adapter (now handles scalar inputs correctly)
        adapted_embeddings = self.seasonal_adapter(building_embeddings, season, is_weekend)
        
        # Create hourly specialized embeddings
        if return_all_hours:
            hourly_embeddings = self.hourly_specializer(adapted_embeddings, current_hour=None)
        else:
            hourly_embeddings = self.hourly_specializer(adapted_embeddings, current_hour=current_hour)
        
        # Compute temporal complementarity
        temporal_complementarity = self.complementarity_scorer(adapted_embeddings, temporal_encoding)
        
        # Predict future consumption
        consumption_predictions = self.consumption_predictor(adapted_embeddings, temporal_encoding)
        
        # Identify peak hours
        peak_hours, peak_probs = self.peak_identifier(temporal_encoding)
        
        # Combine embeddings with temporal features
        combined = torch.cat([adapted_embeddings, temporal_encoding], dim=-1)
        final_embeddings = self.output_projection(combined)
        
        # Update embeddings dictionary
        output_embeddings = embeddings_dict.copy()
        output_embeddings['building'] = final_embeddings.squeeze(0) if B == 1 else final_embeddings
        
        # Prepare output
        output = {
            'embeddings': output_embeddings,
            'temporal_encoding': temporal_encoding,
            'consumption_predictions': consumption_predictions.squeeze(0) if B == 1 else consumption_predictions,
            'temporal_complementarity': temporal_complementarity.squeeze(0) if B == 1 else temporal_complementarity,
            'peak_indicators': peak_hours.squeeze(0) if B == 1 else peak_hours,
            'peak_probabilities': peak_probs.squeeze(0) if B == 1 else peak_probs
        }
        
        if return_all_hours:
            output['hourly_embeddings'] = hourly_embeddings.squeeze(0) if B == 1 else hourly_embeddings
        
        return output


class TemporalComplementarityScorer(nn.Module):
    """Identifies temporal complementarity patterns between buildings"""
    
    def __init__(self, embed_dim: int = 128, temporal_dim: int = 64):
        super().__init__()
        
        self.pattern_analyzer = nn.Sequential(
            nn.Linear(embed_dim + temporal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.complementarity_scorer = nn.Sequential(
            nn.Linear(64, 32),  # 32*2 from pairs
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
    def forward(self, embeddings, temporal_features):
        """
        Args:
            embeddings: [batch_size, num_buildings, embed_dim]
            temporal_features: [batch_size, num_buildings, temporal_dim]
        
        Returns:
            complementarity_matrix: [batch_size, num_buildings, num_buildings]
        """
        B, N, D = embeddings.shape
        
        # Combine embeddings with temporal features
        combined = torch.cat([embeddings, temporal_features], dim=-1)
        patterns = self.pattern_analyzer(combined)  # [B, N, 32]
        
        # Compute pairwise complementarity
        patterns_i = patterns.unsqueeze(2).expand(B, N, N, -1)
        patterns_j = patterns.unsqueeze(1).expand(B, N, N, -1)
        
        # Concatenate pairs
        pairs = torch.cat([patterns_i, patterns_j], dim=-1)
        
        # Score complementarity
        scores = self.complementarity_scorer(pairs).squeeze(-1)
        
        # Make symmetric and convert to complementarity
        # Negative values = complementary (opposite patterns)
        scores = (scores + scores.transpose(1, 2)) / 2
        
        return scores


class ConsumptionPredictor(nn.Module):
    """Predicts future consumption based on temporal patterns"""
    
    def __init__(self, 
                 embed_dim: int = 128,
                 temporal_dim: int = 64,
                 prediction_horizon: int = 24):
        super().__init__()
        
        self.prediction_horizon = prediction_horizon
        
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim + temporal_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, prediction_horizon),
            nn.Softplus()  # Ensure positive consumption
        )
        
    def forward(self, embeddings, temporal_features):
        """
        Args:
            embeddings: [batch_size, num_buildings, embed_dim]
            temporal_features: [batch_size, num_buildings, temporal_dim]
        
        Returns:
            predictions: [batch_size, num_buildings, prediction_horizon]
        """
        combined = torch.cat([embeddings, temporal_features], dim=-1)
        predictions = self.predictor(combined)
        
        return predictions


class PeakHourIdentifier(nn.Module):
    """Identifies peak consumption hours for each building"""
    
    def __init__(self, temporal_dim: int = 64, num_hours: int = 24):
        super().__init__()
        
        self.peak_scorer = nn.Sequential(
            nn.Linear(temporal_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_hours),
            nn.Sigmoid()  # Probability of being peak hour
        )
        
        self.threshold = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, temporal_features):
        """
        Args:
            temporal_features: [batch_size, num_buildings, temporal_dim]
        
        Returns:
            peak_hours: [batch_size, num_buildings, num_hours] (binary mask)
            peak_probabilities: [batch_size, num_buildings, num_hours]
        """
        peak_probs = self.peak_scorer(temporal_features)
        peak_hours = (peak_probs > self.threshold).float()
        
        return peak_hours, peak_probs


class TemporalProcessor(nn.Module):
    """Main temporal processing module integrating all components"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        embed_dim = config.get('hidden_dim', 128)
        temporal_dim = 64
        num_hours = 24
        
        # Components
        self.pattern_extractor = ConsumptionPatternExtractor(
            input_dim=8,  # From your EnergyState features
            output_dim=32
        )
        
        self.sequence_encoder = TemporalSequenceEncoder(
            input_dim=32,
            hidden_dim=temporal_dim,
            num_layers=2,
            bidirectional=True
        )
        
        self.hourly_specializer = HourlyEmbeddingSpecializer(
            embed_dim=embed_dim,
            num_hours=num_hours
        )
        
        self.seasonal_adapter = SeasonalWeekdayAdapter(embed_dim)
        
        self.complementarity_scorer = TemporalComplementarityScorer(
            embed_dim=embed_dim,
            temporal_dim=temporal_dim
        )
        
        self.consumption_predictor = ConsumptionPredictor(
            embed_dim=embed_dim,
            temporal_dim=temporal_dim,
            prediction_horizon=num_hours
        )
        
        self.peak_identifier = PeakHourIdentifier(
            temporal_dim=temporal_dim,
            num_hours=num_hours
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim + temporal_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        logger.info("Initialized TemporalProcessor with all components")
    
    def forward(self, 
                embeddings_dict: Dict,
                temporal_data: Optional[Dict] = None,
                current_hour: Optional[int] = None,
                return_all_hours: bool = False):
        """
        Process temporal information and create time-aware embeddings
        """
        
        # Get building embeddings
        building_embeddings = embeddings_dict.get('building')
        if building_embeddings is None:
            raise ValueError("Building embeddings not found")
        
        # Add batch dimension if needed
        if building_embeddings.dim() == 2:
            building_embeddings = building_embeddings.unsqueeze(0)
        
        B, N, D = building_embeddings.shape
        device = building_embeddings.device
        
        # Process temporal data if provided
        if temporal_data is not None and 'consumption_history' in temporal_data:
            consumption_history = temporal_data['consumption_history']
            
            # Extract consumption patterns
            pattern_features, pattern_types = self.pattern_extractor(consumption_history)
            
            # FIX: Pass pattern_features (32-dim) to sequence encoder, not raw consumption (8-dim)
            # Create sequence from pattern features for temporal encoding
            # We need to expand pattern features to match temporal dimension
            pattern_sequence = pattern_features.unsqueeze(2).expand(B, N, 24, -1)
            
            # Encode temporal sequences using pattern features
            temporal_encoding = self.sequence_encoder(pattern_sequence)
            
        else:
            # Create dummy temporal features if not provided
            temporal_encoding = torch.randn(B, N, 64, device=device)
            pattern_features = torch.randn(B, N, 32, device=device)
        
        
        # Seasonal and weekday adaptation
        if temporal_data is not None:
            season = temporal_data.get('season', torch.tensor(0, device=device))
            is_weekend = temporal_data.get('is_weekend', torch.tensor(False, device=device))
        else:
            season = torch.tensor(0, device=device)
            is_weekend = torch.tensor(False, device=device)
        
        adapted_embeddings = self.seasonal_adapter(building_embeddings, season, is_weekend)
        
        # Create hourly specialized embeddings
        if return_all_hours:
            hourly_embeddings = self.hourly_specializer(adapted_embeddings, current_hour=None)
        else:
            hourly_embeddings = self.hourly_specializer(adapted_embeddings, current_hour=current_hour)
        
        # Compute temporal complementarity
        temporal_complementarity = self.complementarity_scorer(adapted_embeddings, temporal_encoding)
        
        # Predict future consumption
        consumption_predictions = self.consumption_predictor(adapted_embeddings, temporal_encoding)
        
        # Identify peak hours
        peak_hours, peak_probs = self.peak_identifier(temporal_encoding)
        
        # Combine embeddings with temporal features
        combined = torch.cat([adapted_embeddings, temporal_encoding], dim=-1)
        final_embeddings = self.output_projection(combined)
        
        # Update embeddings dictionary
        output_embeddings = embeddings_dict.copy()
        output_embeddings['building'] = final_embeddings.squeeze(0) if B == 1 else final_embeddings
        
        # Prepare output
        output = {
            'embeddings': output_embeddings,
            'temporal_encoding': temporal_encoding,
            'consumption_predictions': consumption_predictions.squeeze(0) if B == 1 else consumption_predictions,
            'temporal_complementarity': temporal_complementarity.squeeze(0) if B == 1 else temporal_complementarity,
            'peak_indicators': peak_hours.squeeze(0) if B == 1 else peak_hours,
            'peak_probabilities': peak_probs.squeeze(0) if B == 1 else peak_probs
        }
        
        if return_all_hours:
            output['hourly_embeddings'] = hourly_embeddings.squeeze(0) if B == 1 else hourly_embeddings
        
        return output


def create_temporal_processor(config: Dict) -> TemporalProcessor:
    """Factory function to create temporal processor"""
    return TemporalProcessor(config)