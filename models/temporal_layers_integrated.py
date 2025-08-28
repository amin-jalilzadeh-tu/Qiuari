# models/temporal_layers_integrated.py
"""
Streamlined temporal processing layers for Energy GNN
Includes pattern extraction, temporal complementarity, seasonal adaptation, peak identification
EXCLUDES consumption prediction (as requested)
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
        # Handle different input shapes
        if consumption_history.dim() == 3:
            # [N, T, F] -> [1, N, T, F]
            consumption_history = consumption_history.unsqueeze(0)
        
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
        
    def forward(self, pattern_features):
        """
        Args:
            pattern_features: [batch_size, num_buildings, features]
        
        Returns:
            temporal_encoding: [batch_size, num_buildings, hidden_dim]
        """
        B, N, F = pattern_features.shape
        
        # Create temporal sequence by repeating pattern features
        # This simulates temporal evolution
        pattern_sequence = pattern_features.unsqueeze(2).expand(B, N, 24, F)
        
        # Reshape for GRU
        sequence_flat = pattern_sequence.reshape(B * N, 24, F)
        
        # Process through GRU
        gru_out, hidden = self.gru(sequence_flat)
        
        # Use last hidden state
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
        
        # Simplified: single transformation with hour embedding
        self.hour_embeddings = nn.Embedding(num_hours, 32)
        
        self.hour_adapter = nn.Sequential(
            nn.Linear(embed_dim + 32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, embeddings, current_hour=None):
        """
        Args:
            embeddings: [batch_size, num_buildings, embed_dim]
            current_hour: Integer hour (0-23)
        
        Returns:
            hourly_embeddings: [batch_size, num_buildings, embed_dim]
        """
        B, N, D = embeddings.shape
        device = embeddings.device
        
        if current_hour is None:
            current_hour = 12  # Default to noon
        
        # Get hour embedding
        hour_tensor = torch.tensor([current_hour], device=device)
        hour_emb = self.hour_embeddings(hour_tensor)  # [1, 32]
        hour_emb = hour_emb.expand(B, N, -1)  # [B, N, 32]
        
        # Combine with embeddings
        combined = torch.cat([embeddings, hour_emb], dim=-1)
        output = self.hour_adapter(combined)
        
        # Residual connection
        output = embeddings + 0.3 * output
        
        return output


class SeasonalWeekdayAdapter(nn.Module):
    """Adapts embeddings based on season and weekday/weekend"""
    
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
        Args:
            embeddings: [batch_size, num_buildings, embed_dim]
            season: Scalar or tensor (0-3)
            is_weekend: Scalar or tensor boolean
        """
        B, N, D = embeddings.shape
        device = embeddings.device
        
        # Handle scalar inputs
        if not isinstance(season, torch.Tensor):
            season = torch.tensor(season, device=device)
        if not isinstance(is_weekend, torch.Tensor):
            is_weekend = torch.tensor(is_weekend, device=device)
        
        if season.dim() == 0:
            season = season.unsqueeze(0)
        if is_weekend.dim() == 0:
            is_weekend = is_weekend.unsqueeze(0)
        
        # Get embeddings
        season_emb = self.season_embeddings(season)  # [1, 32]
        weekday_emb = self.weekday_embeddings(is_weekend.long())  # [1, 16]
        
        # Expand to match batch and building dimensions
        season_emb = season_emb.expand(B, -1).unsqueeze(1).expand(-1, N, -1)  # [B, N, 32]
        weekday_emb = weekday_emb.expand(B, -1).unsqueeze(1).expand(-1, N, -1)  # [B, N, 16]
        
        # Concatenate and adapt
        combined = torch.cat([embeddings, season_emb, weekday_emb], dim=-1)
        adapted = self.adapter(combined)
        
        # Residual connection
        output = embeddings + 0.3 * adapted
        
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
        
        self.threshold = 0.7  # Fixed threshold instead of learnable parameter
        
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


class IntegratedTemporalProcessor(nn.Module):
    """
    Streamlined temporal processor for integration with base_gnn
    Excludes consumption prediction as requested
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        embed_dim = config.get('hidden_dim', 128)
        temporal_dim = 64
        num_hours = 24
        
        # Components (NO consumption predictor)
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
        
        self.peak_identifier = PeakHourIdentifier(
            temporal_dim=temporal_dim,
            num_hours=num_hours
        )
        
        # Output projection to combine temporal features with embeddings
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim + temporal_dim + 32, embed_dim),  # +32 for pattern features
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        logger.info("Initialized IntegratedTemporalProcessor (without consumption prediction)")
    
    def forward(self, 
                building_embeddings: torch.Tensor,
                temporal_data: Optional[torch.Tensor] = None,
                season: int = 0,
                is_weekend: bool = False,
                current_hour: int = 12):
        """
        Process temporal information and enhance building embeddings
        
        Args:
            building_embeddings: [N, D] or [B, N, D] building embeddings
            temporal_data: [N, T, F] or [B, N, T, F] temporal features
            season: Current season (0-3)
            is_weekend: Whether it's weekend
            current_hour: Current hour (0-23)
            
        Returns:
            Dict with enhanced embeddings and temporal features
        """
        
        # Handle batch dimension
        if building_embeddings.dim() == 2:
            building_embeddings = building_embeddings.unsqueeze(0)
        
        B, N, D = building_embeddings.shape
        device = building_embeddings.device
        
        # Process temporal data if provided
        if temporal_data is not None:
            if temporal_data.dim() == 3:
                temporal_data = temporal_data.unsqueeze(0)
            
            # Extract consumption patterns
            pattern_features, pattern_types = self.pattern_extractor(temporal_data)
            
            # Encode temporal sequences
            temporal_encoding = self.sequence_encoder(pattern_features)
            
        else:
            # Create dummy features if no temporal data
            pattern_features = torch.zeros(B, N, 32, device=device)
            pattern_types = torch.zeros(B, N, 4, device=device)
            temporal_encoding = torch.zeros(B, N, 64, device=device)
        
        # Apply seasonal and weekday adaptation
        adapted_embeddings = self.seasonal_adapter(building_embeddings, season, is_weekend)
        
        # Apply hourly specialization
        hourly_embeddings = self.hourly_specializer(adapted_embeddings, current_hour)
        
        # Compute temporal complementarity
        temporal_complementarity = self.complementarity_scorer(hourly_embeddings, temporal_encoding)
        
        # Identify peak hours
        peak_hours, peak_probs = self.peak_identifier(temporal_encoding)
        
        # Combine all features
        combined = torch.cat([
            hourly_embeddings,
            temporal_encoding,
            pattern_features
        ], dim=-1)
        
        final_embeddings = self.output_projection(combined)
        
        # Prepare output
        output = {
            'embeddings': final_embeddings.squeeze(0) if B == 1 else final_embeddings,
            'temporal_encoding': temporal_encoding.squeeze(0) if B == 1 else temporal_encoding,
            'pattern_features': pattern_features.squeeze(0) if B == 1 else pattern_features,
            'pattern_types': pattern_types.squeeze(0) if B == 1 else pattern_types,
            'temporal_complementarity': temporal_complementarity.squeeze(0) if B == 1 else temporal_complementarity,
            'peak_hours': peak_hours.squeeze(0) if B == 1 else peak_hours,
            'peak_probabilities': peak_probs.squeeze(0) if B == 1 else peak_probs
        }
        
        return output