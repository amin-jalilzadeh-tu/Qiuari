"""
Enhanced temporal processing layers with Transformer and LSTM for Energy GNN.
Handles complex temporal patterns, seasonal effects, and long-range dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequences.
    Includes time-of-day, day-of-week, and seasonal encodings.
    """
    
    def __init__(self, d_model: int, max_len: int = 96 * 7):  # 1 week at 15-min intervals
        super().__init__()
        self.d_model = d_model
        
        # Standard positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Time-aware encodings
        self.hour_embedding = nn.Embedding(24, d_model // 4)
        self.day_embedding = nn.Embedding(7, d_model // 4)
        self.month_embedding = nn.Embedding(12, d_model // 4)
        self.season_embedding = nn.Embedding(4, d_model // 4)
        
    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional and temporal encodings to input.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            timestamps: Optional timestamps for each position [batch, seq_len, 4] (hour, day, month, season)
            
        Returns:
            Encoded tensor with positional information
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Add standard positional encoding
        x = x + self.pe[:, :seq_len, :]
        
        # Add temporal encodings if timestamps provided
        if timestamps is not None:
            hour_enc = self.hour_embedding(timestamps[..., 0].long())
            day_enc = self.day_embedding(timestamps[..., 1].long())
            month_enc = self.month_embedding(timestamps[..., 2].long())
            season_enc = self.season_embedding(timestamps[..., 3].long())
            
            # Concatenate and project to d_model
            temporal_enc = torch.cat([hour_enc, day_enc, month_enc, season_enc], dim=-1)
            x = x + temporal_enc
        
        return x


class EnhancedTemporalTransformer(nn.Module):
    """
    Advanced Transformer for temporal energy data processing.
    Includes causal attention for forecasting and pattern-specific heads.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_heads: int = 8, num_layers: int = 4,
                 dropout: float = 0.1, max_seq_len: int = 96):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder layers with different configurations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Pattern-specific attention heads
        self.daily_pattern_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.weekly_pattern_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Causal mask for forecasting
        self.register_buffer(
            'causal_mask',
            self._generate_causal_mask(max_seq_len)
        )
        
        # Output projections
        self.pattern_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Forecasting head
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Generate causal mask for autoregressive attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool()
    
    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None,
                use_causal: bool = False) -> Dict[str, torch.Tensor]:
        """
        Process temporal sequence with Transformer.
        
        Args:
            x: Input temporal features [batch, seq_len, input_dim]
            timestamps: Temporal metadata [batch, seq_len, 4]
            use_causal: Whether to use causal mask for forecasting
            
        Returns:
            Dictionary with processed features and predictions
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Project input
        x_projected = self.input_projection(x)
        
        # Add positional encoding
        x_encoded = self.pos_encoder(x_projected, timestamps)
        
        # Apply transformer
        if use_causal:
            mask = self.causal_mask[:seq_len, :seq_len]
            transformer_out = self.transformer(x_encoded, mask=mask)
        else:
            transformer_out = self.transformer(x_encoded)
        
        # Extract daily patterns (assuming 96 timesteps = 1 day)
        if seq_len >= 96:
            daily_features = transformer_out[:, -96:, :]
            daily_attended, daily_weights = self.daily_pattern_attention(
                daily_features, daily_features, daily_features
            )
        else:
            daily_attended = transformer_out
            daily_weights = None
        
        # Extract weekly patterns if enough data
        weekly_attended = None
        weekly_weights = None
        if seq_len >= 96 * 7:
            weekly_features = transformer_out[:, -96*7:, :]
            weekly_attended, weekly_weights = self.weekly_pattern_attention(
                weekly_features, weekly_features, weekly_features
            )
        
        # Extract patterns
        patterns = self.pattern_extractor(transformer_out)
        
        # Generate forecasts (next timestep prediction)
        forecast = self.forecast_head(transformer_out[:, -1, :])
        
        # Detect anomalies
        anomaly_scores = self.anomaly_head(transformer_out)
        
        return {
            'encoded': transformer_out,
            'patterns': patterns,
            'daily_patterns': daily_attended,
            'weekly_patterns': weekly_attended,
            'forecast': forecast,
            'anomaly_scores': anomaly_scores,
            'daily_attention': daily_weights,
            'weekly_attention': weekly_weights,
            'final_hidden': transformer_out[:, -1, :]  # For downstream tasks
        }


class AdaptiveLSTM(nn.Module):
    """
    LSTM with adaptive memory gates for handling varying temporal patterns.
    Includes attention mechanism and skip connections.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Adaptive gates for memory control
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Attention over LSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Skip connection projection
        self.skip_projection = nn.Linear(input_dim, hidden_dim * 2)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, 
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Process sequence through adaptive LSTM.
        
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            initial_state: Optional initial hidden and cell states
            
        Returns:
            Dictionary with LSTM outputs and attention weights
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x, initial_state)
        
        # Apply adaptive memory gate
        gated_output = lstm_out * self.memory_gate(lstm_out)
        
        # Add skip connection
        skip = self.skip_projection(x)
        combined = gated_output + 0.3 * skip
        
        # Apply attention mechanism
        attention_scores = self.attention(combined)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum of outputs
        context = torch.sum(combined * attention_weights, dim=1)
        
        # Final output
        output = self.output_layer(combined)
        
        return {
            'output': output,
            'context': context,
            'attention_weights': attention_weights.squeeze(-1),
            'hidden_states': hidden,
            'cell_states': cell,
            'final_output': output[:, -1, :]
        }


class TemporalFusionNetwork(nn.Module):
    """
    Combines Transformer and LSTM for robust temporal processing.
    Fuses different temporal representations for better predictions.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Transformer branch
        self.transformer = EnhancedTemporalTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=3,
            dropout=dropout
        )
        
        # LSTM branch
        self.lstm = AdaptiveLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,  # Half size for concatenation
            num_layers=2,
            dropout=dropout
        )
        
        # Fusion mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Cross-attention between branches
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Task-specific heads
        self.peak_prediction = nn.Linear(hidden_dim, 1)
        self.pattern_classifier = nn.Linear(hidden_dim, 4)  # 4 pattern types
        self.seasonality_detector = nn.Linear(hidden_dim, 3)  # Daily, weekly, seasonal
        
    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process temporal data through fusion network.
        
        Args:
            x: Input temporal sequence [batch, seq_len, input_dim]
            timestamps: Temporal metadata
            
        Returns:
            Fused temporal representations and predictions
        """
        # Process through transformer
        transformer_out = self.transformer(x, timestamps, use_causal=False)
        trans_features = transformer_out['encoded']
        
        # Process through LSTM
        lstm_out = self.lstm(x)
        lstm_features = lstm_out['output']
        
        # Ensure compatible dimensions
        if trans_features.size(-1) != lstm_features.size(-1) * 2:
            # Project LSTM features to match transformer
            lstm_features = F.pad(lstm_features, (0, trans_features.size(-1) - lstm_features.size(-1)))
        
        # Cross-attention between branches
        cross_attended, cross_weights = self.cross_attention(
            trans_features, lstm_features, lstm_features
        )
        
        # Fusion with gating
        concat_features = torch.cat([trans_features, lstm_features], dim=-1)
        gate = self.fusion_gate(concat_features)
        fused = gate * trans_features + (1 - gate) * cross_attended
        
        # Final projection
        output = self.output_projection(fused)
        
        # Task-specific predictions
        final_hidden = output[:, -1, :]
        peak_pred = self.peak_prediction(final_hidden)
        pattern_logits = self.pattern_classifier(final_hidden)
        seasonality_logits = self.seasonality_detector(final_hidden)
        
        return {
            'fused_features': output,
            'transformer_features': trans_features,
            'lstm_features': lstm_features,
            'fusion_gate': gate,
            'cross_attention': cross_weights,
            'final_representation': final_hidden,
            'peak_prediction': peak_pred,
            'pattern_classification': F.softmax(pattern_logits, dim=-1),
            'seasonality_detection': F.sigmoid(seasonality_logits),
            'forecast': transformer_out.get('forecast'),
            'anomaly_scores': transformer_out.get('anomaly_scores')
        }


class SeasonalDecomposition(nn.Module):
    """
    Decomposes time series into trend, seasonal, and residual components.
    Uses learnable decomposition for energy-specific patterns.
    """
    
    def __init__(self, hidden_dim: int = 128, season_length: int = 96):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.season_length = season_length
        
        # Trend extractor
        self.trend_conv = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=season_length // 4,
            padding='same'
        )
        
        # Seasonal pattern extractor
        self.seasonal_conv = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=season_length,
            padding='same'
        )
        
        # Residual processor
        self.residual_processor = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Component fusion
        self.component_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose time series into components.
        
        Args:
            x: Input time series [batch, seq_len] or [batch, seq_len, 1]
            
        Returns:
            Dictionary with decomposed components
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Prepare for convolution [batch, channels, seq_len]
        x_conv = x.transpose(1, 2)
        
        # Extract trend
        trend = self.trend_conv(x_conv).transpose(1, 2)
        
        # Extract seasonal component
        seasonal = self.seasonal_conv(x_conv).transpose(1, 2)
        
        # Calculate residual
        trend_reconstructed = F.adaptive_avg_pool1d(trend.transpose(1, 2), 1)
        seasonal_reconstructed = F.adaptive_avg_pool1d(seasonal.transpose(1, 2), 1)
        residual = x - trend_reconstructed.transpose(1, 2) - seasonal_reconstructed.transpose(1, 2)
        residual_features = self.residual_processor(residual)
        
        # Combine components
        combined = torch.cat([trend, seasonal, residual_features], dim=-1)
        fused = self.component_fusion(combined)
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual_features,
            'fused': fused,
            'decomposed_representation': fused[:, -1, :]
        }