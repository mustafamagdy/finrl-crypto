"""
Enhanced Transformer Model for Cryptocurrency Trading - Phase 1 Implementation
Core architecture improvements: extended sequences, temporal attention, multi-scale processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple, Optional, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import warnings
warnings.filterwarnings('ignore')

# ==================== ENHANCED POSITIONAL ENCODING ====================

class EnhancedPositionalEncoding(nn.Module):
    """
    Enhanced positional encoding with learnable temporal embeddings
    """
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()

        # Fixed sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

        # Learnable temporal bias
        self.temporal_bias = nn.Parameter(torch.zeros(1, max_len, 1))

        # Time-aware scaling
        self.time_scale = nn.Parameter(torch.ones(1, max_len, 1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)

        # Apply sinusoidal positional encoding
        x = x + self.pe[:, :seq_len]

        # Apply learnable temporal bias
        temporal_bias = self.temporal_bias[:, :seq_len].expand(x.size(0), -1, -1)
        time_scale = self.time_scale[:, :seq_len].expand(x.size(0), -1, -1)

        x = x * time_scale + temporal_bias
        x = self.dropout(x)

        return x

# ==================== TEMPORAL ATTENTION BIAS ====================

class TemporalAttentionBias(nn.Module):
    """
    Time-decay attention bias that prioritizes recent data
    """
    def __init__(self, max_seq_len: int, decay_rate: float = 0.02):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.decay_rate = decay_rate

        # Create temporal bias matrix
        self.register_buffer('temporal_bias', self._create_temporal_bias())

    def _create_temporal_bias(self) -> torch.Tensor:
        """Create time-decay bias matrix"""
        # Create distance matrix
        positions = torch.arange(self.max_seq_len).unsqueeze(0)
        distances = torch.abs(positions - positions.T)

        # Apply exponential decay
        bias = torch.exp(-self.decay_rate * distances)

        # Mask future positions (for causal attention)
        causal_mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len))
        bias = bias * causal_mask

        # Reshape for attention scores
        bias = bias.unsqueeze(0)  # (1, seq_len, seq_len)

        return bias

    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal bias to attention scores
        attention_scores: (batch_size, num_heads, seq_len, seq_len)
        """
        seq_len = attention_scores.size(-1)
        bias = self.temporal_bias[:, :seq_len, :seq_len]

        # Expand bias to match batch size and heads
        bias = bias.expand(attention_scores.size(0), attention_scores.size(1), -1, -1)

        return attention_scores + bias

# ==================== ENHANCED MULTI-HEAD ATTENTION ====================

class EnhancedMultiHeadAttention(nn.Module):
    """
    Enhanced multi-head attention with temporal bias and gating
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 250
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Standard attention projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Enhanced features
        self.temporal_bias = TemporalAttentionBias(max_seq_len)
        self.attention_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply temporal bias
        scores = self.temporal_bias(scores)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Linear projections and split into heads
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention mechanism
        attn_output, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # Output projection
        output = self.w_o(attn_output)

        # Attention gating
        gate = self.attention_gate(x)
        output = output * gate + x * (1 - gate)  # Gated residual connection

        return self.layer_norm(output), attention_weights

# ==================== MULTI-SCALE PROCESSOR ====================

class MultiScaleProcessor(nn.Module):
    """
    Process multiple timeframe data simultaneously
    """
    def __init__(
        self,
        input_dims: Dict[int, int],  # timeframe -> input_dim
        d_model: int = 256,
        n_heads: int = 8
    ):
        super().__init__()
        self.scales = sorted(input_dims.keys())
        self.scale_processors = nn.ModuleDict()

        # Create processors for each scale with dynamic input dimension
        for scale in self.scales:
            self.scale_processors[str(scale)] = nn.Sequential(
                nn.Linear(input_dims[scale], d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(d_model)
            )

        # Scale fusion (simplified without cross-attention)
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model * len(self.scales), d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

    def forward(self, scale_inputs: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Process inputs from multiple timeframes
        scale_inputs: {timeframe: (batch_size, seq_len, input_dim)}
        """
        scale_features = []

        # Process each scale
        for scale in self.scales:
            if str(scale) in self.scale_processors and scale in scale_inputs:
                processed = self.scale_processors[str(scale)](scale_inputs[scale])
                scale_features.append(processed)

        if not scale_features:
            # Return zero tensor if no valid inputs
            batch_size = next(iter(scale_inputs.values())).size(0)
            return torch.zeros(batch_size, 1, self.scale_fusion[-2].out_features).to(scale_inputs[next(iter(scale_inputs.keys()))].device)

        # Handle different sequence lengths by pooling
        pooled_features = []
        for features in scale_features:
            # Global average pooling across sequence dimension
            pooled = features.mean(dim=1)  # (batch_size, d_model)
            pooled_features.append(pooled)

        # Concatenate pooled features
        concatenated = torch.cat(pooled_features, dim=1)  # (batch_size, n_scales * d_model)

        # Apply fusion
        fused = self.scale_fusion(concatenated)

        # Expand to match expected sequence length
        target_seq_len = max(feat.size(1) for feat in scale_features) if scale_features else 1
        expanded = fused.unsqueeze(1).expand(-1, target_seq_len, -1)

        return expanded

# ==================== ENHANCED TRANSFORMER BLOCK ====================

class EnhancedTransformerBlock(nn.Module):
    """
    Enhanced transformer block with multi-scale processing
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 250
    ):
        super().__init__()

        # Enhanced multi-head attention
        self.attention = EnhancedMultiHeadAttention(
            d_model, n_heads, dropout, max_seq_len
        )

        # Enhanced feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU activation for better performance
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Adaptive dropout
        self.adaptive_dropout = AdaptiveDropout(d_model)

    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, attention_weights = self.attention(x, mask)
        x = self.norm1(x + attn_output)

        # Feed-forward with adaptive dropout
        ff_output = self.feed_forward(x)
        ff_output = self.adaptive_dropout(ff_output)
        x = self.norm2(x + ff_output)

        return x, attention_weights

# ==================== ADAPTIVE DROPOUT ====================

class AdaptiveDropout(nn.Module):
    """
    Adaptive dropout based on input statistics
    """
    def __init__(self, d_model: int, base_rate: float = 0.1):
        super().__init__()
        self.base_rate = base_rate
        self.dropout_rate_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Predict dropout rate based on input
        rate = self.dropout_rate_predictor(x.mean(dim=1, keepdim=True))
        adaptive_rate = self.base_rate + rate * 0.2  # Scale to [0.1, 0.3]

        # Apply dropout with batch-specific rates
        if self.training:
            mask = torch.rand_like(x) > adaptive_rate
            return x * mask.float() / (1.0 - adaptive_rate)
        else:
            return x

# ==================== ENHANCED TRANSFORMER NETWORK ====================

class EnhancedCryptoTransformer(nn.Module):
    """
    Enhanced transformer network for cryptocurrency trading
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        n_heads: int = 16,
        n_layers: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 250,
        dropout: float = 0.15,
        action_dim: int = 1,
        use_multi_scale: bool = True,
        scales: Optional[List[int]] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_multi_scale = use_multi_scale

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Enhanced positional encoding
        self.pos_encoding = EnhancedPositionalEncoding(d_model, max_seq_len, dropout)

        # Multi-scale processing (disabled by default to avoid dimension issues)
        if use_multi_scale:
            scales = scales or [5, 15, 30, 60]
            # Will be initialized during forward pass with correct dimensions
            self.multi_scale_processor = None
            self.scales = scales
        else:
            self.multi_scale_processor = None
            self.scales = []

        # Enhanced transformer layers
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(d_model)

        # Enhanced prediction heads
        self.market_regime_head = self._create_regime_head(d_model)
        self.action_head = self._create_action_head(d_model, action_dim)
        self.confidence_head = self._create_confidence_head(d_model)
        self.volatility_head = self._create_volatility_head(d_model)
        self.risk_head = self._create_risk_head(d_model)

        # Attention weight storage for analysis
        self.attention_weights = []

    def _create_regime_head(self, d_model: int) -> nn.Module:
        """Create market regime classification head"""
        return nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4),  # Bull, Bear, Ranging, Volatile
            nn.Softmax(dim=-1)
        )

    def _create_action_head(self, d_model: int, action_dim: int) -> nn.Module:
        """Create action prediction head"""
        return nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Bounded actions [-1, 1]
        )

    def _create_confidence_head(self, d_model: int) -> nn.Module:
        """Create confidence estimation head"""
        return nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _create_volatility_head(self, d_model: int) -> nn.Module:
        """Create volatility prediction head"""
        return nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def _create_risk_head(self, d_model: int) -> nn.Module:
        """Create risk assessment head"""
        return nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),  # Low, Medium, High risk
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor, scale_inputs: Optional[Dict[int, torch.Tensor]] = None):
        """
        Forward pass through enhanced transformer
        x: (batch_size, seq_len, input_dim)
        scale_inputs: {timeframe: (batch_size, seq_len, input_dim)}
        """
        batch_size, seq_len, _ = x.size()

        # Input projection
        x = self.input_projection(x)

        # Multi-scale processing
        if self.use_multi_scale and scale_inputs is not None:
            # Initialize multi-scale processor dynamically if needed
            if self.multi_scale_processor is None:
                # Get feature dimensions from scale inputs
                scale_dims = {scale: scale_inputs[scale].size(-1) for scale in self.scales if scale in scale_inputs}
                if scale_dims:
                    self.multi_scale_processor = MultiScaleProcessor(
                        scale_dims, d_model=self.d_model, n_heads=8
                    )
                    self.multi_scale_processor.to(x.device)

            # Process multi-scale features
            if self.multi_scale_processor is not None:
                multi_scale_features = self.multi_scale_processor(scale_inputs)
                # Ensure compatible dimensions
                if multi_scale_features.size(-1) == x.size(-1):
                    x = x + multi_scale_features  # Residual connection

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer layers
        self.attention_weights = []
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)
            self.attention_weights.append(attn_weights)

        x = self.final_norm(x)

        # Use last token for predictions
        final_hidden = x[:, -1]

        # Multiple prediction heads
        market_regime = self.market_regime_head(final_hidden)
        action = self.action_head(final_hidden)
        confidence = self.confidence_head(final_hidden)
        volatility = self.volatility_head(final_hidden)
        risk_assessment = self.risk_head(final_hidden)

        return {
            'action': action,
            'market_regime': market_regime,
            'confidence': confidence,
            'volatility': volatility,
            'risk_assessment': risk_assessment,
            'attention_weights': self.attention_weights,
            'hidden_state': final_hidden
        }

# ==================== ENHANCED FEATURES EXTRACTOR ====================

class EnhancedTransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Enhanced features extractor for Stable-Baselines3 integration
    """
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        sequence_length: int = 250,
        use_multi_scale: bool = True,
        **kwargs
    ):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[-1]
        self.sequence_length = sequence_length
        self.use_multi_scale = use_multi_scale

        # Enhanced transformer
        self.transformer = EnhancedCryptoTransformer(
            input_dim=input_dim,
            d_model=features_dim,
            max_seq_len=sequence_length,
            use_multi_scale=use_multi_scale,
            **kwargs
        )

        # Enhanced sequence buffer
        self.register_buffer(
            'sequence_buffer',
            torch.zeros(1, sequence_length, input_dim)
        )
        self.current_pos = 0

        # Multi-scale buffers
        if use_multi_scale:
            self.scale_buffers = nn.ModuleDict()
            self.scale_positions = {}
            for scale in [5, 15, 30, 60]:
                self.scale_buffers[str(scale)] = nn.Parameter(
                    torch.zeros(1, sequence_length // (scale // 5), input_dim),
                    requires_grad=False
                )
                self.scale_positions[scale] = 0

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.size(0)

        if batch_size == 1:
            # Single step inference - use buffer
            self.sequence_buffer[:, self.current_pos] = observations
            self.current_pos = (self.current_pos + 1) % self.sequence_length

            # Use the buffered sequence
            seq_input = self.sequence_buffer

            # Prepare multi-scale inputs
            scale_inputs = {}
            if self.use_multi_scale:
                for scale in [5, 15, 30, 60]:
                    buffer = self.scale_buffers[str(scale)]
                    pos = self.scale_positions[scale]
                    buffer[:, pos] = observations
                    self.scale_positions[scale] = (pos + 1) % buffer.size(1)
                    scale_inputs[scale] = buffer
        else:
            # Batch training - reshape observations to sequences
            seq_input = observations.view(batch_size, -1, observations.size(-1))
            scale_inputs = None

        # Forward through enhanced transformer
        outputs = self.transformer(seq_input, scale_inputs)

        return outputs['hidden_state']

# ==================== CONFIGURATION ====================

def create_enhanced_transformer_config():
    """
    Configuration for enhanced transformer-based crypto trading model
    """
    return {
        'model_params': {
            'd_model': 512,           # Increased from 256
            'n_heads': 16,            # Increased from 8
            'n_layers': 8,            # Increased from 6
            'd_ff': 2048,             # Increased from 1024
            'dropout': 0.15,          # Increased from 0.1
            'max_seq_len': 250,       # Increased from 50
            'use_multi_scale': True,   # New feature
            'scales': [5, 15, 30, 60] # Multiple timeframes
        },
        'training_params': {
            'learning_rate': 5e-5,    # Reduced for stability
            'batch_size': 32,         # Reduced for larger sequences
            'n_epochs': 150,          # Increased training
            'warmup_steps': 2000,     # Increased warmup
            'weight_decay': 1e-5,
            'gradient_clipping': 1.0
        },
        'environment_params': {
            'initial_amount': 100000,
            'transaction_cost_pct': 0.001,
            'sequence_length': 250,   # Extended sequence
            'use_multi_scale': True
        }
    }

# ==================== TESTING ====================

if __name__ == "__main__":
    # Test enhanced transformer model
    config = create_enhanced_transformer_config()

    # Create dummy data for testing
    seq_len = 250  # Extended sequence
    batch_size = 16  # Reduced batch size
    input_dim = 25   # Increased features

    model = EnhancedCryptoTransformer(
        input_dim=input_dim,
        **config['model_params']
    )

    # Test forward pass
    test_input = torch.randn(batch_size, seq_len, input_dim)

    # Create multi-scale inputs for testing
    scale_inputs = {
        5: torch.randn(batch_size, seq_len, input_dim),
        15: torch.randn(batch_size, seq_len // 3, input_dim),
        30: torch.randn(batch_size, seq_len // 6, input_dim),
        60: torch.randn(batch_size, seq_len // 12, input_dim)
    }

    outputs = model(test_input, scale_inputs)

    print("Enhanced Transformer Model Output Shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🔧 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Model Size: {total_params * 4 / 1024 / 1024:.1f} MB")

    print("\n✅ Enhanced transformer model created successfully!")
    print("🚀 Phase 1 improvements implemented:")
    print("   - Extended sequence length (250 steps)")
    print("   - Temporal attention bias")
    print("   - Multi-scale processing")
    print("   - Enhanced architecture")
    print("   - Adaptive dropout")
    print("   - Multiple prediction heads")