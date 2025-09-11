"""
Transformer-based Cryptocurrency Trading Model
Enhanced FinRL implementation with transformer architecture for improved sequence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple, Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer to handle temporal relationships
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for capturing temporal dependencies
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, v), attention_weights
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        attention_output, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attention_output), attention_weights

class TransformerBlock(nn.Module):
    """
    Single transformer block with multi-head attention and feed-forward network
    """
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, attn_weights = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights

class CryptoTransformerNetwork(nn.Module):
    """
    Complete transformer network for cryptocurrency trading
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 100,
        dropout: float = 0.1,
        action_dim: int = 1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Market regime detection head
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # bull, bear, sideways
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, action_dim)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Volatility prediction head
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        
    def create_padding_mask(self, seq_len, actual_len):
        """Create padding mask for variable length sequences"""
        mask = torch.ones(1, seq_len, seq_len)
        if actual_len < seq_len:
            mask[:, actual_len:, :] = 0
            mask[:, :, actual_len:] = 0
        return mask
    
    def forward(self, x, seq_lengths=None):
        """
        Forward pass through transformer
        x: (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Store attention weights for analysis
        attention_weights = []
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)
            attention_weights.append(attn_weights)
        
        x = self.layer_norm(x)
        
        # Use the last token for predictions (or average pooling)
        # We'll use the last non-padded token
        if seq_lengths is not None:
            batch_indices = torch.arange(batch_size)
            last_indices = seq_lengths - 1
            final_hidden = x[batch_indices, last_indices]
        else:
            final_hidden = x[:, -1]  # Last token
        
        # Multiple prediction heads
        market_regime = F.softmax(self.regime_head(final_hidden), dim=-1)
        action = torch.tanh(self.action_head(final_hidden))
        confidence = self.confidence_head(final_hidden)
        volatility = self.volatility_head(final_hidden)
        
        return {
            'action': action,
            'market_regime': market_regime,
            'confidence': confidence,
            'volatility': volatility,
            'attention_weights': attention_weights,
            'hidden_state': final_hidden
        }

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for Stable-Baselines3 integration
    """
    def __init__(
        self, 
        observation_space: gym.Space,
        features_dim: int = 256,
        sequence_length: int = 50,
        **kwargs
    ):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[-1]
        self.sequence_length = sequence_length
        
        self.transformer = CryptoTransformerNetwork(
            input_dim=input_dim,
            d_model=features_dim,
            max_seq_len=sequence_length,
            **kwargs
        )
        
        # Buffer for maintaining sequence history
        self.register_buffer(
            'sequence_buffer', 
            torch.zeros(1, sequence_length, input_dim)
        )
        self.current_pos = 0
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.size(0)
        
        if batch_size == 1:
            # Single step inference - use buffer
            self.sequence_buffer[:, self.current_pos] = observations
            self.current_pos = (self.current_pos + 1) % self.sequence_length
            
            # Use the buffered sequence
            seq_input = self.sequence_buffer
        else:
            # Batch training - reshape observations to sequences
            seq_input = observations.view(batch_size, -1, observations.size(-1))
        
        # Forward through transformer
        outputs = self.transformer(seq_input)
        
        return outputs['hidden_state']

class EnhancedCryptoTradingEnv(gym.Env):
    """
    Enhanced trading environment with transformer-specific features
    """
    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int = 50,
        initial_amount: float = 100000,
        transaction_cost_pct: float = 0.001,
        **kwargs
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        
        # Calculate feature dimensions
        feature_columns = [col for col in df.columns if col not in ['date', 'timestamp']]
        self.feature_dim = len(feature_columns)
        
        # Action space: continuous position sizing [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: sequence of features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(sequence_length, self.feature_dim), 
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self):
        self.current_step = self.sequence_length
        self.portfolio_value = self.initial_amount
        self.cash = self.initial_amount
        self.holdings = 0.0
        self.last_price = self._get_price(self.current_step)
        
        return self._get_observation()
    
    def _get_price(self, step):
        # Assuming 'close' column exists for price
        return self.df.iloc[step]['close'] if 'close' in self.df.columns else self.df.iloc[step].iloc[0]
    
    def _get_observation(self):
        start_idx = max(0, self.current_step - self.sequence_length)
        end_idx = self.current_step
        
        obs_data = self.df.iloc[start_idx:end_idx]
        
        # Pad if necessary
        if len(obs_data) < self.sequence_length:
            padding_rows = self.sequence_length - len(obs_data)
            padding_data = pd.DataFrame(
                np.zeros((padding_rows, self.feature_dim)),
                columns=obs_data.columns
            )
            obs_data = pd.concat([padding_data, obs_data], ignore_index=True)
        
        return obs_data.select_dtypes(include=[np.number]).values.astype(np.float32)
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = self._get_price(self.current_step)
        
        # Calculate target position
        target_position = action[0] * 0.95  # Max 95% allocation
        
        # Calculate position change
        current_position_value = self.holdings * current_price
        current_total_value = self.cash + current_position_value
        target_position_value = target_position * current_total_value
        position_change = target_position_value - current_position_value
        
        # Execute trade with transaction costs
        if abs(position_change) > current_total_value * 0.001:  # Minimum trade threshold
            transaction_cost = abs(position_change) * self.transaction_cost_pct
            
            if position_change > 0:  # Buy
                available_cash = self.cash - transaction_cost
                if available_cash > 0:
                    buy_amount = min(position_change, available_cash)
                    shares_bought = buy_amount / current_price
                    self.holdings += shares_bought
                    self.cash -= buy_amount + transaction_cost
            else:  # Sell
                sell_value = min(abs(position_change), self.holdings * current_price)
                shares_sold = sell_value / current_price
                self.holdings -= shares_sold
                self.cash += sell_value - transaction_cost
        
        # Move to next step
        self.current_step += 1
        next_price = self._get_price(self.current_step)
        
        # Calculate reward
        new_portfolio_value = self.cash + self.holdings * next_price
        returns = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        
        # Risk-adjusted reward
        reward = returns - 0.5 * (returns ** 2)  # Penalize high volatility
        
        self.portfolio_value = new_portfolio_value
        
        done = self.current_step >= len(self.df) - 1
        
        info = {
            'portfolio_value': self.portfolio_value,
            'returns': returns,
            'cash': self.cash,
            'holdings': self.holdings,
            'current_price': next_price
        }
        
        return self._get_observation(), reward, done, info

def create_transformer_model_config():
    """
    Configuration for transformer-based crypto trading model
    """
    return {
        'model_params': {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 1024,
            'dropout': 0.1,
            'max_seq_len': 50
        },
        'training_params': {
            'learning_rate': 1e-4,
            'batch_size': 64,
            'n_epochs': 100,
            'warmup_steps': 1000,
            'weight_decay': 1e-5
        },
        'environment_params': {
            'initial_amount': 100000,
            'transaction_cost_pct': 0.001,
            'sequence_length': 50
        }
    }

if __name__ == "__main__":
    # Test the transformer model
    config = create_transformer_model_config()
    
    # Create dummy data for testing
    seq_len = 50
    batch_size = 32
    input_dim = 20
    
    model = CryptoTransformerNetwork(
        input_dim=input_dim,
        **config['model_params']
    )
    
    # Test forward pass
    test_input = torch.randn(batch_size, seq_len, input_dim)
    outputs = model(test_input)
    
    print("Model Output Shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Transformer model created successfully!")