"""
GPU-Optimized Transformer-based Cryptocurrency Trading Model
Full-scale production model for high-end GPU servers with maximum parameters
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

class AdvancedPositionalEncoding(nn.Module):
    """
    Enhanced positional encoding with learnable parameters for better adaptation
    """
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Traditional sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Learnable position embeddings for fine-tuning
        self.learned_pe = nn.Embedding(max_len, d_model)
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        
        # Combine sinusoidal and learned embeddings
        sinusoidal = self.pe[:, :seq_len]
        learned = self.learned_pe(positions).unsqueeze(0)
        
        pos_encoding = self.alpha * sinusoidal + (1 - self.alpha) * learned
        return self.dropout(x + pos_encoding)

class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism for capturing different temporal patterns
    """
    def __init__(self, d_model: int, n_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Multiple attention scales
        self.short_term_attn = nn.MultiheadAttention(
            d_model, n_heads // 2, dropout=dropout, batch_first=True
        )
        self.medium_term_attn = nn.MultiheadAttention(
            d_model, n_heads // 2, dropout=dropout, batch_first=True
        )
        self.long_term_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Scale combination
        self.scale_combine = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def create_temporal_mask(self, seq_len, scale_factor):
        """Create attention masks for different temporal scales"""
        mask = torch.ones(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - scale_factor)
            mask[i, :start] = 0
        return mask.bool()
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Short-term attention (last 10 steps)
        short_mask = self.create_temporal_mask(seq_len, 10).to(x.device)
        short_out, _ = self.short_term_attn(x, x, x, attn_mask=short_mask)
        
        # Medium-term attention (last 30 steps)
        medium_mask = self.create_temporal_mask(seq_len, 30).to(x.device)
        medium_out, _ = self.medium_term_attn(x, x, x, attn_mask=medium_mask)
        
        # Long-term attention (all steps)
        long_out, attn_weights = self.long_term_attn(x, x, x, attn_mask=mask)
        
        # Combine scales
        combined = torch.cat([short_out, medium_out, long_out], dim=-1)
        output = self.scale_combine(combined)
        
        return self.norm(output + x), attn_weights

class EnhancedFeedForward(nn.Module):
    """
    Enhanced feed-forward network with gating and multiple pathways
    """
    def __init__(self, d_model: int, d_ff: int = 4096, dropout: float = 0.1):
        super().__init__()
        
        # Main pathway
        self.main_ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, d_model)
        )
        
        # Gating pathway
        self.gate_ff = nn.Sequential(
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Linear(d_ff // 2, d_model),
            nn.Sigmoid()
        )
        
        # Expert pathways for different market conditions
        self.bull_expert = nn.Linear(d_model, d_model)
        self.bear_expert = nn.Linear(d_model, d_model)
        self.sideways_expert = nn.Linear(d_model, d_model)
        
        # Market condition classifier
        self.market_classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # Main pathway
        main_output = self.main_ff(x)
        gate = self.gate_ff(x)
        gated_output = main_output * gate
        
        # Market-specific experts
        market_probs = self.market_classifier(x.mean(dim=1, keepdim=True))  # Pool over sequence
        market_probs = market_probs.expand(-1, x.size(1), -1)
        
        bull_out = self.bull_expert(x) * market_probs[:, :, 0:1]
        bear_out = self.bear_expert(x) * market_probs[:, :, 1:2]
        sideways_out = self.sideways_expert(x) * market_probs[:, :, 2:3]
        
        expert_output = bull_out + bear_out + sideways_out
        
        return gated_output + expert_output

class GPUTransformerBlock(nn.Module):
    """
    GPU-optimized transformer block with enhanced components
    """
    def __init__(self, d_model: int, n_heads: int = 16, d_ff: int = 4096, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiScaleAttention(d_model, n_heads, dropout)
        self.feed_forward = EnhancedFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-scale attention with residual connection
        attn_output, attn_weights = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Enhanced feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x, attn_weights

class ProductionCryptoTransformer(nn.Module):
    """
    Production-scale transformer for cryptocurrency trading
    Optimized for high-end GPU servers with maximum parameters
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 1024,  # Larger model dimension
        n_heads: int = 16,    # More attention heads
        n_layers: int = 12,   # Deeper network
        d_ff: int = 4096,     # Larger feed-forward
        max_seq_len: int = 200,  # Longer sequences
        dropout: float = 0.1,
        action_dim: int = 1,
        num_assets: int = 5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_assets = num_assets
        
        # Enhanced input processing
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Advanced positional encoding
        self.pos_encoding = AdvancedPositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            GPUTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Multiple specialized prediction heads
        
        # Portfolio allocation head (multi-asset)
        self.portfolio_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_assets),
            nn.Softmax(dim=-1)  # Portfolio weights sum to 1
        )
        
        # Risk assessment head
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)  # VaR, CVaR, Sharpe, Max Drawdown predictions
        )
        
        # Market regime detection head
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5),  # bull, bear, sideways, volatile, crisis
            nn.Softmax(dim=-1)
        )
        
        # Price prediction head
        self.price_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_assets * 3)  # next_price, price_1h, price_4h for each asset
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Volatility prediction head
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_assets),
            nn.ReLU()  # Volatility is always positive
        )
        
        # Cross-asset correlation head
        self.correlation_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, (num_assets * (num_assets - 1)) // 2),  # Upper triangular correlation matrix
            nn.Tanh()  # Correlations between -1 and 1
        )
        
    def forward(self, x, seq_lengths=None, return_attention=False):
        """
        Forward pass through production transformer
        x: (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Enhanced input processing
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Store attention weights for analysis
        attention_weights = []
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)
            if return_attention:
                attention_weights.append(attn_weights)
        
        x = self.layer_norm(x)
        
        # Use the last token for predictions (or average pooling for stability)
        if seq_lengths is not None:
            batch_indices = torch.arange(batch_size, device=x.device)
            last_indices = seq_lengths - 1
            final_hidden = x[batch_indices, last_indices]
        else:
            # Use both last token and global average for robustness
            last_token = x[:, -1]
            global_avg = x.mean(dim=1)
            final_hidden = 0.7 * last_token + 0.3 * global_avg
        
        # Multiple prediction heads
        portfolio_weights = self.portfolio_head(final_hidden)
        risk_metrics = self.risk_head(final_hidden)
        market_regime = self.regime_head(final_hidden)
        price_predictions = self.price_head(final_hidden).view(batch_size, self.num_assets, 3)
        confidence = self.confidence_head(final_hidden)
        volatility = self.volatility_head(final_hidden)
        correlations = self.correlation_head(final_hidden)
        
        outputs = {
            'portfolio_weights': portfolio_weights,
            'risk_metrics': risk_metrics,
            'market_regime': market_regime,
            'price_predictions': price_predictions,
            'confidence': confidence,
            'volatility': volatility,
            'correlations': correlations,
            'hidden_state': final_hidden
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            
        return outputs

class ProductionFeaturesExtractor(BaseFeaturesExtractor):
    """
    Production features extractor for Stable-Baselines3 integration
    """
    def __init__(
        self, 
        observation_space: gym.Space,
        features_dim: int = 1024,
        sequence_length: int = 200,
        num_assets: int = 5,
        **kwargs
    ):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[-1]
        self.sequence_length = sequence_length
        self.num_assets = num_assets
        
        self.transformer = ProductionCryptoTransformer(
            input_dim=input_dim,
            d_model=features_dim,
            max_seq_len=sequence_length,
            num_assets=num_assets,
            **kwargs
        )
        
        # Larger buffer for longer sequences
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
            
            seq_input = self.sequence_buffer
        else:
            # Batch training
            seq_input = observations.view(batch_size, -1, observations.size(-1))
        
        # Forward through transformer
        outputs = self.transformer(seq_input)
        
        return outputs['hidden_state']

class ProductionTradingEnvironment(gym.Env):
    """
    Production trading environment with enhanced features
    """
    def __init__(
        self,
        df: pd.DataFrame,
        assets: List[str] = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL'],
        sequence_length: int = 200,
        initial_amount: float = 1000000,  # $1M initial capital
        transaction_cost_pct: float = 0.0005,  # 0.05% transaction cost
        max_position_size: float = 0.3,  # Max 30% per asset
        **kwargs
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.assets = assets
        self.num_assets = len(assets)
        self.sequence_length = sequence_length
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.max_position_size = max_position_size
        
        # Enhanced feature dimensions
        feature_columns = [col for col in df.columns if col not in ['date', 'timestamp']]
        self.feature_dim = len(feature_columns)
        
        # Multi-asset action space: portfolio weights for each asset
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.num_assets,), 
            dtype=np.float32
        )
        
        # Enhanced observation space
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
        self.positions = np.zeros(self.num_assets)  # Positions in each asset
        self.last_prices = self._get_prices(self.current_step)
        
        # Performance tracking
        self.max_portfolio_value = self.initial_amount
        self.trade_history = []
        
        return self._get_observation()
    
    def _get_prices(self, step):
        """Get prices for all assets at given step"""
        prices = []
        for asset in self.assets:
            price_col = f'{asset}_close' if f'{asset}_close' in self.df.columns else 'close'
            prices.append(self.df.iloc[step][price_col] if price_col in self.df.columns else 100.0)
        return np.array(prices)
    
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
        
        current_prices = self._get_prices(self.current_step)
        
        # Normalize action to ensure portfolio weights sum to <= 1
        action = np.clip(action, 0, self.max_position_size)
        total_weight = np.sum(action)
        if total_weight > 1.0:
            action = action / total_weight
        
        # Calculate target portfolio value for each asset
        current_portfolio_value = self.cash + np.sum(self.positions * current_prices)
        target_values = action * current_portfolio_value
        current_values = self.positions * current_prices
        
        # Execute rebalancing trades
        total_transaction_cost = 0
        for i in range(self.num_assets):
            value_change = target_values[i] - current_values[i]
            
            if abs(value_change) > current_portfolio_value * 0.001:  # Min trade threshold
                transaction_cost = abs(value_change) * self.transaction_cost_pct
                total_transaction_cost += transaction_cost
                
                if value_change > 0:  # Buy
                    if self.cash >= abs(value_change) + transaction_cost:
                        shares_bought = abs(value_change) / current_prices[i]
                        self.positions[i] += shares_bought
                        self.cash -= abs(value_change) + transaction_cost
                        
                        self.trade_history.append({
                            'step': self.current_step,
                            'asset': self.assets[i],
                            'action': 'buy',
                            'shares': shares_bought,
                            'price': current_prices[i],
                            'cost': transaction_cost
                        })
                
                else:  # Sell
                    shares_to_sell = min(abs(value_change) / current_prices[i], self.positions[i])
                    if shares_to_sell > 0:
                        self.positions[i] -= shares_to_sell
                        self.cash += shares_to_sell * current_prices[i] - transaction_cost
                        
                        self.trade_history.append({
                            'step': self.current_step,
                            'asset': self.assets[i],
                            'action': 'sell',
                            'shares': shares_to_sell,
                            'price': current_prices[i],
                            'cost': transaction_cost
                        })
        
        # Move to next step
        self.current_step += 1
        next_prices = self._get_prices(self.current_step)
        
        # Calculate new portfolio value
        new_portfolio_value = self.cash + np.sum(self.positions * next_prices)
        
        # Calculate reward with multiple components
        returns = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        
        # Update max portfolio value for drawdown calculation
        if new_portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = new_portfolio_value
        
        drawdown = (self.max_portfolio_value - new_portfolio_value) / self.max_portfolio_value
        
        # Multi-objective reward function
        return_reward = returns * 100  # Scale returns
        risk_penalty = -drawdown * 50  # Penalize drawdowns
        transaction_penalty = -total_transaction_cost / self.portfolio_value * 1000  # Penalize excessive trading
        
        reward = return_reward + risk_penalty + transaction_penalty
        
        self.portfolio_value = new_portfolio_value
        
        done = self.current_step >= len(self.df) - 1
        
        info = {
            'portfolio_value': self.portfolio_value,
            'returns': returns,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'prices': next_prices.copy(),
            'drawdown': drawdown,
            'transaction_cost': total_transaction_cost,
            'num_trades': len(self.trade_history)
        }
        
        return self._get_observation(), reward, done, info

def create_production_model_config():
    """
    Configuration for production transformer model
    """
    return {
        'model_params': {
            'd_model': 1024,
            'n_heads': 16,
            'n_layers': 12,
            'd_ff': 4096,
            'dropout': 0.1,
            'max_seq_len': 200,
            'num_assets': 5
        },
        'training_params': {
            'learning_rate': 5e-5,
            'batch_size': 128,  # Large batches for GPU efficiency
            'n_epochs': 500,
            'warmup_steps': 2000,
            'weight_decay': 1e-6,
            'gradient_clip': 1.0
        },
        'environment_params': {
            'initial_amount': 1000000,
            'transaction_cost_pct': 0.0005,
            'sequence_length': 200,
            'max_position_size': 0.3
        }
    }

if __name__ == "__main__":
    # Test the production model
    config = create_production_model_config()
    
    # Create production model
    seq_len = 200
    batch_size = 128
    input_dim = 50  # Rich feature set
    
    model = ProductionCryptoTransformer(
        input_dim=input_dim,
        **config['model_params']
    )
    
    # Test forward pass
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    print("Testing Production Transformer Model...")
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        outputs = model(test_input, return_attention=True)
    
    print("\nModel Output Shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        elif isinstance(value, list):
            print(f"{key}: List of {len(value)} attention weight tensors")
        else:
            print(f"{key}: {type(value)}")
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    print("\nProduction Transformer Model created successfully!")
    print("Ready for GPU server deployment!")