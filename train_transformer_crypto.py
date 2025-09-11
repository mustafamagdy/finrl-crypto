"""
Transformer-based Cryptocurrency Trading Training Script
Advanced FinRL implementation with transformer architecture and enhanced features
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import ta
from typing import Dict, List, Tuple, Optional
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import transformer model
from transformer_crypto_model import (
    CryptoTransformerNetwork, 
    EnhancedCryptoTradingEnv,
    create_transformer_model_config
)

# FinRL imports
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config
from finrl import config_tickers

# Stable baselines
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class TransformerTrainingCallback(BaseCallback):
    """Custom callback for monitoring transformer training"""
    
    def __init__(self, save_freq: int = 10000, save_path: str = "./models/"):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        # Save model periodically
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"transformer_model_step_{self.n_calls}")
            self.model.save(model_path)
            print(f"Model saved at step {self.n_calls}")
            
        return True

def download_crypto_data(
    symbols: List[str] = ["BTC-USD"],
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    interval: str = "1h"
) -> pd.DataFrame:
    """Download cryptocurrency data with enhanced features"""
    
    print(f"📊 Downloading crypto data for {symbols}...")
    all_data = []
    
    for symbol in symbols:
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                print(f"⚠️ No data found for {symbol}")
                continue
                
            # Basic OHLCV
            df = df.reset_index()
            df['symbol'] = symbol.replace('-USD', '').lower()
            df['date'] = df['Datetime'] if 'Datetime' in df.columns else df.index
            
            # Rename columns to match FinRL format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            all_data.append(df)
            print(f"✅ Downloaded {len(df)} records for {symbol}")
            
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data downloaded for any symbol")
        
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    print(f"📈 Total combined data: {len(combined_df)} records")
    return combined_df

def calculate_enhanced_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators for transformer model"""
    
    print("🔧 Calculating enhanced technical indicators...")
    df = df.copy()
    
    # Group by symbol for indicator calculation
    enhanced_df = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        
        # Price-based indicators
        symbol_df['sma_10'] = ta.trend.sma_indicator(symbol_df['close'], window=10)
        symbol_df['sma_20'] = ta.trend.sma_indicator(symbol_df['close'], window=20)
        symbol_df['sma_50'] = ta.trend.sma_indicator(symbol_df['close'], window=50)
        symbol_df['ema_12'] = ta.trend.ema_indicator(symbol_df['close'], window=12)
        symbol_df['ema_26'] = ta.trend.ema_indicator(symbol_df['close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(close=symbol_df['close'])
        symbol_df['macd'] = macd.macd()
        symbol_df['macd_signal'] = macd.macd_signal()
        symbol_df['macd_histogram'] = macd.macd_diff()
        
        # RSI
        symbol_df['rsi'] = ta.momentum.rsi(symbol_df['close'], window=14)
        symbol_df['rsi_30'] = ta.momentum.rsi(symbol_df['close'], window=30)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=symbol_df['close'], window=20, window_dev=2)
        symbol_df['bb_high'] = bollinger.bollinger_hband()
        symbol_df['bb_low'] = bollinger.bollinger_lband()
        symbol_df['bb_mid'] = bollinger.bollinger_mavg()
        symbol_df['bb_width'] = (symbol_df['bb_high'] - symbol_df['bb_low']) / symbol_df['bb_mid']
        symbol_df['bb_position'] = (symbol_df['close'] - symbol_df['bb_low']) / (symbol_df['bb_high'] - symbol_df['bb_low'])
        
        # Volatility indicators
        symbol_df['atr'] = ta.volatility.average_true_range(symbol_df['high'], symbol_df['low'], symbol_df['close'])
        symbol_df['volatility_10'] = symbol_df['close'].pct_change().rolling(10).std()
        symbol_df['volatility_30'] = symbol_df['close'].pct_change().rolling(30).std()
        
        # Volume indicators
        symbol_df['volume_sma'] = ta.volume.volume_sma(symbol_df['close'], symbol_df['volume'])
        symbol_df['volume_ratio'] = symbol_df['volume'] / symbol_df['volume_sma']
        
        # Momentum indicators
        symbol_df['stoch_k'] = ta.momentum.stoch(symbol_df['high'], symbol_df['low'], symbol_df['close'])
        symbol_df['stoch_d'] = ta.momentum.stoch_signal(symbol_df['high'], symbol_df['low'], symbol_df['close'])
        symbol_df['williams_r'] = ta.momentum.williams_r(symbol_df['high'], symbol_df['low'], symbol_df['close'])
        
        # Trend indicators
        symbol_df['adx'] = ta.trend.adx(symbol_df['high'], symbol_df['low'], symbol_df['close'])
        symbol_df['cci'] = ta.trend.cci(symbol_df['high'], symbol_df['low'], symbol_df['close'])
        symbol_df['dx'] = ta.trend.vortex_indicator_pos(symbol_df['high'], symbol_df['low'], symbol_df['close'])
        
        # Price action features
        symbol_df['returns'] = symbol_df['close'].pct_change()
        symbol_df['log_returns'] = np.log(symbol_df['close'] / symbol_df['close'].shift(1))
        symbol_df['price_momentum_5'] = symbol_df['close'] / symbol_df['close'].shift(5) - 1
        symbol_df['price_momentum_10'] = symbol_df['close'] / symbol_df['close'].shift(10) - 1
        
        # Market regime features
        symbol_df['trend_strength'] = abs(symbol_df['close'] - symbol_df['sma_50']) / symbol_df['sma_50']
        symbol_df['market_regime'] = np.where(
            symbol_df['close'] > symbol_df['sma_50'], 1,  # Bull
            np.where(symbol_df['close'] < symbol_df['sma_50'], -1, 0)  # Bear, Sideways
        )
        
        # Time-based features
        symbol_df['hour'] = pd.to_datetime(symbol_df['date']).dt.hour
        symbol_df['day_of_week'] = pd.to_datetime(symbol_df['date']).dt.dayofweek
        symbol_df['hour_sin'] = np.sin(2 * np.pi * symbol_df['hour'] / 24)
        symbol_df['hour_cos'] = np.cos(2 * np.pi * symbol_df['hour'] / 24)
        symbol_df['dow_sin'] = np.sin(2 * np.pi * symbol_df['day_of_week'] / 7)
        symbol_df['dow_cos'] = np.cos(2 * np.pi * symbol_df['day_of_week'] / 7)
        
        enhanced_df.append(symbol_df)
    
    result_df = pd.concat(enhanced_df, ignore_index=True)
    result_df = result_df.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    # Fill NaN values
    result_df = result_df.fillna(method='bfill').fillna(0)
    
    print(f"✅ Enhanced features calculated. Total columns: {len(result_df.columns)}")
    return result_df

def prepare_transformer_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Prepare features specifically for transformer model"""
    
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'macd', 'macd_signal', 'macd_histogram',
        'rsi', 'rsi_30',
        'bb_high', 'bb_low', 'bb_mid', 'bb_width', 'bb_position',
        'atr', 'volatility_10', 'volatility_30',
        'volume_ratio', 'stoch_k', 'stoch_d', 'williams_r',
        'adx', 'cci', 'dx',
        'returns', 'log_returns', 'price_momentum_5', 'price_momentum_10',
        'trend_strength', 'market_regime',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]
    
    # Select available features
    available_features = [col for col in feature_columns if col in df.columns]
    
    # Create feature matrix
    feature_data = df[available_features].values
    
    # Normalize features
    scaler = StandardScaler()
    feature_data = scaler.fit_transform(feature_data)
    
    return feature_data, available_features, scaler

def create_sequences(data: np.ndarray, sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for transformer training"""
    
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        # Target: next period return
        if i < len(data) - 1:
            current_close = data[i, 3]  # Assuming close is at index 3
            next_close = data[i+1, 3]
            y.append((next_close - current_close) / current_close)
        else:
            y.append(0)
    
    return np.array(X), np.array(y)

class TransformerDRLTrainer:
    """Enhanced DRL trainer with transformer architecture"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.transformer = None
        self.scaler = None
        
    def train_transformer_supervised(
        self, 
        train_data: np.ndarray, 
        val_data: np.ndarray,
        feature_dim: int
    ):
        """Pre-train transformer with supervised learning"""
        
        print("🤖 Pre-training transformer with supervised learning...")
        
        # Create sequences
        X_train, y_train = create_sequences(train_data, self.config['model_params']['sequence_length'])
        X_val, y_val = create_sequences(val_data, self.config['model_params']['sequence_length'])
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create transformer model
        self.transformer = CryptoTransformerNetwork(
            input_dim=feature_dim,
            **self.config['model_params']
        )
        
        # Setup training
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.transformer.to(device)
        
        optimizer = optim.AdamW(
            self.transformer.parameters(),
            lr=self.config['training_params']['learning_rate'],
            weight_decay=self.config['training_params']['weight_decay']
        )
        
        criterion = nn.MSELoss()
        
        # Training loop
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training_params']['batch_size'],
            shuffle=True
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['training_params']['n_epochs']):
            # Training
            self.transformer.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.transformer(batch_X)
                loss = criterion(outputs['action'], batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.transformer.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = self.transformer(batch_X)
                    loss = criterion(outputs['action'], batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        print("✅ Transformer pre-training completed!")
        
        # Save pretrained transformer
        torch.save(self.transformer.state_dict(), "pretrained_transformer.pth")
        
        return train_losses, val_losses
    
    def train_reinforcement_learning(self, env, total_timesteps: int = 100000):
        """Train with reinforcement learning using pretrained transformer"""
        
        print("🎯 Starting reinforcement learning training...")
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: env])
        
        # Initialize PPO with custom policy
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./logs/"
        )
        
        # Setup callback
        callback = TransformerTrainingCallback(
            save_freq=10000,
            save_path="./models/"
        )
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        
        self.model = model
        return model

def run_transformer_crypto_training():
    """Main training pipeline"""
    
    print("🚀 Starting Transformer-based Crypto Trading Training")
    print("=" * 60)
    
    # Configuration
    config = create_transformer_model_config()
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Step 1: Download data
    crypto_data = download_crypto_data(
        symbols=["BTC-USD", "ETH-USD"],
        start_date="2022-01-01",
        end_date="2024-12-31",
        interval="1h"
    )
    
    # Step 2: Calculate technical indicators
    enhanced_data = calculate_enhanced_technical_indicators(crypto_data)
    
    # Step 3: Prepare transformer features
    feature_data, feature_names, scaler = prepare_transformer_features(enhanced_data)
    
    print(f"📊 Feature matrix shape: {feature_data.shape}")
    print(f"📋 Features: {len(feature_names)}")
    
    # Step 4: Split data
    split_index = int(0.8 * len(feature_data))
    train_data = feature_data[:split_index]
    val_data = feature_data[split_index:]
    
    # Step 5: Initialize trainer
    trainer = TransformerDRLTrainer(config)
    
    # Step 6: Pre-train transformer
    train_losses, val_losses = trainer.train_transformer_supervised(
        train_data, val_data, len(feature_names)
    )
    
    # Step 7: Create trading environment
    # For RL training, we need to prepare data in FinRL format
    finrl_data = enhanced_data.copy()
    finrl_data['tic'] = finrl_data['symbol']
    finrl_data = finrl_data.rename(columns={'symbol': 'tic'})
    
    # Create enhanced trading environment
    trading_env = EnhancedCryptoTradingEnv(
        df=finrl_data,
        **config['environment_params']
    )
    
    # Step 8: Train with reinforcement learning
    rl_model = trainer.train_reinforcement_learning(
        trading_env, 
        total_timesteps=200000
    )
    
    # Step 9: Save everything
    results = {
        'config': config,
        'feature_names': feature_names,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'data_shape': feature_data.shape,
        'model_path': './models/transformer_crypto_final.zip'
    }
    
    # Save model and results
    rl_model.save("./models/transformer_crypto_final")
    
    with open("./results/training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save scaler
    with open("./results/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Transformer Pre-training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("🎉 Training completed successfully!")
    print(f"📁 Model saved: ./models/transformer_crypto_final.zip")
    print(f"📊 Results saved: ./results/training_results.json")
    
    return trainer, rl_model, results

if __name__ == "__main__":
    # Run the training
    trainer, model, results = run_transformer_crypto_training()
    
    print("\n🏆 Transformer Crypto Trading Model Training Complete!")
    print("=" * 60)
    print(f"📈 Features: {len(results['feature_names'])}")
    print(f"📊 Data points: {results['data_shape'][0]:,}")
    print(f"🤖 Model saved: {results['model_path']}")