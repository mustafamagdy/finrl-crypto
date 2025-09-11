"""
Transformer-based Cryptocurrency Trading Training Script - FIXED VERSION
Advanced FinRL implementation with transformer architecture and validation split
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import transformer model
from transformer_crypto_model import (
    CryptoTransformerNetwork, 
    EnhancedCryptoTradingEnv,
    create_transformer_model_config
)

def download_crypto_data_fixed(
    symbols: List[str] = ["BTC-USD"],
    days_back: int = 600,  # Within 730-day limit
    interval: str = "1h"
) -> pd.DataFrame:
    """Download cryptocurrency data with proper date handling"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"📊 Downloading crypto data for {symbols}...")
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    all_data = []
    
    for symbol in symbols:
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'), 
                end=end_date.strftime('%Y-%m-%d'), 
                interval=interval
            )
            
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

def prepare_transformer_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], StandardScaler]:
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

def split_data_chronologically(data: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split data chronologically for proper time series validation"""
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"📊 Data split:")
    print(f"  Training: {len(train_data):,} samples ({train_ratio*100:.1f}%)")
    print(f"  Validation: {len(val_data):,} samples ({val_ratio*100:.1f}%)")
    print(f"  Test: {len(test_data):,} samples ({(1-train_ratio-val_ratio)*100:.1f}%)")
    
    return train_data, val_data, test_data

class TransformerDRLTrainer:
    """Enhanced DRL trainer with transformer architecture and validation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.transformer = None
        self.scaler = None
        self.validation_results = {}
        
    def train_transformer_supervised(
        self, 
        train_data: np.ndarray, 
        val_data: np.ndarray,
        test_data: np.ndarray,
        feature_dim: int
    ):
        """Pre-train transformer with supervised learning and validation"""
        
        print("🤖 Pre-training transformer with supervised learning...")
        
        # Create sequences for all splits
        X_train, y_train = create_sequences(train_data, self.config['model_params']['max_seq_len'])
        X_val, y_val = create_sequences(val_data, self.config['model_params']['max_seq_len'])
        X_test, y_test = create_sequences(test_data, self.config['model_params']['max_seq_len'])
        
        print(f"📊 Sequence shapes:")
        print(f"  Train: X={X_train.shape}, y={y_train.shape}")
        print(f"  Validation: X={X_val.shape}, y={y_val.shape}")
        print(f"  Test: X={X_test.shape}, y={y_test.shape}")
        
        # Convert to tensors
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"🔧 Using device: {device}")
        
        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_val = torch.FloatTensor(X_val).to(device)
        y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_test = torch.FloatTensor(y_test).unsqueeze(1).to(device)
        
        # Create transformer model
        self.transformer = CryptoTransformerNetwork(
            input_dim=feature_dim,
            **self.config['model_params']
        ).to(device)
        
        print(f"🔧 Transformer parameters: {sum(p.numel() for p in self.transformer.parameters()):,}")
        
        # Setup training
        optimizer = optim.AdamW(
            self.transformer.parameters(),
            lr=self.config['training_params']['learning_rate'],
            weight_decay=self.config['training_params']['weight_decay']
        )
        
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
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
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print("🚀 Starting transformer training...")
        
        for epoch in range(self.config['training_params']['n_epochs']):
            # Training
            self.transformer.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.transformer(batch_X)
                loss = criterion(outputs['action'], batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.transformer.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.transformer(batch_X)
                    loss = criterion(outputs['action'], batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.transformer.state_dict(), "best_transformer.pth")
            else:
                patience_counter += 1
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.transformer.load_state_dict(torch.load("best_transformer.pth"))
        
        # Final validation on test set
        print("📊 Evaluating on test set...")
        self.transformer.eval()
        
        with torch.no_grad():
            test_outputs = self.transformer(X_test)
            test_predictions = test_outputs['action'].cpu().numpy().flatten()
            test_targets = y_test.cpu().numpy().flatten()
            
            # Calculate metrics
            mse = mean_squared_error(test_targets, test_predictions)
            mae = mean_absolute_error(test_targets, test_predictions)
            r2 = r2_score(test_targets, test_predictions)
            
            # Directional accuracy
            direction_correct = np.sum((test_predictions > 0) == (test_targets > 0)) / len(test_targets)
            
            self.validation_results = {
                'test_mse': mse,
                'test_mae': mae,
                'test_r2': r2,
                'direction_accuracy': direction_correct,
                'best_val_loss': best_val_loss,
                'total_epochs': epoch + 1
            }
            
            print(f"🏆 Test Results:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  R²: {r2:.6f}")
            print(f"  Direction Accuracy: {direction_correct:.3f}")
        
        print("✅ Transformer pre-training completed!")
        
        return train_losses, val_losses
        
def run_transformer_crypto_training():
    """Main training pipeline with validation"""
    
    print("🚀 Starting Transformer-based Crypto Trading Training")
    print("=" * 60)
    
    # Configuration
    config = create_transformer_model_config()
    config['training_params']['n_epochs'] = 50  # Reduce for testing
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Step 1: Download data (with valid timeframe)
    crypto_data = download_crypto_data_fixed(
        symbols=["BTC-USD"],  # Just BTC for faster testing
        days_back=600,  # Within limit
        interval="1h"
    )
    
    # Step 2: Calculate technical indicators
    enhanced_data = calculate_enhanced_technical_indicators(crypto_data)
    
    # Step 3: Prepare transformer features
    feature_data, feature_names, scaler = prepare_transformer_features(enhanced_data)
    
    print(f"📊 Feature matrix shape: {feature_data.shape}")
    print(f"📋 Features: {len(feature_names)}")
    
    # Step 4: Split data chronologically
    train_data, val_data, test_data = split_data_chronologically(feature_data)
    
    # Step 5: Initialize trainer
    trainer = TransformerDRLTrainer(config)
    
    # Step 6: Pre-train transformer
    train_losses, val_losses = trainer.train_transformer_supervised(
        train_data, val_data, test_data, len(feature_names)
    )
    
    # Step 7: Save everything
    results = {
        'config': config,
        'feature_names': feature_names,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'validation_results': trainer.validation_results,
        'data_shape': feature_data.shape,
        'model_path': './models/transformer_crypto_pretrained.pth'
    }
    
    # Save model and results
    torch.save(trainer.transformer.state_dict(), "./models/transformer_crypto_pretrained.pth")
    
    with open("./results/training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save scaler
    with open("./results/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.bar(['MSE', 'MAE', 'R²', 'Direction Acc'], [
        trainer.validation_results['test_mse'],
        trainer.validation_results['test_mae'], 
        trainer.validation_results['test_r2'],
        trainer.validation_results['direction_accuracy']
    ])
    plt.title('Test Set Performance')
    plt.ylabel('Score')
    
    plt.subplot(1, 3, 3)
    epochs = list(range(len(train_losses)))
    plt.plot(epochs, train_losses, label='Train', alpha=0.7)
    plt.plot(epochs, val_losses, label='Validation', alpha=0.7)
    plt.axvline(x=trainer.validation_results['total_epochs']-10, color='red', linestyle='--', alpha=0.5, label='Early Stop')
    plt.title('Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("🎉 Training completed successfully!")
    print(f"📁 Model saved: ./models/transformer_crypto_pretrained.pth")
    print(f"📊 Results saved: ./results/training_results.json")
    
    return trainer, results

if __name__ == "__main__":
    # Run the training
    trainer, results = run_transformer_crypto_training()
    
    print("\n🏆 Transformer Crypto Trading Model Training Complete!")
    print("=" * 60)
    print(f"📈 Features: {len(results['feature_names'])}")
    print(f"📊 Data points: {results['data_shape'][0]:,}")
    print(f"🎯 Test R²: {results['validation_results']['test_r2']:.4f}")
    print(f"🎯 Direction Accuracy: {results['validation_results']['direction_accuracy']:.3f}")
    print(f"🤖 Model saved: {results['model_path']}")