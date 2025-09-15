"""
Transformer-based Cryptocurrency Trading Training Script - 5 MINUTE DATA
Using pre-downloaded 5-minute data with full validation split
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

def load_5min_crypto_data(csv_path: str = "crypto_5min_2years.csv") -> pd.DataFrame:
    """Load the pre-downloaded 5-minute crypto data"""
    
    print(f"📊 Loading 5-minute crypto data from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} records")
        
        # Basic data info
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])
        else:
            print("⚠️ No timestamp column found, using index")
            df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='5min')
        
        # Ensure we have OHLCV columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
            # Try alternative column names
            column_mapping = {
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
                'price': 'close', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
                    print(f"  Mapped {old_col} -> {new_col}")
        
        # Add symbol if not present
        if 'symbol' not in df.columns:
            df['symbol'] = 'btc'  # Default to BTC
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"📋 Columns: {list(df.columns)}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ File {csv_path} not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def calculate_enhanced_technical_indicators_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate enhanced technical indicators optimized for 5-minute data"""
    
    print("🔧 Calculating enhanced technical indicators for 5-minute data...")
    df = df.copy()
    
    # Group by symbol for indicator calculation
    enhanced_df = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        print(f"  Processing {symbol}: {len(symbol_df)} records")
        
        # Price-based indicators (shorter periods for 5min data)
        symbol_df['sma_5'] = ta.trend.sma_indicator(symbol_df['close'], window=5)    # 25min
        symbol_df['sma_12'] = ta.trend.sma_indicator(symbol_df['close'], window=12)  # 1hour  
        symbol_df['sma_24'] = ta.trend.sma_indicator(symbol_df['close'], window=24)  # 2hours
        symbol_df['sma_48'] = ta.trend.sma_indicator(symbol_df['close'], window=48)  # 4hours
        symbol_df['sma_144'] = ta.trend.sma_indicator(symbol_df['close'], window=144) # 12hours
        
        symbol_df['ema_5'] = ta.trend.ema_indicator(symbol_df['close'], window=5)
        symbol_df['ema_12'] = ta.trend.ema_indicator(symbol_df['close'], window=12)
        symbol_df['ema_24'] = ta.trend.ema_indicator(symbol_df['close'], window=24)
        
        # MACD (optimized for 5min)
        macd = ta.trend.MACD(close=symbol_df['close'], window_slow=26, window_fast=12, window_sign=9)
        symbol_df['macd'] = macd.macd()
        symbol_df['macd_signal'] = macd.macd_signal()
        symbol_df['macd_histogram'] = macd.macd_diff()
        
        # RSI (multiple timeframes)
        symbol_df['rsi_14'] = ta.momentum.rsi(symbol_df['close'], window=14)  # 70min
        symbol_df['rsi_24'] = ta.momentum.rsi(symbol_df['close'], window=24)  # 2hours
        symbol_df['rsi_48'] = ta.momentum.rsi(symbol_df['close'], window=48)  # 4hours
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=symbol_df['close'], window=20, window_dev=2)
        symbol_df['bb_high'] = bollinger.bollinger_hband()
        symbol_df['bb_low'] = bollinger.bollinger_lband()
        symbol_df['bb_mid'] = bollinger.bollinger_mavg()
        symbol_df['bb_width'] = (symbol_df['bb_high'] - symbol_df['bb_low']) / symbol_df['bb_mid']
        symbol_df['bb_position'] = (symbol_df['close'] - symbol_df['bb_low']) / (symbol_df['bb_high'] - symbol_df['bb_low'])
        
        # Volatility indicators  
        symbol_df['atr_14'] = ta.volatility.average_true_range(symbol_df['high'], symbol_df['low'], symbol_df['close'], window=14)
        symbol_df['volatility_12'] = symbol_df['close'].pct_change().rolling(12).std()  # 1hour
        symbol_df['volatility_24'] = symbol_df['close'].pct_change().rolling(24).std()  # 2hours
        symbol_df['volatility_48'] = symbol_df['close'].pct_change().rolling(48).std()  # 4hours
        
        # Volume indicators
        if 'volume' in symbol_df.columns and symbol_df['volume'].sum() > 0:
            symbol_df['volume_sma'] = symbol_df['volume'].rolling(window=20).mean()
            symbol_df['volume_ratio'] = symbol_df['volume'] / symbol_df['volume_sma']
            # Volume oscillator
            symbol_df['volume_osc'] = (symbol_df['volume'] - symbol_df['volume_sma']) / symbol_df['volume_sma']
        else:
            symbol_df['volume_sma'] = 0
            symbol_df['volume_ratio'] = 1
            symbol_df['volume_osc'] = 0
        
        # Momentum indicators
        symbol_df['stoch_k'] = ta.momentum.stoch(symbol_df['high'], symbol_df['low'], symbol_df['close'], window=14)
        symbol_df['stoch_d'] = ta.momentum.stoch_signal(symbol_df['high'], symbol_df['low'], symbol_df['close'], window=14)
        symbol_df['williams_r'] = ta.momentum.williams_r(symbol_df['high'], symbol_df['low'], symbol_df['close'])
        
        # Trend indicators
        symbol_df['adx'] = ta.trend.adx(symbol_df['high'], symbol_df['low'], symbol_df['close'])
        symbol_df['cci'] = ta.trend.cci(symbol_df['high'], symbol_df['low'], symbol_df['close'])
        
        # Price action features (multiple timeframes)
        symbol_df['returns_1'] = symbol_df['close'].pct_change()                      # 5min returns
        symbol_df['returns_12'] = symbol_df['close'].pct_change(periods=12)          # 1hour returns
        symbol_df['returns_24'] = symbol_df['close'].pct_change(periods=24)          # 2hour returns
        symbol_df['returns_48'] = symbol_df['close'].pct_change(periods=48)          # 4hour returns
        
        symbol_df['log_returns'] = np.log(symbol_df['close'] / symbol_df['close'].shift(1))
        
        # Momentum features
        symbol_df['price_momentum_12'] = symbol_df['close'] / symbol_df['close'].shift(12) - 1    # 1hour
        symbol_df['price_momentum_24'] = symbol_df['close'] / symbol_df['close'].shift(24) - 1    # 2hour
        symbol_df['price_momentum_48'] = symbol_df['close'] / symbol_df['close'].shift(48) - 1    # 4hour
        
        # Market regime features
        symbol_df['trend_strength'] = abs(symbol_df['close'] - symbol_df['sma_48']) / symbol_df['sma_48']
        symbol_df['market_regime'] = np.where(
            symbol_df['close'] > symbol_df['sma_48'], 1,  # Bull
            np.where(symbol_df['close'] < symbol_df['sma_48'], -1, 0)  # Bear, Sideways
        )
        
        # High-frequency features (5min specific)
        symbol_df['price_range'] = (symbol_df['high'] - symbol_df['low']) / symbol_df['open']
        symbol_df['body_size'] = abs(symbol_df['close'] - symbol_df['open']) / symbol_df['open'] 
        symbol_df['upper_shadow'] = (symbol_df['high'] - np.maximum(symbol_df['open'], symbol_df['close'])) / symbol_df['open']
        symbol_df['lower_shadow'] = (np.minimum(symbol_df['open'], symbol_df['close']) - symbol_df['low']) / symbol_df['open']
        
        # Time-based features (5min resolution)
        symbol_df['minute'] = pd.to_datetime(symbol_df['date']).dt.minute
        symbol_df['hour'] = pd.to_datetime(symbol_df['date']).dt.hour
        symbol_df['day_of_week'] = pd.to_datetime(symbol_df['date']).dt.dayofweek
        
        # Cyclical encoding
        symbol_df['minute_sin'] = np.sin(2 * np.pi * symbol_df['minute'] / 60)
        symbol_df['minute_cos'] = np.cos(2 * np.pi * symbol_df['minute'] / 60)
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

def prepare_transformer_features_5min(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """Prepare features specifically for 5-minute transformer model"""
    
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'sma_5', 'sma_12', 'sma_24', 'sma_48', 'sma_144',
        'ema_5', 'ema_12', 'ema_24',
        'macd', 'macd_signal', 'macd_histogram',
        'rsi_14', 'rsi_24', 'rsi_48',
        'bb_high', 'bb_low', 'bb_mid', 'bb_width', 'bb_position',
        'atr_14', 'volatility_12', 'volatility_24', 'volatility_48',
        'volume_ratio', 'volume_osc', 'stoch_k', 'stoch_d', 'williams_r',
        'adx', 'cci',
        'returns_1', 'returns_12', 'returns_24', 'returns_48', 'log_returns',
        'price_momentum_12', 'price_momentum_24', 'price_momentum_48',
        'trend_strength', 'market_regime',
        'price_range', 'body_size', 'upper_shadow', 'lower_shadow',
        'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]
    
    # Select available features
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"📊 Selected {len(available_features)} features for training")
    
    # Create feature matrix
    feature_data = df[available_features].values
    
    # Handle infinite values
    feature_data = np.nan_to_num(feature_data, nan=0, posinf=0, neginf=0)
    
    # Normalize features
    scaler = StandardScaler()
    feature_data = scaler.fit_transform(feature_data)
    
    print(f"✅ Feature preparation complete: {feature_data.shape}")
    return feature_data, available_features, scaler

def create_sequences_5min(data: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for transformer training (60 = 5 hours of 5min data)"""
    
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        # Target: next 5-minute return
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

class TransformerDRLTrainer5Min:
    """Enhanced DRL trainer for 5-minute data with transformer architecture"""
    
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
        
        print("🤖 Pre-training transformer with supervised learning (5-minute data)...")
        
        # Create sequences for all splits  
        sequence_length = self.config['model_params']['max_seq_len']
        X_train, y_train = create_sequences_5min(train_data, sequence_length)
        X_val, y_val = create_sequences_5min(val_data, sequence_length)
        X_test, y_test = create_sequences_5min(test_data, sequence_length)
        
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
        patience = 15
        patience_counter = 0
        
        print("🚀 Starting transformer training...")
        
        for epoch in range(self.config['training_params']['n_epochs']):
            # Training
            self.transformer.train()
            train_loss = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
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
                torch.save(self.transformer.state_dict(), "best_transformer_5min.pth")
            else:
                patience_counter += 1
                
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.transformer.load_state_dict(torch.load("best_transformer_5min.pth"))
        
        # Final validation on test set (memory-optimized batching)
        print("📊 Evaluating on test set...")
        self.transformer.eval()
        
        test_predictions = []
        test_targets_list = []
        batch_size = 1000  # Smaller batches for memory efficiency
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_end = min(i + batch_size, len(X_test))
                X_batch = X_test[i:batch_end]
                y_batch = y_test[i:batch_end]
                
                batch_outputs = self.transformer(X_batch)
                test_predictions.extend(batch_outputs['action'].cpu().numpy().flatten())
                test_targets_list.extend(y_batch.cpu().numpy().flatten())
                
                # Clear cache periodically
                if i % (batch_size * 10) == 0:
                    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
                    print(f"  Processed {i + len(X_batch)}/{len(X_test)} test samples...")
        
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets_list)
        
        # Calculate metrics
        mse = mean_squared_error(test_targets, test_predictions)
        mae = mean_absolute_error(test_targets, test_predictions)
        r2 = r2_score(test_targets, test_predictions)
        
        # Directional accuracy
        direction_correct = np.sum((test_predictions > 0) == (test_targets > 0)) / len(test_targets)
        
        # Volatility metrics
        pred_vol = np.std(test_predictions)
        actual_vol = np.std(test_targets)
        
        self.validation_results = {
                'test_mse': mse,
                'test_mae': mae, 
                'test_r2': r2,
                'direction_accuracy': direction_correct,
                'predicted_volatility': pred_vol,
                'actual_volatility': actual_vol,
                'best_val_loss': best_val_loss,
                'total_epochs': epoch + 1
        }
        
        print(f"🏆 Test Results:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²: {r2:.6f}")
        print(f"  Direction Accuracy: {direction_correct:.3f}")
        print(f"  Predicted Vol: {pred_vol:.4f}")
        print(f"  Actual Vol: {actual_vol:.4f}")
        
        print("✅ Transformer pre-training completed!")
        
        return train_losses, val_losses
        
def run_transformer_5min_training():
    """Main training pipeline for 5-minute data"""
    
    print("🚀 Starting Transformer-based Crypto Trading Training (5-Minute Data)")
    print("=" * 70)
    
    # Configuration optimized for 5-minute data
    config = create_transformer_model_config()
    config['model_params']['max_seq_len'] = 60  # 5 hours of 5-min data
    config['model_params']['d_model'] = 256
    config['model_params']['n_heads'] = 8
    config['model_params']['n_layers'] = 6
    config['training_params']['n_epochs'] = 100
    config['training_params']['batch_size'] = 32  # Smaller batch for memory
    config['training_params']['learning_rate'] = 1e-4
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Step 1: Load 5-minute data
    crypto_data = load_5min_crypto_data()
    if crypto_data is None:
        return None, None
    
    # Step 2: Calculate technical indicators
    enhanced_data = calculate_enhanced_technical_indicators_5min(crypto_data)
    
    # Step 3: Prepare transformer features
    feature_data, feature_names, scaler = prepare_transformer_features_5min(enhanced_data)
    
    print(f"📊 Feature matrix shape: {feature_data.shape}")
    print(f"📋 Features: {len(feature_names)}")
    
    # Step 4: Split data chronologically
    train_data, val_data, test_data = split_data_chronologically(feature_data)
    
    # Step 5: Initialize trainer
    trainer = TransformerDRLTrainer5Min(config)
    
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
        'model_path': './models/transformer_crypto_5min.pth'
    }
    
    # Save model and results
    torch.save(trainer.transformer.state_dict(), "./models/transformer_crypto_5min.pth")
    
    with open("./results/training_results_5min.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save scaler
    with open("./results/scaler_5min.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Plot training curves
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8, linewidth=2)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8, linewidth=2)
    plt.title('Training Loss (5-Minute Data)', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    metrics = ['MSE', 'MAE', 'R²', 'Direction\nAccuracy']
    values = [
        trainer.validation_results['test_mse'],
        trainer.validation_results['test_mae'], 
        trainer.validation_results['test_r2'],
        trainer.validation_results['direction_accuracy']
    ]
    bars = plt.bar(metrics, values, color=['red', 'orange', 'green', 'blue'], alpha=0.7)
    plt.title('Test Set Performance', fontsize=14)
    plt.ylabel('Score')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.subplot(1, 3, 3)
    epochs = list(range(len(train_losses)))
    plt.plot(epochs, train_losses, label='Train', alpha=0.8, linewidth=2)
    plt.plot(epochs, val_losses, label='Validation', alpha=0.8, linewidth=2)
    plt.axvline(x=trainer.validation_results['total_epochs']-15, color='red', linestyle='--', alpha=0.5, label='Early Stop')
    plt.title('Loss Convergence', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/training_analysis_5min.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("🎉 Training completed successfully!")
    print(f"📁 Model saved: ./models/transformer_crypto_5min.pth")
    print(f"📊 Results saved: ./results/training_results_5min.json")
    
    return trainer, results

if __name__ == "__main__":
    # Run the training
    trainer, results = run_transformer_5min_training()
    
    if results:
        print("\n🏆 Transformer Crypto Trading Model Training Complete!")
        print("=" * 70)
        print(f"📈 Features: {len(results['feature_names'])}")
        print(f"📊 Data points: {results['data_shape'][0]:,}")
        print(f"🎯 Test R²: {results['validation_results']['test_r2']:.4f}")
        print(f"🎯 Direction Accuracy: {results['validation_results']['direction_accuracy']:.3f}")
        print(f"🤖 Model saved: {results['model_path']}")
        print("\n📈 Expected Improvements vs Original PPO:")
        print(f"  Win Rate: 17.56% → {results['validation_results']['direction_accuracy']*100:.1f}%")
        print(f"  Sequence Modeling: ✅ (60 timesteps = 5 hours)")
        print(f"  Multi-Head Attention: ✅ (8 heads)")
        print(f"  Market Regime Detection: ✅")
        print(f"  Enhanced Features: ✅ ({len(results['feature_names'])} features)")
    else:
        print("❌ Training failed - check data file exists")