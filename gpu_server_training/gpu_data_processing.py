"""
GPU-Optimized Data Processing for Cryptocurrency Trading
Complete data pipeline for production training on high-end GPU servers
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import requests
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import ta
import talib
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

warnings.filterwarnings('ignore')

class ProductionDataProcessor:
    """
    Production-grade data processor with parallel processing and advanced features
    """
    
    def __init__(
        self,
        assets: List[str] = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'LINK', 'MATIC'],
        timeframe: str = '5m',
        lookback_days: int = 730,  # 2 years
        n_jobs: int = -1
    ):
        self.assets = assets
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        
        # Initialize scalers
        self.price_scaler = RobustScaler()
        self.volume_scaler = RobustScaler()
        self.indicator_scaler = StandardScaler()
        
        print(f"Initialized ProductionDataProcessor for {len(assets)} assets")
        print(f"Timeframe: {timeframe}, Lookback: {lookback_days} days")
        print(f"Using {self.n_jobs} parallel workers")
    
    def download_binance_data(self, symbol: str) -> pd.DataFrame:
        """Download high-quality data from Binance API"""
        try:
            exchange = ccxt.binance({
                'rateLimit': 1200,
                'enableRateLimit': True,
            })
            
            # Calculate timeframe in milliseconds
            timeframe_ms = {
                '1m': 60000,
                '5m': 300000,
                '15m': 900000,
                '1h': 3600000,
                '4h': 14400000,
                '1d': 86400000
            }
            
            ms_per_candle = timeframe_ms.get(self.timeframe, 300000)
            since = exchange.milliseconds() - (self.lookback_days * 24 * 60 * 60 * 1000)
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(
                f'{symbol}/USDT', 
                timeframe=self.timeframe, 
                since=since,
                limit=1000
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add symbol prefix to columns
            df.columns = [f'{symbol}_{col}' for col in df.columns]
            
            print(f"Downloaded {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error downloading {symbol} data: {e}")
            return pd.DataFrame()
    
    def download_yfinance_data(self, symbol: str) -> pd.DataFrame:
        """Fallback: Download data from Yahoo Finance"""
        try:
            ticker_map = {
                'BTC': 'BTC-USD',
                'ETH': 'ETH-USD',
                'BNB': 'BNB-USD',
                'ADA': 'ADA-USD',
                'SOL': 'SOL-USD',
                'DOT': 'DOT-USD',
                'LINK': 'LINK-USD',
                'MATIC': 'MATIC-USD'
            }
            
            yahoo_symbol = ticker_map.get(symbol, f'{symbol}-USD')
            
            # Download with appropriate interval
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '1h': '1h',
                '4h': '1h',  # Yahoo doesn't have 4h, use 1h
                '1d': '1d'
            }
            
            interval = interval_map.get(self.timeframe, '5m')
            period = f"{min(self.lookback_days, 60)}d"  # Yahoo limit
            
            df = yf.download(
                yahoo_symbol,
                period=period,
                interval=interval,
                progress=False
            )
            
            if df.empty:
                return df
            
            # Rename columns
            df.columns = [f'{symbol}_{col.lower()}' for col in df.columns]
            df.reset_index(inplace=True)
            df.rename(columns={'Date': 'timestamp', 'Datetime': 'timestamp'}, inplace=True)
            df.set_index('timestamp', inplace=True)
            
            print(f"Downloaded {len(df)} records for {symbol} (Yahoo Finance)")
            return df
            
        except Exception as e:
            print(f"Error downloading {symbol} from Yahoo Finance: {e}")
            return pd.DataFrame()
    
    def download_single_asset(self, symbol: str) -> pd.DataFrame:
        """Download data for a single asset with fallback"""
        # Try Binance first
        df = self.download_binance_data(symbol)
        
        # Fallback to Yahoo Finance if Binance fails
        if df.empty:
            df = self.download_yfinance_data(symbol)
        
        return df
    
    def download_all_data(self) -> pd.DataFrame:
        """Download data for all assets in parallel"""
        print(f"Starting parallel download for {len(self.assets)} assets...")
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(self.download_single_asset, asset) for asset in self.assets]
            dataframes = [future.result() for future in futures]
        
        # Filter out empty dataframes
        valid_dfs = [df for df in dataframes if not df.empty]
        
        if not valid_dfs:
            raise ValueError("No data downloaded for any asset!")
        
        # Merge all dataframes
        combined_df = valid_dfs[0]
        for df in valid_dfs[1:]:
            combined_df = combined_df.join(df, how='outer')
        
        # Forward fill missing values
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Combined dataset shape: {combined_df.shape}")
        return combined_df.reset_index()
    
    def calculate_technical_indicators(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for a single asset"""
        
        # Price columns
        open_col = f'{asset}_open'
        high_col = f'{asset}_high'
        low_col = f'{asset}_low'
        close_col = f'{asset}_close'
        volume_col = f'{asset}_volume'
        
        if not all(col in df.columns for col in [open_col, high_col, low_col, close_col]):
            return pd.DataFrame()
        
        result_df = pd.DataFrame(index=df.index)
        
        try:
            # Extract price data
            open_prices = df[open_col].values
            high_prices = df[high_col].values
            low_prices = df[low_col].values
            close_prices = df[close_col].values
            volumes = df[volume_col].values if volume_col in df.columns else np.ones_like(close_prices)
            
            # Trend Indicators
            result_df[f'{asset}_sma_5'] = talib.SMA(close_prices, timeperiod=5)
            result_df[f'{asset}_sma_10'] = talib.SMA(close_prices, timeperiod=10)
            result_df[f'{asset}_sma_20'] = talib.SMA(close_prices, timeperiod=20)
            result_df[f'{asset}_sma_50'] = talib.SMA(close_prices, timeperiod=50)
            result_df[f'{asset}_sma_100'] = talib.SMA(close_prices, timeperiod=100)
            
            result_df[f'{asset}_ema_5'] = talib.EMA(close_prices, timeperiod=5)
            result_df[f'{asset}_ema_10'] = talib.EMA(close_prices, timeperiod=10)
            result_df[f'{asset}_ema_20'] = talib.EMA(close_prices, timeperiod=20)
            result_df[f'{asset}_ema_50'] = talib.EMA(close_prices, timeperiod=50)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            result_df[f'{asset}_macd'] = macd
            result_df[f'{asset}_macd_signal'] = macd_signal
            result_df[f'{asset}_macd_hist'] = macd_hist
            
            # Momentum Indicators
            result_df[f'{asset}_rsi_14'] = talib.RSI(close_prices, timeperiod=14)
            result_df[f'{asset}_rsi_21'] = talib.RSI(close_prices, timeperiod=21)
            result_df[f'{asset}_stoch_k'], result_df[f'{asset}_stoch_d'] = talib.STOCH(high_prices, low_prices, close_prices)
            result_df[f'{asset}_williams_r'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
            result_df[f'{asset}_cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
            result_df[f'{asset}_mfi'] = talib.MFI(high_prices, low_prices, close_prices, volumes, timeperiod=14)
            
            # Volatility Indicators
            result_df[f'{asset}_atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            result_df[f'{asset}_natr'] = talib.NATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
            result_df[f'{asset}_bb_upper'] = bb_upper
            result_df[f'{asset}_bb_middle'] = bb_middle
            result_df[f'{asset}_bb_lower'] = bb_lower
            result_df[f'{asset}_bb_width'] = (bb_upper - bb_lower) / bb_middle
            result_df[f'{asset}_bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
            
            # Volume Indicators
            result_df[f'{asset}_obv'] = talib.OBV(close_prices, volumes)
            result_df[f'{asset}_ad'] = talib.AD(high_prices, low_prices, close_prices, volumes)
            result_df[f'{asset}_adosc'] = talib.ADOSC(high_prices, low_prices, close_prices, volumes)
            
            # Pattern Recognition (top patterns)
            result_df[f'{asset}_doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            result_df[f'{asset}_hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            result_df[f'{asset}_engulfing'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            result_df[f'{asset}_morning_star'] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
            result_df[f'{asset}_evening_star'] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
            
            # Price Action Features
            result_df[f'{asset}_price_change'] = close_prices / np.roll(close_prices, 1) - 1
            result_df[f'{asset}_price_change_5'] = close_prices / np.roll(close_prices, 5) - 1
            result_df[f'{asset}_price_change_20'] = close_prices / np.roll(close_prices, 20) - 1
            
            result_df[f'{asset}_high_low_ratio'] = high_prices / low_prices
            result_df[f'{asset}_close_open_ratio'] = close_prices / open_prices
            
            # Volatility measures
            result_df[f'{asset}_volatility_5'] = pd.Series(close_prices).pct_change().rolling(5).std()
            result_df[f'{asset}_volatility_20'] = pd.Series(close_prices).pct_change().rolling(20).std()
            
            # Volume analysis
            if volume_col in df.columns:
                result_df[f'{asset}_volume_sma_5'] = talib.SMA(volumes.astype(float), timeperiod=5)
                result_df[f'{asset}_volume_ratio'] = volumes / talib.SMA(volumes.astype(float), timeperiod=20)
                result_df[f'{asset}_price_volume'] = close_prices * volumes
            
            # Support and Resistance levels (simplified)
            rolling_max_20 = pd.Series(high_prices).rolling(20).max()
            rolling_min_20 = pd.Series(low_prices).rolling(20).min()
            result_df[f'{asset}_resistance_distance'] = (rolling_max_20 - close_prices) / close_prices
            result_df[f'{asset}_support_distance'] = (close_prices - rolling_min_20) / close_prices
            
            print(f"Calculated {len(result_df.columns)} indicators for {asset}")
            
        except Exception as e:
            print(f"Error calculating indicators for {asset}: {e}")
            return pd.DataFrame()
        
        return result_df
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for all assets in parallel"""
        print("Calculating technical indicators for all assets...")
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(self.calculate_technical_indicators, df, asset) 
                for asset in self.assets
            ]
            indicator_dfs = [future.result() for future in futures]
        
        # Filter out empty dataframes
        valid_indicator_dfs = [df_ind for df_ind in indicator_dfs if not df_ind.empty]
        
        if not valid_indicator_dfs:
            print("No indicators calculated!")
            return df
        
        # Combine with original data
        combined_df = df.set_index('timestamp') if 'timestamp' in df.columns else df
        for indicator_df in valid_indicator_dfs:
            combined_df = combined_df.join(indicator_df, how='left')
        
        return combined_df.reset_index()
    
    def add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market-wide features and cross-asset correlations"""
        print("Adding market-wide features...")
        
        # Market cap weighted index (simplified)
        close_cols = [col for col in df.columns if col.endswith('_close')]
        if len(close_cols) > 1:
            # Simple equal-weight index
            df['market_index'] = df[close_cols].mean(axis=1)
            df['market_index_change'] = df['market_index'].pct_change()
            df['market_volatility'] = df['market_index_change'].rolling(20).std()
            
            # Market momentum
            df['market_momentum_5'] = df['market_index'].pct_change(5)
            df['market_momentum_20'] = df['market_index'].pct_change(20)
        
        # Time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Cross-asset correlations (rolling)
        if len(close_cols) > 1:
            for i, col1 in enumerate(close_cols[:3]):  # Limit to avoid too many features
                for col2 in close_cols[i+1:4]:
                    asset1 = col1.split('_')[0]
                    asset2 = col2.split('_')[0]
                    corr_col = f'corr_{asset1}_{asset2}'
                    df[corr_col] = df[col1].pct_change().rolling(50).corr(df[col2].pct_change())
        
        print(f"Added market features. Total columns: {len(df.columns)}")
        return df
    
    def clean_and_normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize the dataset"""
        print("Cleaning and normalizing data...")
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows with too many missing values
        missing_threshold = 0.3 * len(df.columns)
        df = df.dropna(thresh=len(df.columns) - missing_threshold)
        
        # Identify price, volume, and indicator columns
        price_cols = [col for col in df.columns if any(x in col for x in ['_open', '_high', '_low', '_close'])]
        volume_cols = [col for col in df.columns if '_volume' in col]
        indicator_cols = [col for col in df.columns if col not in price_cols + volume_cols + ['timestamp']]
        
        # Apply different scaling strategies
        if price_cols:
            df[price_cols] = self.price_scaler.fit_transform(df[price_cols])
        
        if volume_cols:
            df[volume_cols] = self.volume_scaler.fit_transform(df[volume_cols])
        
        if indicator_cols:
            df[indicator_cols] = self.indicator_scaler.fit_transform(df[indicator_cols])
        
        # Remove extreme outliers (beyond 5 standard deviations)
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'timestamp':
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[col] = np.clip(df[col], mean_val - 5*std_val, mean_val + 5*std_val)
        
        print(f"Cleaned dataset shape: {df.shape}")
        return df
    
    def save_processing_artifacts(self, save_dir: str):
        """Save scalers and processing artifacts"""
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.price_scaler, os.path.join(save_dir, 'price_scaler.pkl'))
        joblib.dump(self.volume_scaler, os.path.join(save_dir, 'volume_scaler.pkl'))
        joblib.dump(self.indicator_scaler, os.path.join(save_dir, 'indicator_scaler.pkl'))
        
        # Save asset list and configuration
        config = {
            'assets': self.assets,
            'timeframe': self.timeframe,
            'lookback_days': self.lookback_days,
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(os.path.join(save_dir, 'processing_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved processing artifacts to {save_dir}")
    
    def process_complete_dataset(self, save_path: str = None) -> pd.DataFrame:
        """Complete data processing pipeline"""
        print("Starting complete data processing pipeline...")
        
        # Step 1: Download raw data
        df = self.download_all_data()
        
        # Step 2: Calculate technical indicators
        df = self.calculate_all_indicators(df)
        
        # Step 3: Add market features
        df = self.add_market_features(df)
        
        # Step 4: Clean and normalize
        df = self.clean_and_normalize_data(df)
        
        # Step 5: Final validation
        df = df.dropna()  # Remove any remaining NaN values
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Saved processed dataset to {save_path}")
            
            # Save processing artifacts
            save_dir = os.path.dirname(save_path)
            self.save_processing_artifacts(save_dir)
        
        print(f"Processing complete! Final dataset shape: {df.shape}")
        print(f"Features: {len(df.columns)} columns")
        print(f"Samples: {len(df)} rows")
        
        return df

def create_synthetic_data(base_df: pd.DataFrame, n_synthetic: int = 100000) -> pd.DataFrame:
    """Create synthetic data using statistical properties of real data"""
    print(f"Creating {n_synthetic} synthetic samples...")
    
    # Calculate statistical properties
    numeric_cols = base_df.select_dtypes(include=[np.number]).columns
    
    # Use the last 10000 samples for better statistical representation
    recent_data = base_df[numeric_cols].tail(10000)
    
    # Calculate correlations and statistics
    correlations = recent_data.corr()
    means = recent_data.mean()
    stds = recent_data.std()
    
    # Generate synthetic data preserving correlations
    np.random.seed(42)  # For reproducibility
    
    # Use multivariate normal distribution
    synthetic_data = np.random.multivariate_normal(
        mean=means.values,
        cov=correlations.values * np.outer(stds.values, stds.values),
        size=n_synthetic
    )
    
    # Create synthetic dataframe
    synthetic_df = pd.DataFrame(synthetic_data, columns=numeric_cols)
    
    # Add timestamp column (synthetic timestamps)
    start_time = pd.to_datetime(base_df['timestamp'].max()) + timedelta(minutes=5)
    synthetic_df['timestamp'] = pd.date_range(start=start_time, periods=n_synthetic, freq='5T')
    
    print(f"Created synthetic dataset with shape: {synthetic_df.shape}")
    return synthetic_df

if __name__ == "__main__":
    # Production data processing
    processor = ProductionDataProcessor(
        assets=['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'LINK', 'MATIC'],
        timeframe='5m',
        lookback_days=730
    )
    
    # Process complete dataset
    save_path = 'gpu_server_training/crypto_production_dataset.csv'
    dataset = processor.process_complete_dataset(save_path)
    
    # Create additional synthetic data for training
    synthetic_data = create_synthetic_data(dataset, n_synthetic=50000)
    synthetic_path = 'gpu_server_training/crypto_synthetic_dataset.csv'
    synthetic_data.to_csv(synthetic_path, index=False)
    
    print("\n" + "="*60)
    print("DATA PROCESSING COMPLETE!")
    print("="*60)
    print(f"Real data: {dataset.shape[0]} samples, {dataset.shape[1]} features")
    print(f"Synthetic data: {synthetic_data.shape[0]} samples")
    print(f"Total training data available: {dataset.shape[0] + synthetic_data.shape[0]} samples")
    print("="*60)