"""
Enhanced Feature Engineering for Cryptocurrency Trading - Phase 1 Implementation
Advanced technical indicators, order flow metrics, and market microstructure features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib not available, using manual calculations")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("⚠️ TA library not available, using manual calculations")

# ==================== CORE TECHNICAL INDICATORS ====================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    if TALIB_AVAILABLE:
        return talib.RSI(prices.values, timeperiod=period)
    else:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if TALIB_AVAILABLE:
        macd_line, signal_line, histogram = talib.MACD(
            prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        return pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }, index=prices.index)
    else:
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }, index=prices.index)

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """Calculate Bollinger Bands"""
    if TALIB_AVAILABLE:
        upper, middle, lower = talib.BBANDS(
            prices.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
        )
        return pd.DataFrame({
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower
        }, index=prices.index)
    else:
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return pd.DataFrame({
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower
        }, index=prices.index)

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator"""
    if TALIB_AVAILABLE:
        k, d = talib.STOCH(high.values, low.values, close.values, fastk_period=k_period, slowd_period=d_period)
        return pd.DataFrame({
            'stoch_k': k,
            'stoch_d': d
        }, index=close.index)
    else:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return pd.DataFrame({
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }, index=close.index)

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index"""
    if TALIB_AVAILABLE:
        return talib.ADX(high.values, low.values, close.values, timeperiod=period)
    else:
        # Manual ADX calculation
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=high.index).rolling(window=period).mean()
        minus_dm = pd.Series(minus_dm, index=high.index).rolling(window=period).mean()

        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1).rolling(window=period).mean()

        plus_di = 100 * (plus_dm / tr)
        minus_di = 100 * (minus_dm / tr)

        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = dx.rolling(window=period).mean()

        return adx

# ==================== ORDER FLOW INDICATORS ====================

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

def calculate_twap(df: pd.DataFrame) -> pd.Series:
    """Calculate Time Weighted Average Price"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    twap = typical_price.expanding().mean()
    return twap

def calculate_money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    # Positive and negative money flow
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    money_ratio = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + money_ratio))

    return mfi

def calculate_volume_profile(df: pd.DataFrame, periods: int = 20) -> pd.Series:
    """Calculate Volume Profile indicator"""
    high_vol = df['high'].rolling(window=periods).max()
    low_vol = df['low'].rolling(window=periods).min()
    close_vol = df['close'].rolling(window=periods).mean()
    volume_profile = (close_vol - low_vol) / (high_vol - low_vol) * df['volume'].rolling(window=periods).mean()

    return volume_profile

# ==================== MARKET MICROSTRUCTURE FEATURES ====================

def calculate_spread_pressure(df: pd.DataFrame) -> pd.Series:
    """Calculate spread pressure indicator"""
    spread = (df['high'] - df['low']) / df['close']
    spread_pressure = spread.rolling(window=20).mean() / spread.rolling(window=100).mean()
    return spread_pressure

def calculate_order_book_imbalance(df: pd.DataFrame) -> pd.Series:
    """Estimate order book imbalance from price and volume"""
    # Simple approximation using price changes and volume
    price_change = df['close'].pct_change()
    volume_change = df['volume'].pct_change()

    # Positive price change with increasing volume = buy imbalance
    buy_pressure = (price_change > 0) & (volume_change > 0)
    sell_pressure = (price_change < 0) & (volume_change > 0)

    imbalance = pd.Series(0.0, index=df.index)
    imbalance[buy_pressure] = 1.0
    imbalance[sell_pressure] = -1.0

    # Smooth the signal
    return imbalance.rolling(window=10).mean()

def calculate_tick_indicator(df: pd.DataFrame) -> pd.Series:
    """Calculate tick indicator (up vs down movements)"""
    price_changes = df['close'].diff()
    up_ticks = (price_changes > 0).astype(int)
    down_ticks = (price_changes < 0).astype(int)

    tick_ratio = (up_ticks - down_ticks).rolling(window=20).sum() / 20
    return tick_ratio

def calculate_trin(df: pd.DataFrame) -> pd.Series:
    """Calculate TRIN (Arms Index) - simplified version"""
    # Volume-based TRIN approximation
    advancing_volume = df['volume'].where(df['close'] > df['close'].shift(1), 0)
    declining_volume = df['volume'].where(df['close'] < df['close'].shift(1), 0)

    advancing_ma = advancing_volume.rolling(window=10).mean()
    declining_ma = declining_volume.rolling(window=10).mean()

    trin = advancing_ma / (declining_ma + 1e-8)
    return trin

# ==================== VOLATILITY INDICATORS ====================

def calculate_volatility_regime(df: pd.DataFrame, short_period: int = 20, long_period: int = 100) -> pd.DataFrame:
    """Calculate volatility regime classification"""
    close_prices = df['close']

    # Calculate different volatility measures
    volatility_short = close_prices.rolling(window=short_period).std()
    volatility_long = close_prices.rolling(window=long_period).std()

    # Volatility ratio
    volatility_ratio = volatility_short / volatility_long

    # ATR-based volatility
    atr = calculate_atr(df, period=14)
    atr_volatility = atr / close_prices

    # Regime classification
    high_vol_threshold = volatility_ratio.quantile(0.75)
    low_vol_threshold = volatility_ratio.quantile(0.25)

    regime = pd.Series('Normal', index=df.index)
    regime[volatility_ratio > high_vol_threshold] = 'High'
    regime[volatility_ratio < low_vol_threshold] = 'Low'

    return pd.DataFrame({
        'volatility_short': volatility_short,
        'volatility_long': volatility_long,
        'volatility_ratio': volatility_ratio,
        'atr_volatility': atr_volatility,
        'volatility_regime': regime
    }, index=df.index)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    if TALIB_AVAILABLE:
        return talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
    else:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

def calculate_garch_volatility(prices: pd.Series, period: int = 20) -> pd.Series:
    """Simple GARCH volatility approximation"""
    returns = prices.pct_change().dropna()

    # Simple GARCH(1,1) approximation
    omega = 0.0001
    alpha = 0.1
    beta = 0.85

    volatility = pd.Series(0.0, index=prices.index)
    variance = returns.var()

    for i in range(len(returns)):
        if i == 0:
            volatility.iloc[i] = np.sqrt(variance)
        else:
            variance = omega + alpha * returns.iloc[i-1]**2 + beta * variance
            volatility.iloc[i] = np.sqrt(variance)

    return volatility

# ==================== TREND AND MOMENTUM ====================

def calculate_trend_strength(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate trend strength using ADX and other indicators"""
    adx = calculate_adx(df['high'], df['low'], df['close'], period)

    # Normalize ADX to 0-1 scale
    trend_strength = adx / 100.0

    # Add momentum confirmation
    momentum = df['close'].pct_change(period)
    momentum_strength = abs(momentum) / momentum.abs().rolling(window=50).mean()

    # Combine trend and momentum
    combined_strength = (trend_strength + momentum_strength) / 2

    return combined_strength.fillna(0)

def calculate_price_momentum(df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Calculate multi-period price momentum"""
    momentum_data = {}

    for period in periods:
        momentum = df['close'].pct_change(period)
        momentum_data[f'momentum_{period}'] = momentum

    return pd.DataFrame(momentum_data, index=df.index)

def calculate_rate_of_change(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Calculate Rate of Change indicator"""
    roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
    return roc

# ==================== SUPPORT AND RESISTANCE ====================

def calculate_support_resistance_levels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate support and resistance levels"""
    highs = df['high'].rolling(window=window).max()
    lows = df['low'].rolling(window=window).min()

    resistance = highs.rolling(window=5).mean()
    support = lows.rolling(window=5).mean()

    current_price = df['close']
    support_distance = (current_price - support) / support
    resistance_distance = (resistance - current_price) / current_price

    return pd.DataFrame({
        'support': support,
        'resistance': resistance,
        'support_distance': support_distance,
        'resistance_distance': resistance_distance
    }, index=df.index)

def calculate_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate pivot points and support/resistance levels"""
    high = df['high']
    low = df['low']
    close = df['close']

    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)

    return pd.DataFrame({
        'pivot': pivot,
        'resistance_1': r1,
        'support_1': s1,
        'resistance_2': r2,
        'support_2': s2
    }, index=df.index)

# ==================== MULTI-TIMEFRAME FEATURES ====================

def calculate_multi_timeframe_features(df: pd.DataFrame, timeframes: List[int] = [5, 15, 30, 60]) -> Dict[int, pd.DataFrame]:
    """Calculate features for multiple timeframes"""
    timeframe_features = {}

    for timeframe in timeframes:
        print(f"🔧 Processing {timeframe}-minute timeframe...")

        # Resample data
        resampled = df.resample(f'{timeframe}T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Calculate timeframe-specific features
        features = calculate_timeframe_features(resampled, timeframe)
        timeframe_features[timeframe] = features

    return timeframe_features

def calculate_timeframe_features(df: pd.DataFrame, timeframe: int) -> pd.DataFrame:
    """Calculate features for a specific timeframe"""
    features = df.copy()

    # Basic indicators
    features['rsi'] = calculate_rsi(df['close'])
    macd_data = calculate_macd(df['close'])
    features = pd.concat([features, macd_data], axis=1)

    # Bollinger Bands
    bb_data = calculate_bollinger_bands(df['close'])
    features = pd.concat([features, bb_data], axis=1)

    # Volatility
    features['atr'] = calculate_atr(df)
    features['volatility'] = df['close'].rolling(window=20).std()

    # Trend strength
    features['trend_strength'] = calculate_trend_strength(df)

    # Additional indicators to match main feature count
    features['volume_sma'] = features['volume'].rolling(window=20).mean()
    features['volume_ratio'] = features['volume'] / features['volume_sma']
    features['price_change'] = features['close'].pct_change()
    features['high_low_ratio'] = features['high'] / features['low']

    # Timeframe-specific periods
    rsi_period = min(14, max(5, timeframe // 5))
    bb_period = min(20, max(10, timeframe // 3))

    return features

# ==================== MAIN FEATURE ENGINEERING FUNCTION ====================

def calculate_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all enhanced features for cryptocurrency trading
    """
    print("🔧 Calculating enhanced trading features...")

    features = df.copy()

    # Core technical indicators
    print("  📊 Calculating core technical indicators...")
    features['rsi'] = calculate_rsi(features['close'])
    features['rsi_7'] = calculate_rsi(features['close'], 7)
    features['rsi_21'] = calculate_rsi(features['close'], 21)

    macd_data = calculate_macd(features['close'])
    features = pd.concat([features, macd_data], axis=1)

    bb_data = calculate_bollinger_bands(features['close'])
    features = pd.concat([features, bb_data], axis=1)

    # Stochastic oscillator
    stoch_data = calculate_stochastic(features['high'], features['low'], features['close'])
    features = pd.concat([features, stoch_data], axis=1)

    # ADX
    features['adx'] = calculate_adx(features['high'], features['low'], features['close'])

    # Order flow indicators
    print("  💰 Calculating order flow indicators...")
    features['vwap'] = calculate_vwap(features)
    features['twap'] = calculate_twap(features)
    features['mfi'] = calculate_money_flow_index(features)
    features['volume_profile'] = calculate_volume_profile(features)

    # Market microstructure
    print("  🏛️ Calculating market microstructure features...")
    features['spread_pressure'] = calculate_spread_pressure(features)
    features['order_imbalance'] = calculate_order_book_imbalance(features)
    features['tick_indicator'] = calculate_tick_indicator(features)
    features['trin'] = calculate_trin(features)

    # Volatility analysis
    print("  📈 Calculating volatility features...")
    vol_data = calculate_volatility_regime(features)
    features = pd.concat([features, vol_data], axis=1)

    features['garch_volatility'] = calculate_garch_volatility(features['close'])

    # Momentum and trend
    print("  🚀 Calculating momentum and trend features...")
    momentum_data = calculate_price_momentum(features, [5, 10, 20])
    features = pd.concat([features, momentum_data], axis=1)

    features['roc'] = calculate_rate_of_change(features)
    features['trend_strength'] = calculate_trend_strength(features)

    # Support and resistance
    print("  📊 Calculating support and resistance levels...")
    sr_data = calculate_support_resistance_levels(features)
    features = pd.concat([features, sr_data], axis=1)

    pivot_data = calculate_pivot_points(features)
    features = pd.concat([features, pivot_data], axis=1)

    # Price-based features
    print("  💹 Calculating price-based features...")
    features['price_change'] = features['close'].pct_change()
    features['log_return'] = np.log(features['close'] / features['close'].shift(1))
    features['high_low_ratio'] = features['high'] / features['low']
    features['close_sma_ratio'] = features['close'] / features['close'].rolling(window=20).mean()

    # Volume-based features
    print("  📦 Calculating volume-based features...")
    features['volume_sma'] = features['volume'].rolling(window=20).mean()
    features['volume_ratio'] = features['volume'] / features['volume_sma']
    features['volume_change'] = features['volume'].pct_change()

    # Time-based features
    print("  ⏰ Calculating time-based features...")
    if hasattr(features.index, 'hour'):
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['is_session_open'] = ((features['hour'] >= 9) & (features['hour'] <= 16)).astype(int)

    # Lagged features
    print("  ⏪ Calculating lagged features...")
    for lag in [1, 2, 3, 5, 10]:
        features[f'close_lag_{lag}'] = features['close'].shift(lag)
        features[f'volume_lag_{lag}'] = features['volume'].shift(lag)

    print(f"✅ Enhanced features calculated: {features.shape[1]} features")
    return features

def prepare_multi_scale_features(df: pd.DataFrame) -> Dict[int, torch.Tensor]:
    """
    Prepare multi-scale features for transformer input
    """
    if not TORCH_AVAILABLE:
        print("⚠️ PyTorch not available, returning empty dict")
        return {}

    print("🔧 Preparing multi-scale features...")

    # Calculate features for different timeframes
    timeframe_features = calculate_multi_timeframe_features(df)

    # Prepare tensors for each timeframe
    scale_tensors = {}

    for timeframe, features_df in timeframe_features.items():
        # Select numeric features
        numeric_features = features_df.select_dtypes(include=[np.number])

        # Handle missing values
        numeric_features = numeric_features.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(numeric_features)

        # Convert to tensor
        tensor_data = torch.FloatTensor(normalized_features)

        # Adjust sequence length based on timeframe
        seq_len = min(250, len(tensor_data) // (timeframe // 5))

        if len(tensor_data) > seq_len:
            tensor_data = tensor_data[-seq_len:]

        scale_tensors[timeframe] = tensor_data.unsqueeze(0)  # Add batch dimension

    return scale_tensors

# ==================== FEATURE SELECTION ====================

def select_important_features(df: pd.DataFrame, n_features: int = 50) -> pd.DataFrame:
    """
    Select most important features using correlation and variance
    """
    print("🎯 Selecting important features...")

    # Remove non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Remove features with too many missing values
    missing_threshold = 0.3
    numeric_df = numeric_df.loc[:, numeric_df.isnull().mean() < missing_threshold]

    # Remove low variance features
    variance_threshold = 0.001
    variance = numeric_df.var()
    high_variance_features = variance[variance > variance_threshold].index
    numeric_df = numeric_df[high_variance_features]

    # Remove highly correlated features
    correlation_matrix = numeric_df.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # Find features to remove
    threshold = 0.95
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    # Keep only uncorrelated features
    selected_features = [col for col in numeric_df.columns if col not in to_drop]

    # Select top features by importance (simplified)
    if len(selected_features) > n_features:
        # Use correlation with target (close price) as importance metric
        target_correlation = numeric_df[selected_features].corrwith(numeric_df['close']).abs()
        top_features = target_correlation.nlargest(n_features).index
        selected_features = list(top_features)

    print(f"✅ Selected {len(selected_features)} important features")
    return df[selected_features]

# ==================== TESTING ====================

if __name__ == "__main__":
    # Test enhanced feature engineering
    print("🧪 Testing enhanced feature engineering...")

    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5T')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(50000, 60000, 1000),
        'high': np.random.uniform(50000, 60000, 1000),
        'low': np.random.uniform(50000, 60000, 1000),
        'close': np.random.uniform(50000, 60000, 1000),
        'volume': np.random.uniform(100, 1000, 1000)
    }, index=dates)

    # Add symbol column
    sample_data['symbol'] = 'BTCUSDT'

    print(f"📊 Sample data shape: {sample_data.shape}")

    # Calculate enhanced features
    enhanced_features = calculate_enhanced_features(sample_data)

    print(f"✅ Enhanced features shape: {enhanced_features.shape}")
    print(f"📋 Feature columns: {list(enhanced_features.columns[:10])}...")

    # Test feature selection
    selected_features = select_important_features(enhanced_features, n_features=30)
    print(f"🎯 Selected features shape: {selected_features.shape}")

    # Test multi-scale features
    if TORCH_AVAILABLE:
        multi_scale_tensors = prepare_multi_scale_features(sample_data)
        print(f"🔧 Multi-scale tensors for timeframes: {list(multi_scale_tensors.keys())}")
        for timeframe, tensor in multi_scale_tensors.items():
            print(f"   {timeframe}min: {tensor.shape}")
    else:
        print("⚠️ PyTorch not available for multi-scale tensor test")

    print("\n✅ Enhanced feature engineering test completed successfully!")