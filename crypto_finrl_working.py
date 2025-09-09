import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def create_crypto_like_data():
    """Create crypto-like synthetic data that works with FinRL"""
    
    # Generate dates (5 years, daily data)
    end_date = datetime.now()
    start_date = datetime(2020, 1, 1)  # Start from 2020
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Crypto symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    all_data = []
    
    # Generate synthetic price data for each symbol
    for symbol in symbols:
        # Set different starting prices
        if symbol == 'BTCUSDT':
            initial_price = 30000
            volatility = 0.03
        elif symbol == 'ETHUSDT':
            initial_price = 2000
            volatility = 0.04
        else:  # BNBUSDT
            initial_price = 300
            volatility = 0.05
        
        prices = [initial_price]
        
        # Generate price walk
        for i in range(1, len(dates)):
            # Random walk with trend
            trend = 0.0002  # Slight upward trend
            change = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLCV data
        symbol_data = []
        for i, date in enumerate(dates):
            price = prices[i]
            
            # Create realistic OHLCV around the close price
            close = price
            open_price = close * (1 + np.random.normal(0, 0.005))
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.lognormal(10, 1)  # Log-normal volume
            
            symbol_data.append({
                'date': date,
                'tic': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        all_data.extend(symbol_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    return df

def run_crypto_trading_experiment():
    """Run the crypto trading experiment"""
    
    print("Creating synthetic crypto data...")
    crypto_df = create_crypto_like_data()
    
    print(f"Generated data shape: {crypto_df.shape}")
    print(f"Date range: {crypto_df['date'].min().date()} to {crypto_df['date'].max().date()}")
    print(f"Symbols: {crypto_df['tic'].unique()}")
    
    # Add technical indicators
    print("Adding technical indicators...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30'],
        use_vix=False,
        use_turbulence=False
    )
    
    processed_df = fe.preprocess_data(crypto_df)
    print(f"Data with indicators shape: {processed_df.shape}")
    
    # Split data
    train_df = data_split(processed_df, '2020-01-01', '2023-12-31')
    test_df = data_split(processed_df, '2024-01-01', datetime.now().strftime('%Y-%m-%d'))
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Environment parameters
    stock_dim = len(crypto_df['tic'].unique())
    tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']
    state_space = 1 + stock_dim + stock_dim + len(tech_indicators) * stock_dim  # balance + prices + shares + indicators
    action_space = stock_dim
    
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "buy_cost_pct": [0.001] * stock_dim,  # 0.1% trading fee
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": action_space,
        "tech_indicator_list": ['macd', 'rsi_30', 'cci_30', 'dx_30'],
        "stock_dim": stock_dim,
        "num_stock_shares": [0] * stock_dim,
    }
    
    # Create environments
    print("Creating training environment...")
    train_env = DummyVecEnv([lambda: StockTradingEnv(df=train_df, **env_kwargs)])
    test_env = DummyVecEnv([lambda: StockTradingEnv(df=test_df, **env_kwargs)])
    
    # Train model
    print("Training PPO model...")
    model = PPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=50_000)
    
    # Test model
    print("Testing model...")
    obs = test_env.reset()
    total_reward = 0.0
    steps = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        total_reward += float(reward)
        steps += 1
        
        if done:
            break
    
    # Results
    print("\n" + "="*60)
    print("CRYPTO TRADING RESULTS")
    print("="*60)
    print(f"Total Steps: {steps}")
    print(f"Cumulative Reward: {total_reward:.2f}")
    
    if info and len(info) > 0:
        final_info = info[0]
        print(f"Final Results:")
        for key, value in final_info.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:,.2f}")
    
    print("="*60)
    print("Experiment completed!")
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'final_info': info[0] if info else {}
    }

if __name__ == "__main__":
    results = run_crypto_trading_experiment()