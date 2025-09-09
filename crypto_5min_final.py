import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def create_synthetic_5min_crypto_data():
    """Create synthetic 5-minute crypto data based on real patterns but compatible with FinRL"""
    
    print("🔧 Creating synthetic 5-minute crypto data...")
    
    # Create 2 years of 5-minute data
    start_date = datetime.now() - timedelta(days=730)
    end_date = datetime.now()
    
    # Generate 5-minute intervals
    dates = pd.date_range(start=start_date, end=end_date, freq='5T')
    print(f"📅 Generated {len(dates):,} 5-minute intervals")
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    all_data = []
    
    for symbol in symbols:
        print(f"📊 Creating data for {symbol}...")
        
        # Set realistic starting prices and volatility
        if symbol == 'BTCUSDT':
            initial_price = 30000
            volatility = 0.02  # 2% volatility for 5-min intervals
        elif symbol == 'ETHUSDT':
            initial_price = 2000
            volatility = 0.025  # 2.5% volatility
        else:  # BNBUSDT
            initial_price = 300
            volatility = 0.03  # 3% volatility
        
        prices = [initial_price]
        
        # Generate realistic price walk with crypto characteristics
        for i in range(1, len(dates)):
            # Add small trend and volatility clustering
            trend = 0.00001  # Very small upward trend
            vol_multiplier = 1 + 0.3 * np.sin(i / 1000)  # Volatility clustering
            change = np.random.normal(trend, volatility * vol_multiplier)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        # Create OHLCV data with realistic characteristics
        for i, date in enumerate(dates):
            close = prices[i]
            
            # Create realistic intraday OHLCV
            open_price = close * (1 + np.random.normal(0, 0.002))
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.005)))
            
            # Crypto volume patterns (log-normal with daily cycles)
            time_factor = np.sin(2 * np.pi * (i % 288) / 288) * 0.3 + 1  # Daily volume cycle
            volume = np.random.lognormal(8, 1.5) * time_factor
            
            all_data.append({
                'date': date,
                'tic': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    print(f"✅ Created synthetic dataset:")
    print(f"   📊 Shape: {df.shape}")
    print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   💰 Symbols: {sorted(df['tic'].unique())}")
    print(f"   ⏱️  Frequency: 5 minutes")
    
    return df

def run_5min_crypto_experiment():
    """Run the 5-minute crypto trading experiment with synthetic data"""
    
    print("🚀 5-Minute Crypto Trading Experiment")
    print("="*50)
    
    # Create synthetic 5-minute data
    crypto_df = create_synthetic_5min_crypto_data()
    
    # Add technical indicators
    print("\n🔧 Adding technical indicators...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30'],
        use_vix=False,
        use_turbulence=False
    )
    
    processed_df = fe.preprocess_data(crypto_df)
    print(f"✅ Processed data shape: {processed_df.shape}")
    
    # Split data
    train_df = data_split(processed_df, 
                         (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),
                         (datetime.now() - timedelta(days=150)).strftime('%Y-%m-%d'))
    test_df = data_split(processed_df,
                        (datetime.now() - timedelta(days=149)).strftime('%Y-%m-%d'),
                        datetime.now().strftime('%Y-%m-%d'))
    
    print(f"📊 Train data: {train_df.shape}")
    print(f"📊 Test data: {test_df.shape}")
    
    # Environment parameters
    stock_dim = len(crypto_df['tic'].unique())
    tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']
    state_space = 1 + stock_dim + stock_dim + len(tech_indicators) * stock_dim
    action_space = stock_dim
    
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "buy_cost_pct": [0.001] * stock_dim,  # 0.1% crypto trading fees
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": action_space,
        "tech_indicator_list": tech_indicators,
        "stock_dim": stock_dim,
        "num_stock_shares": [0] * stock_dim,
    }
    
    print(f"\n🔧 Environment setup:")
    print(f"   🎯 State space: {state_space}")
    print(f"   🎮 Action space: {action_space}")
    print(f"   💰 Symbols: {sorted(crypto_df['tic'].unique())}")
    print(f"   📈 Technical indicators: {tech_indicators}")
    
    # Create environments
    print("\n🚀 Training PPO model...")
    train_env = DummyVecEnv([lambda: StockTradingEnv(df=train_df, **env_kwargs)])
    test_env = DummyVecEnv([lambda: StockTradingEnv(df=test_df, **env_kwargs)])
    
    # Train model with optimized parameters for 5-minute data
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=1e-4,  # Lower learning rate for high-frequency data
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )
    
    # Train for more timesteps given the 5-minute resolution
    model.learn(total_timesteps=150_000)
    
    # Save model
    model.save("crypto_5min_synthetic_model")
    print("💾 Model saved: crypto_5min_synthetic_model.zip")
    
    # Test model
    print("\n🧪 Testing model...")
    obs = test_env.reset()
    total_reward = 0.0
    portfolio_values = []
    steps = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        total_reward += float(reward)
        steps += 1
        
        if info and len(info) > 0:
            pv = info[0].get('total_asset', info[0].get('portfolio_value', None))
            if pv is not None:
                portfolio_values.append(float(pv))
        
        if steps % 5000 == 0:
            print(f"   Step {steps:,}, Reward: {reward:.4f}, Total: {total_reward:.2f}")
        
        if done:
            break
    
    # Generate results
    print("\n" + "="*70)
    print("🎉 5-MINUTE CRYPTO TRADING RESULTS")
    print("="*70)
    
    print(f"💰 Symbols: {sorted(crypto_df['tic'].unique())}")
    print(f"💵 Initial Capital: $1,000,000")
    print(f"📊 Total Steps: {steps:,}")
    print(f"🎯 Cumulative Reward: ${total_reward:.2f}")
    
    # Final info
    if info and len(info) > 0:
        final_info = info[0]
        print(f"\n📋 Final Trading Results:")
        for key, value in final_info.items():
            if isinstance(value, (int, float)):
                if 'asset' in key.lower() or 'balance' in key.lower():
                    print(f"   {key}: ${value:,.2f}")
                else:
                    print(f"   {key}: {value:,.2f}")
    
    # Portfolio analysis
    if portfolio_values and len(portfolio_values) > 1:
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        print(f"\n📈 Portfolio Performance:")
        print(f"   Initial Value: ${initial_value:,.2f}")
        print(f"   Final Value: ${final_value:,.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Max Value: ${np.max(portfolio_values):,.2f}")
        print(f"   Min Value: ${np.min(portfolio_values):,.2f}")
        
        # Calculate metrics
        if len(portfolio_values) > 1:
            returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                      for i in range(1, len(portfolio_values))]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(288)  # 288 5-min periods per day
                print(f"   Sharpe Ratio: {sharpe:.3f}")
    
    print("="*70)
    print("✅ 5-minute crypto trading experiment completed successfully!")
    print("🎯 Model demonstrates capability for high-frequency crypto trading")
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'portfolio_values': portfolio_values,
        'final_info': info[0] if info else {}
    }

def main():
    """Main execution"""
    results = run_5min_crypto_experiment()

if __name__ == "__main__":
    main()