import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def create_test_crypto_data():
    """Create test crypto data for performance evaluation"""
    
    print("🔧 Creating test crypto data for performance evaluation...")
    
    # Create 6 months of 5-minute data for testing
    start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.now()
    
    # Generate 5-minute intervals
    dates = pd.date_range(start=start_date, end=end_date, freq='5T')
    print(f"📅 Generated {len(dates):,} 5-minute test intervals")
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    all_data = []
    
    for symbol in symbols:
        print(f"📊 Creating test data for {symbol}...")
        
        # Set realistic starting prices and volatility
        if symbol == 'BTCUSDT':
            initial_price = 45000
            volatility = 0.02
        elif symbol == 'ETHUSDT':
            initial_price = 2800
            volatility = 0.025
        else:  # BNBUSDT
            initial_price = 400
            volatility = 0.03
        
        # Use numpy for faster generation
        n_periods = len(dates)
        
        # Generate price walk with crypto characteristics
        trend = 0.00008  # Small upward trend for testing profitability
        changes = np.random.normal(trend, volatility, n_periods)
        
        # Apply volatility clustering
        vol_multiplier = 1 + 0.2 * np.sin(np.arange(n_periods) / 500)
        changes = changes * vol_multiplier
        
        # Calculate prices
        prices = [initial_price]
        for change in changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))
        
        prices = np.array(prices)
        
        # Vectorized OHLCV generation
        opens = prices * (1 + np.random.normal(0, 0.001, n_periods))
        closes = prices
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.003, n_periods)))
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.003, n_periods)))
        
        # Volume with daily cycles
        time_factors = np.sin(2 * np.pi * (np.arange(n_periods) % 288) / 288) * 0.2 + 1
        volumes = np.random.lognormal(7, 1.2, n_periods) * time_factors
        
        # Create DataFrame efficiently
        for i, date in enumerate(dates):
            all_data.append({
                'date': date,
                'tic': symbol,
                'open': opens[i],
                'high': highs[i],
                'low': lows[i],
                'close': closes[i],
                'volume': volumes[i]
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    print(f"✅ Created test dataset:")
    print(f"   📊 Shape: {df.shape}")
    print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   💰 Symbols: {sorted(df['tic'].unique())}")
    
    return df

def test_model_performance(model_path):
    """Test the trained model and calculate performance metrics"""
    
    print(f"\n🧪 Testing Model: {model_path}")
    print("="*60)
    
    # Create test data
    test_df = create_test_crypto_data()
    
    # Add technical indicators
    print("\n🔧 Adding technical indicators...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30'],
        use_vix=False,
        use_turbulence=False
    )
    
    processed_df = fe.preprocess_data(test_df)
    print(f"✅ Processed test data shape: {processed_df.shape}")
    
    # Environment parameters
    stock_dim = len(test_df['tic'].unique())
    tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']
    state_space = 1 + stock_dim + stock_dim + len(tech_indicators) * stock_dim
    action_space = stock_dim
    
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": action_space,
        "tech_indicator_list": tech_indicators,
        "stock_dim": stock_dim,
        "num_stock_shares": [0] * stock_dim,
    }
    
    # Create test environment
    test_env = DummyVecEnv([lambda: StockTradingEnv(df=processed_df, **env_kwargs)])
    
    # Load model
    print(f"\n📂 Loading model: {model_path}")
    model = PPO.load(model_path)
    
    # Test the model
    print("🚀 Running performance test...")
    obs = test_env.reset()
    total_reward = 0.0
    portfolio_values = []
    actions_log = []
    rewards_log = []
    steps = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        
        total_reward += float(reward)
        rewards_log.append(float(reward))
        actions_log.append(action[0])
        steps += 1
        
        # Track portfolio value
        if info and len(info) > 0:
            pv = info[0].get('total_asset', info[0].get('portfolio_value', None))
            if pv is not None:
                portfolio_values.append(float(pv))
        
        if steps % 10000 == 0:
            print(f"   Progress: {steps:,} steps, Current Reward: {float(reward):.4f}")
        
        if done:
            break
    
    return {
        'model_name': model_path,
        'total_reward': total_reward,
        'steps': steps,
        'portfolio_values': portfolio_values,
        'actions': actions_log,
        'rewards': rewards_log,
        'final_info': info[0] if info else {},
        'symbols': sorted(test_df['tic'].unique())
    }

def generate_profitability_report(results):
    """Generate comprehensive profitability analysis"""
    
    print("\n" + "="*80)
    print("💰 CRYPTO TRADING ALGORITHM PROFITABILITY REPORT")
    print("="*80)
    
    model_name = results['model_name']
    print(f"🤖 Model: {model_name}")
    print(f"💱 Symbols: {results['symbols']}")
    print(f"💵 Initial Capital: $1,000,000")
    print(f"📊 Trading Steps: {results['steps']:,}")
    print(f"🎯 Cumulative Reward: ${results['total_reward']:,.2f}")
    
    # Final trading results
    final_info = results['final_info']
    if final_info:
        print(f"\n📋 Final Trading Session:")
        for key, value in final_info.items():
            if isinstance(value, (int, float)):
                if 'asset' in key.lower() or 'balance' in key.lower():
                    print(f"  💰 {key}: ${value:,.2f}")
                else:
                    print(f"  📈 {key}: {value:.4f}")
    
    # Portfolio performance analysis
    portfolio_values = results['portfolio_values']
    if portfolio_values and len(portfolio_values) > 1:
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        max_value = np.max(portfolio_values)
        min_value = np.min(portfolio_values)
        
        print(f"\n💼 PORTFOLIO PERFORMANCE:")
        print(f"  💵 Initial Value: ${initial_value:,.2f}")
        print(f"  💰 Final Value: ${final_value:,.2f}")
        print(f"  📈 Total Return: {total_return:+.2f}%")
        print(f"  🔥 Max Value: ${max_value:,.2f}")
        print(f"  📉 Min Value: ${min_value:,.2f}")
        print(f"  💎 Profit/Loss: ${final_value - initial_value:+,.2f}")
        
        # Performance metrics
        if total_return > 0:
            print(f"  ✅ PROFITABLE: +{total_return:.2f}% return")
        else:
            print(f"  ❌ LOSS: {total_return:.2f}% return")
        
        # Calculate max drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        print(f"  📊 Max Drawdown: {max_drawdown*100:.2f}%")
        
        # Calculate Sharpe ratio (approximate)
        if len(portfolio_values) > 1:
            returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                      for i in range(1, len(portfolio_values))]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(288)
                print(f"  📐 Sharpe Ratio: {sharpe:.3f}")
                if sharpe > 1.0:
                    print(f"    ✅ EXCELLENT risk-adjusted returns")
                elif sharpe > 0.5:
                    print(f"    ✅ GOOD risk-adjusted returns") 
                else:
                    print(f"    ⚠️ POOR risk-adjusted returns")
    
    # Trading activity analysis
    rewards = results['rewards']
    if rewards:
        positive_rewards = [r for r in rewards if r > 0]
        negative_rewards = [r for r in rewards if r < 0]
        
        win_rate = len(positive_rewards) / len(rewards) * 100
        avg_win = np.mean(positive_rewards) if positive_rewards else 0
        avg_loss = np.mean(negative_rewards) if negative_rewards else 0
        
        print(f"\n📊 TRADING PERFORMANCE:")
        print(f"  🎯 Win Rate: {win_rate:.1f}%")
        print(f"  💚 Average Win: ${avg_win:.6f}")
        print(f"  💔 Average Loss: ${avg_loss:.6f}")
        print(f"  📈 Profitable Trades: {len(positive_rewards):,}")
        print(f"  📉 Losing Trades: {len(negative_rewards):,}")
    
    # Overall assessment
    if portfolio_values and len(portfolio_values) > 1:
        final_value = portfolio_values[-1]
        initial_value = portfolio_values[0]
        
        print(f"\n🏆 ALGORITHM ASSESSMENT:")
        if final_value > initial_value * 1.1:  # 10%+ profit
            print(f"  🔥 HIGHLY PROFITABLE - Algorithm generated significant returns")
        elif final_value > initial_value * 1.02:  # 2%+ profit
            print(f"  ✅ PROFITABLE - Algorithm beat basic investment")
        elif final_value > initial_value:  # Any profit
            print(f"  ✅ MODERATELY PROFITABLE - Algorithm generated positive returns")
        else:
            print(f"  ⚠️ NOT PROFITABLE - Algorithm lost money")
        
        # Annualized return (approximate for 6 months test)
        periods_per_year = 2  # 6 months = 0.5 year, so 2 periods per year
        annualized_return = ((final_value / initial_value) ** periods_per_year - 1) * 100
        print(f"  📅 Estimated Annual Return: {annualized_return:+.2f}%")
    
    print("="*80)

def main():
    """Test both available models"""
    
    print("🚀 CRYPTO TRADING ALGORITHM PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Test both models
    models = [
        'crypto_5min_synthetic_model.zip',
        'crypto_5min_gpu_model.zip'
    ]
    
    for model_path in models:
        try:
            results = test_model_performance(model_path)
            generate_profitability_report(results)
            print(f"\n✅ Successfully tested {model_path}")
        except Exception as e:
            print(f"❌ Error testing {model_path}: {e}")
            continue
    
    print(f"\n🎉 Performance analysis completed!")

if __name__ == "__main__":
    main()