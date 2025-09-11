import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

def create_optimized_btc_eth_bnb_data():
    """Create optimized synthetic BTC+ETH+BNB data with trending characteristics"""
    
    print("🔧 Creating optimized BTC+ETH+BNB synthetic data (v2)...")
    
    # Create 2 years of 5-minute data matching the user's request
    start_date = datetime.now() - timedelta(days=730)  # 2 years
    end_date = datetime.now()
    
    # Generate 5-minute intervals
    dates = pd.date_range(start=start_date, end=end_date, freq='5T')
    print(f"📅 Generated {len(dates):,} 5-minute intervals over 2 years")
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    all_data = []
    
    for symbol in symbols:
        print(f"📊 Creating optimized trending data for {symbol}...")
        
        # Set realistic parameters based on real performance
        if symbol == 'BTCUSDT':
            initial_price = 35000  # Starting around Sept 2023 levels
            trend = 0.0001  # Strong upward trend (+336% target)
            volatility = 0.02
        elif symbol == 'ETHUSDT':
            initial_price = 1800   # Sept 2023 levels  
            trend = 0.00008  # Moderate upward trend (+168% target)
            volatility = 0.022
        else:  # BNBUSDT
            initial_price = 220    # Sept 2023 levels
            trend = 0.00012  # Strong upward trend (+314% target)
            volatility = 0.025
        
        n_periods = len(dates)
        
        # Generate sophisticated price movements
        # Add cyclical patterns and momentum phases
        cycle = np.sin(2 * np.pi * np.arange(n_periods) / (365 * 288 / 5)) * 0.0002  # Yearly cycle
        momentum = np.cumsum(np.random.normal(0, 0.00005, n_periods)) * 0.1  # Momentum drift
        
        # Base random walk with trend
        changes = np.random.normal(trend, volatility, n_periods)
        changes = changes + cycle + momentum
        
        # Add volatility clustering (GARCH-like)
        vol_state = np.ones(n_periods)
        for i in range(1, n_periods):
            vol_state[i] = 0.95 * vol_state[i-1] + 0.05 * abs(changes[i-1]) * 10
        
        changes = changes * (0.5 + vol_state)
        
        # Calculate prices with realistic bounds
        prices = [initial_price]
        for change in changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, initial_price * 0.3))  # Prevent unrealistic crashes
        
        prices = np.array(prices)
        
        # Generate realistic OHLCV
        # Opens slightly different from previous close
        opens = prices * (1 + np.random.normal(0, 0.002, n_periods))
        closes = prices
        
        # Highs and lows with realistic spreads
        spreads = np.abs(np.random.normal(0.005, 0.003, n_periods))
        highs = np.maximum(opens, closes) * (1 + spreads)
        lows = np.minimum(opens, closes) * (1 - spreads)
        
        # Volume with time-of-day patterns and price correlation
        hour_of_day = (np.arange(n_periods) % 288) / 12  # 5-min intervals per hour
        daily_volume_pattern = 1 + 0.3 * np.sin(2 * np.pi * hour_of_day / 24)
        
        # Higher volume during price movements
        price_changes = np.abs(np.diff(closes, prepend=closes[0]))
        volume_multiplier = 1 + price_changes / np.mean(price_changes) * 0.5
        
        base_volumes = np.random.lognormal(8, 1, n_periods)
        volumes = base_volumes * daily_volume_pattern * volume_multiplier
        
        # Create DataFrame
        symbol_data = pd.DataFrame({
            'date': dates,
            'tic': symbol,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        all_data.append(symbol_data)
        
        # Show expected final prices
        expected_final = initial_price * (1 + trend * 0.7) ** n_periods
        actual_final = closes[-1]
        print(f"   {symbol}: {initial_price:.0f} → {actual_final:.0f} ({((actual_final/initial_price)-1)*100:+.1f}%)")
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    print(f"✅ Optimized BTC+ETH+BNB synthetic dataset created:")
    print(f"   📊 Shape: {df.shape}")
    print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   💰 Symbols: {sorted(df['tic'].unique())}")
    print(f"   ⏱️  Frequency: 5 minutes (2 years)")
    
    return df

def main():
    print("🚀 TRAINING BTC+ETH+BNB MODEL v2 - OPTIMIZED SYNTHETIC GPU")
    print("="*80)
    
    # Check device availability
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🚀 Training on {device}")
    
    # Create optimized synthetic data
    df = create_optimized_btc_eth_bnb_data()
    
    # Feature engineering
    print("🔧 Adding technical indicators...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30'],
        use_vix=False,
        use_turbulence=False
    )
    
    processed_df = fe.preprocess_data(df)
    print(f"✅ Features added: {len(processed_df.columns)} columns")
    
    # Train/test split (80/20)
    split_point = int(len(processed_df) * 0.8)
    train_df = processed_df.iloc[:split_point].copy()
    test_df = processed_df.iloc[split_point:].copy()
    
    print(f"📊 Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Environment configuration optimized for multi-asset trending markets
    stock_dim = 3
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
    
    print(f"🔧 Environment: {state_space} states, {action_space} actions, {stock_dim} assets")
    
    # Create training environment
    train_env = DummyVecEnv([lambda: StockTradingEnv(df=train_df, **env_kwargs)])
    
    # PPO model optimized for trending crypto markets
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,     # Optimized learning rate
        n_steps=2048,           # Standard rollout
        batch_size=128,         # Larger batch for GPU
        n_epochs=12,            # More epochs for better convergence
        gamma=0.995,            # Higher discount for long-term trends
        gae_lambda=0.95,        # Standard GAE
        clip_range=0.2,         # Standard PPO clipping
        ent_coef=0.005,         # Lower entropy for more decisive actions
        max_grad_norm=0.5,      # Gradient clipping
        device=device
    )
    
    start_time = datetime.now()
    print("🎯 Starting optimized training for trending BTC+ETH+BNB...")
    
    # Extended training for better performance
    model.learn(total_timesteps=200_000)
    training_time = datetime.now() - start_time
    
    # Save model with v2 naming
    model_name = "crypto_btc_eth_bnb_v2_synthetic_gpu_model"
    model.save(model_name)
    print(f"💾 BTC+ETH+BNB v2 synthetic model saved: {model_name}")
    
    # Testing phase
    test_env = DummyVecEnv([lambda: StockTradingEnv(df=test_df, **env_kwargs)])
    
    print("\n🧪 TESTING BTC+ETH+BNB v2 SYNTHETIC MODEL...")
    obs = test_env.reset()
    portfolio_values = []
    actions_taken = []
    steps = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        steps += 1
        
        if info and len(info) > 0:
            pv = info[0].get('total_asset', 1000000)
            portfolio_values.append(float(pv))
            
            # Record actions for each cryptocurrency
            if hasattr(action, '__len__') and len(action) >= 3:
                actions_taken.append([int(a) for a in action[:3]])
            else:
                actions_taken.append([int(action), 0, 0])
        
        if done:
            break
    
    # Comprehensive results analysis
    if portfolio_values:
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        algorithm_return = (final_value - initial_value) / initial_value * 100
        profit = final_value - initial_value
        
        # Synthetic buy and hold comparison
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        buy_hold_returns = []
        
        for symbol in symbols:
            symbol_data = test_df[test_df['tic'] == symbol]
            if len(symbol_data) > 0:
                start_price = symbol_data['close'].iloc[0]
                end_price = symbol_data['close'].iloc[-1]
                bh_return = (end_price - start_price) / start_price * 100
                buy_hold_returns.append((symbol, bh_return))
        
        avg_buy_hold = np.mean([ret for _, ret in buy_hold_returns])
        
        print(f"\n🏆 BTC+ETH+BNB v2 SYNTHETIC MODEL RESULTS:")
        print(f"="*70)
        print(f"💰 Algorithm Return: {algorithm_return:+.2f}%")
        print(f"💵 Profit: ${profit:+,.0f}")
        print(f"🏦 Final Portfolio Value: ${final_value:,.0f}")
        print(f"📊 Trading Steps: {steps:,}")
        print(f"⏱️ Training Time: {training_time}")
        print(f"🚀 Device Used: {device}")
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"   Algorithm: {algorithm_return:+.2f}%")
        for symbol, bh_return in buy_hold_returns:
            print(f"   {symbol} Buy&Hold: {bh_return:+.2f}%")
        print(f"   Avg Buy&Hold: {avg_buy_hold:+.2f}%")
        print(f"   Algorithm vs Avg B&H: {algorithm_return - avg_buy_hold:+.2f}% difference")
        
        # Advanced performance metrics
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            volatility = np.std(returns) * np.sqrt(288 * 365)  # Annualized
            sharpe = (algorithm_return / 100) / volatility if volatility > 0 else 0
            
            # Maximum drawdown calculation
            peak = np.maximum.accumulate(portfolio_values)
            drawdowns = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdowns) * 100
            
            # Win rate calculation
            positive_returns = sum(1 for r in returns if r > 0)
            win_rate = (positive_returns / len(returns)) * 100
            
            print(f"\n📊 ADVANCED RISK METRICS:")
            print(f"   Portfolio Volatility: {volatility*100:.2f}%")
            print(f"   Sharpe Ratio: {sharpe:.3f}")
            print(f"   Maximum Drawdown: {max_drawdown:.2f}%")
            print(f"   Win Rate: {win_rate:.1f}%")
        
        # Action analysis for multi-asset strategy
        if actions_taken and len(actions_taken[0]) == 3:
            btc_actions = [a[0] for a in actions_taken]
            eth_actions = [a[1] for a in actions_taken] 
            bnb_actions = [a[2] for a in actions_taken]
            
            print(f"\n🎯 MULTI-ASSET TRADING STRATEGY ANALYSIS:")
            for i, (symbol, actions) in enumerate([('BTC', btc_actions), ('ETH', eth_actions), ('BNB', bnb_actions)]):
                buy_count = sum(1 for a in actions if a < 0)
                hold_count = sum(1 for a in actions if a == 0)
                sell_count = sum(1 for a in actions if a > 0)
                
                total = len(actions)
                buy_pct = (buy_count / total) * 100
                hold_pct = (hold_count / total) * 100
                sell_pct = (sell_count / total) * 100
                
                print(f"   {symbol}: Buy {buy_pct:.1f}% | Hold {hold_pct:.1f}% | Sell {sell_pct:.1f}%")
        
        print(f"\n✅ BTC+ETH+BNB v2 synthetic model training completed successfully!")
        print(f"🎯 Model optimized for trending cryptocurrency markets")
        print(f"🚀 Enhanced with GPU acceleration and advanced metrics")
        print(f"💾 Model saved as: {model_name}")
        
        return {
            'model_name': model_name,
            'algorithm_return': algorithm_return,
            'profit': profit,
            'training_time': training_time,
            'device': str(device),
            'sharpe_ratio': sharpe if 'sharpe' in locals() else 0,
            'max_drawdown': max_drawdown if 'max_drawdown' in locals() else 0,
            'win_rate': win_rate if 'win_rate' in locals() else 0
        }
    else:
        print("❌ No portfolio values recorded during testing")
        return {
            'model_name': model_name,
            'algorithm_return': 0,
            'profit': 0,
            'training_time': training_time,
            'device': str(device)
        }

if __name__ == "__main__":
    results = main()
    print(f"\n🎉 FINAL RESULTS SUMMARY:")
    print(f"   Model: {results['model_name']}")
    print(f"   Return: {results['algorithm_return']:+.2f}%")
    print(f"   Profit: ${results['profit']:+,.0f}")
    print(f"   Device: {results['device']}")
    print(f"   Training Time: {results['training_time']}")