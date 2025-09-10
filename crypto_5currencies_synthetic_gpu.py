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
import os

def check_device_availability():
    """Check available devices and set optimal device for Mac M4"""
    
    print("🔧 Checking device availability...")
    
    # Check PyTorch version
    print(f"   PyTorch version: {torch.__version__}")
    
    # Check MPS availability (Mac GPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"   ✅ MPS (Mac GPU) is available!")
        print(f"   🚀 Using device: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"   ✅ CUDA GPU available!")
        print(f"   🚀 Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"   ⚠️ Only CPU available")
        print(f"   🔄 Using device: {device}")
    
    return device

def create_synthetic_crypto_data(symbol, periods=210240):
    """Create synthetic crypto data for individual symbol based on real characteristics"""
    
    print(f"🔧 Creating synthetic {symbol} data...")
    
    # Create time series
    start_date = datetime.now() - timedelta(days=730)
    dates = pd.date_range(start=start_date, periods=periods, freq='5T')
    
    # Symbol-specific parameters based on real crypto characteristics
    crypto_params = {
        'ADAUSDT': {'initial_price': 0.35, 'volatility': 0.025, 'trend': 0.00005},
        'SOLUSDT': {'initial_price': 25.0, 'volatility': 0.035, 'trend': 0.00008},
        'MATICUSDT': {'initial_price': 0.85, 'volatility': 0.030, 'trend': 0.00003},
        'DOTUSDT': {'initial_price': 6.5, 'volatility': 0.028, 'trend': 0.00004},
        'LINKUSDT': {'initial_price': 12.0, 'volatility': 0.032, 'trend': 0.00006}
    }
    
    params = crypto_params.get(symbol, {'initial_price': 1.0, 'volatility': 0.025, 'trend': 0.00005})
    initial_price = params['initial_price']
    volatility = params['volatility']
    trend = params['trend']
    
    # Generate price walk with crypto-specific characteristics
    changes = np.random.normal(trend, volatility, periods)
    
    # Apply volatility clustering and cycles
    vol_multiplier = 1 + 0.25 * np.sin(np.arange(periods) / 2000)  # Longer cycles
    changes = changes * vol_multiplier
    
    # Calculate prices
    prices = [initial_price]
    for change in changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.001))  # Minimum price floor
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    all_data = []
    
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        # Realistic intraday movement
        open_price = close_price * (1 + np.random.normal(0, 0.002))
        high = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.004)))
        low = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.004)))
        
        # Volume with daily patterns
        hour_of_day = date.hour
        volume_factor = 1 + 0.3 * np.sin(2 * np.pi * hour_of_day / 24)
        volume = np.random.lognormal(6, 1.0) * volume_factor
        
        all_data.append({
            'date': date,
            'tic': symbol,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(all_data)
    print(f"✅ Created {symbol} synthetic data: {len(df):,} records")
    return df

def train_individual_crypto_gpu_synthetic(symbol, device):
    """Train GPU model on individual cryptocurrency using synthetic data"""
    
    print(f"\n🚀 TRAINING GPU MODEL FOR {symbol} (SYNTHETIC)")
    print("="*60)
    
    # Create synthetic data
    crypto_df = create_synthetic_crypto_data(symbol)
    
    # Add technical indicators
    print("🔧 Adding technical indicators...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30'],
        use_vix=False,
        use_turbulence=False
    )
    
    processed_df = fe.preprocess_data(crypto_df)
    print(f"✅ Processed data shape: {processed_df.shape}")
    
    # Split data (80% train, 20% test)
    split_point = int(len(processed_df) * 0.8)
    train_df = processed_df.iloc[:split_point].copy()
    test_df = processed_df.iloc[split_point:].copy()
    
    print(f"📊 Train data: {len(train_df):,} records")
    print(f"📊 Test data: {len(test_df):,} records")
    
    # Environment parameters (single asset)
    stock_dim = 1  # Single cryptocurrency
    tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']
    state_space = 1 + stock_dim + stock_dim + len(tech_indicators) * stock_dim
    action_space = stock_dim
    
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "buy_cost_pct": [0.001],  # Single asset
        "sell_cost_pct": [0.001],
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": action_space,
        "tech_indicator_list": tech_indicators,
        "stock_dim": stock_dim,
        "num_stock_shares": [0],
    }
    
    print(f"🔧 Environment setup for {symbol}:")
    print(f"   🎯 State space: {state_space}")
    print(f"   🎮 Action space: {action_space}")
    print(f"   🚀 Device: {device}")
    
    # Create environments
    train_env = DummyVecEnv([lambda: StockTradingEnv(df=train_df, **env_kwargs)])
    test_env = DummyVecEnv([lambda: StockTradingEnv(df=test_df, **env_kwargs)])
    
    # Create PPO model with GPU optimization
    print(f"🚀 Training PPO model for {symbol} on {device}...")
    if device.type == "mps":
        print("   🍎 Using Apple Silicon GPU acceleration!")
    
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=2e-4,  # GPU optimized
        n_steps=2048,
        batch_size=256,      # Larger batch for GPU
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device=device
    )
    
    print(f"   📊 Model device: {next(model.policy.parameters()).device}")
    
    # Training
    training_timesteps = 80_000  # Optimized for individual crypto
    print(f"   ⏱️  Training timesteps: {training_timesteps:,}")
    start_time = datetime.now()
    
    model.learn(total_timesteps=training_timesteps)
    
    training_duration = datetime.now() - start_time
    print(f"⏱️  Training completed in: {training_duration}")
    
    # Save model
    symbol_clean = symbol.replace('USDT', '').lower()
    model_filename = f"crypto_{symbol_clean}_gpu_synthetic_model"
    model.save(model_filename)
    print(f"💾 Model saved: {model_filename}.zip")
    
    # Test model
    print(f"🧪 Testing {symbol} model...")
    obs = test_env.reset()
    total_reward = 0.0
    portfolio_values = []
    steps = 0
    
    test_start = datetime.now()
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
            current_reward = float(reward) if hasattr(reward, '__float__') else float(reward[0]) if hasattr(reward, '__getitem__') else 0
            print(f"   Step {steps:,}, Current Reward: {current_reward:.4f}, Total: {total_reward:.2f}")
        
        if done:
            break
    
    test_duration = datetime.now() - test_start
    
    # Calculate performance metrics
    performance = {
        'symbol': symbol,
        'model_file': f"{model_filename}.zip",
        'training_duration': training_duration,
        'test_duration': test_duration,
        'total_reward': total_reward,
        'steps': steps,
        'portfolio_values': portfolio_values,
        'final_info': info[0] if info else {}
    }
    
    return performance

def generate_individual_report(performance):
    """Generate performance report for individual cryptocurrency"""
    
    symbol = performance['symbol']
    
    print(f"\n" + "="*70)
    print(f"📊 {symbol} PERFORMANCE REPORT")
    print("="*70)
    
    print(f"🤖 Model: {performance['model_file']}")
    print(f"⏱️  Training Time: {performance['training_duration']}")
    print(f"⏱️  Testing Time: {performance['test_duration']}")
    print(f"💰 Initial Capital: $1,000,000")
    print(f"📊 Total Steps: {performance['steps']:,}")
    print(f"🎯 Cumulative Reward: ${performance['total_reward']:,.2f}")
    
    # Final info
    final_info = performance['final_info']
    if final_info:
        print(f"\n📋 Final Trading Results for {symbol}:")
        for key, value in final_info.items():
            if isinstance(value, (int, float)):
                if 'asset' in key.lower() or 'balance' in key.lower():
                    print(f"   {key}: ${value:,.2f}")
                else:
                    print(f"   {key}: {value:.4f}")
    
    # Portfolio analysis
    portfolio_values = performance['portfolio_values']
    if portfolio_values and len(portfolio_values) > 1:
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        print(f"\n💼 {symbol} PORTFOLIO PERFORMANCE:")
        print(f"   💵 Initial Value: ${initial_value:,.2f}")
        print(f"   💰 Final Value: ${final_value:,.2f}")
        print(f"   📈 Total Return: {total_return:+.2f}%")
        print(f"   🔥 Max Value: ${np.max(portfolio_values):,.2f}")
        print(f"   📉 Min Value: ${np.min(portfolio_values):,.2f}")
        print(f"   💎 Profit/Loss: ${final_value - initial_value:+,.2f}")
        
        # Profitability assessment
        if total_return > 10:
            print(f"   🔥 HIGHLY PROFITABLE: {symbol} generated excellent returns!")
        elif total_return > 2:
            print(f"   ✅ PROFITABLE: {symbol} beat basic investment")
        elif total_return > 0:
            print(f"   ✅ POSITIVE: {symbol} generated modest profits")
        else:
            print(f"   ❌ LOSS: {symbol} lost money")
        
        # Calculate Sharpe ratio
        if len(portfolio_values) > 1:
            returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                      for i in range(1, len(portfolio_values))]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(288)
                print(f"   📐 Sharpe Ratio: {sharpe:.3f}")
                if sharpe > 1.0:
                    print(f"     ✅ EXCELLENT risk-adjusted returns")
                elif sharpe > 0.5:
                    print(f"     ✅ GOOD risk-adjusted returns")
                else:
                    print(f"     ⚠️ MODERATE risk-adjusted returns")
        else:
            sharpe = 0
    else:
        total_return = 0
        sharpe = 0
    
    return {
        'symbol': symbol,
        'return_pct': total_return,
        'profitable': total_return > 0,
        'sharpe_ratio': sharpe,
        'final_value': portfolio_values[-1] if portfolio_values else 1000000
    }

def main():
    """Main execution for training 5 cryptocurrencies with synthetic data"""
    
    print("🚀 CRYPTOCURRENCY GPU TRAINING - 5 SYNTHETIC MODELS")
    print("="*70)
    
    # Set PyTorch optimizations
    device = check_device_availability()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except:
            pass
    
    # Define cryptocurrencies to train
    symbols = ['ADAUSDT', 'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT']
    print(f"\n💰 Training individual models for: {symbols}")
    
    results = []
    
    # Train each cryptocurrency individually
    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'='*20} [{i}/{len(symbols)}] {symbol} {'='*20}")
        
        try:
            performance = train_individual_crypto_gpu_synthetic(symbol, device)
            summary = generate_individual_report(performance)
            results.append(summary)
            
            print(f"✅ {symbol} training completed successfully!")
            
        except Exception as e:
            print(f"❌ Error training {symbol}: {e}")
            results.append({
                'symbol': symbol,
                'return_pct': 0,
                'profitable': False,
                'sharpe_ratio': 0,
                'error': str(e),
                'final_value': 1000000
            })
            continue
    
    # Generate summary report
    print(f"\n" + "="*80)
    print("🏆 FINAL SUMMARY - ALL 5 CRYPTOCURRENCIES")
    print("="*80)
    
    profitable_count = sum(1 for r in results if r.get('profitable', False))
    total_return_avg = np.mean([r.get('return_pct', 0) for r in results if 'error' not in r])
    
    print(f"📊 Models Trained: {len(results)}")
    print(f"✅ Profitable Models: {profitable_count}/{len(results)}")
    print(f"💰 Success Rate: {profitable_count/len(results)*100:.1f}%")
    print(f"📈 Average Return: {total_return_avg:+.2f}%")
    
    print(f"\n📈 Individual Performance Rankings:")
    # Sort by return percentage
    sorted_results = sorted([r for r in results if 'error' not in r], 
                          key=lambda x: x.get('return_pct', 0), reverse=True)
    
    for rank, result in enumerate(sorted_results, 1):
        symbol = result['symbol']
        return_pct = result['return_pct']
        status = "✅ PROFIT" if result['profitable'] else "❌ LOSS"
        sharpe = result.get('sharpe_ratio', 0)
        final_val = result.get('final_value', 1000000)
        profit_loss = final_val - 1000000
        
        print(f"   {rank}. {symbol}: {status} ({return_pct:+.2f}%, ${profit_loss:+,.0f}, Sharpe: {sharpe:.2f})")
    
    # Failed models
    failed_results = [r for r in results if 'error' in r]
    for result in failed_results:
        symbol = result['symbol'] 
        print(f"   ❌ {symbol}: FAILED - {result['error'][:50]}...")
    
    # Best performer
    if sorted_results:
        best = sorted_results[0]
        worst = sorted_results[-1]
        print(f"\n🏆 BEST PERFORMER: {best['symbol']} ({best['return_pct']:+.2f}%)")
        print(f"📉 WORST PERFORMER: {worst['symbol']} ({worst['return_pct']:+.2f}%)")
    
    print("="*80)
    print("🎉 All 5 cryptocurrency models training completed!")
    print("💾 Models saved individually for each crypto")

if __name__ == "__main__":
    main()