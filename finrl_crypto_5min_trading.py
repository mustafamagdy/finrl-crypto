import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import os

def load_crypto_5min_data(filename='crypto_5min_2years.csv'):
    """Load 5-minute crypto data"""
    
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found!")
        print("Please wait for the download to complete first.")
        return None
    
    print("Loading 5-minute crypto data...")
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date and ticker
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    print(f"✅ Loaded data shape: {df.shape}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"💰 Crypto symbols: {sorted(df['tic'].unique())}")
    print(f"⏰ Total time span: {(df['date'].max() - df['date'].min()).days} days")
    print(f"📊 Records per symbol: {len(df) // len(df['tic'].unique()):,}")
    
    return df

def prepare_crypto_data_for_training(df, add_indicators=True):
    """Prepare crypto data for FinRL training"""
    
    if add_indicators:
        print("Adding technical indicators...")
        
        # Use crypto-friendly technical indicators
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=['rsi_30', 'macd', 'cci_30', 'dx_30'],
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False
        )
        
        try:
            processed_df = fe.preprocess_data(df)
            print(f"✅ Technical indicators added successfully")
            print(f"📊 Data shape after indicators: {processed_df.shape}")
            return processed_df
        except Exception as e:
            print(f"⚠️ Error adding technical indicators: {e}")
            print("Proceeding without technical indicators...")
            return df
    else:
        return df

def create_train_test_split(df, test_size=0.2):
    """Create train/test split maintaining temporal order"""
    
    # Get unique dates and split them
    unique_dates = sorted(df['date'].unique())
    split_idx = int(len(unique_dates) * (1 - test_size))
    split_date = unique_dates[split_idx]
    
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()
    
    print(f"\n📊 Data Split Summary:")
    print(f"  Training: {len(train_df):,} records ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  Testing:  {len(test_df):,} records ({test_df['date'].min()} to {test_df['date'].max()})")
    print(f"  Split ratio: {len(train_df)/len(df)*100:.1f}% train, {len(test_df)/len(df)*100:.1f}% test")
    
    return train_df, test_df

def setup_trading_environment(df, symbols):
    """Setup FinRL trading environment for crypto"""
    
    stock_dim = len(symbols)
    
    # Determine if we have technical indicators
    tech_cols = [col for col in df.columns if col in ['rsi_30', 'macd', 'cci_30', 'dx_30']]
    
    if tech_cols:
        # With technical indicators
        state_space = 1 + stock_dim + stock_dim + len(tech_cols) * stock_dim
        tech_indicator_list = tech_cols
        print(f"🔧 Environment with technical indicators: {tech_cols}")
    else:
        # Without technical indicators
        state_space = 1 + stock_dim + stock_dim
        tech_indicator_list = []
        print(f"🔧 Environment without technical indicators")
    
    env_kwargs = {
        "stock_dim": stock_dim,
        "hmax": 100,  # Max shares per asset
        "initial_amount": 1_000_000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,  # 0.1% trading fee
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": tech_indicator_list,
    }
    
    print(f"🎯 State space: {state_space} dimensions")
    print(f"🎮 Action space: {stock_dim} dimensions")
    
    return env_kwargs

def train_crypto_model_5min(train_df, env_kwargs, timesteps=100000):
    """Train PPO model on 5-minute crypto data"""
    
    print(f"\n🚀 Training PPO model...")
    print(f"⏱️  Training timesteps: {timesteps:,}")
    print(f"💾 Training data size: {len(train_df):,} records")
    
    # Create training environment
    train_env = DummyVecEnv([lambda: StockTradingEnv(df=train_df, **env_kwargs)])
    
    # Initialize PPO model with optimized hyperparameters for crypto
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./crypto_5min_ppo_logs/"
    )
    
    # Train the model
    model.learn(total_timesteps=timesteps)
    
    # Save the model
    model_filename = "crypto_5min_ppo_model"
    model.save(model_filename)
    print(f"💾 Model saved as: {model_filename}.zip")
    
    return model

def test_crypto_model_5min(model, test_df, env_kwargs):
    """Test the trained model on 5-minute data"""
    
    print(f"\n🧪 Testing model...")
    print(f"📊 Test data size: {len(test_df):,} records")
    
    # Create test environment
    test_env = DummyVecEnv([lambda: StockTradingEnv(df=test_df, **env_kwargs)])
    
    # Reset environment and run episode
    obs = test_env.reset()
    total_reward = 0.0
    actions_log = []
    rewards_log = []
    portfolio_values = []
    
    step = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        
        total_reward += float(reward)
        actions_log.append(action[0].copy() if hasattr(action[0], 'copy') else action[0])
        rewards_log.append(float(reward))
        
        # Track portfolio value
        if info and len(info) > 0:
            portfolio_value = info[0].get('total_asset', info[0].get('portfolio_value', None))
            if portfolio_value is not None:
                portfolio_values.append(float(portfolio_value))
        
        step += 1
        if step % 1000 == 0:  # Progress update
            print(f"  Step {step:,}, Reward: {reward:.4f}, Total: {total_reward:.2f}")
        
        if done:
            break
    
    print(f"✅ Testing completed!")
    print(f"📈 Total steps: {step:,}")
    print(f"💰 Cumulative reward: {total_reward:.2f}")
    
    return {
        'total_reward': total_reward,
        'final_info': info[0] if info and len(info) > 0 else {},
        'actions': actions_log,
        'rewards': rewards_log,
        'portfolio_values': portfolio_values,
        'steps': step
    }

def generate_5min_performance_report(results, symbols, initial_amount=1000000):
    """Generate detailed performance report for 5-minute trading"""
    
    print("\n" + "="*80)
    print("🚀 CRYPTO 5-MINUTE TRADING PERFORMANCE REPORT")
    print("="*80)
    
    # Basic info
    print(f"💰 Symbols Traded: {symbols}")
    print(f"💵 Initial Capital: ${initial_amount:,}")
    print(f"⏱️  Total Trading Steps: {results['steps']:,}")
    print(f"📊 Cumulative Reward: ${results['total_reward']:,.2f}")
    
    # Final session info
    final_info = results['final_info']
    if final_info:
        print(f"\n📋 Final Trading Session:")
        for key, value in final_info.items():
            if isinstance(value, (int, float)):
                if 'asset' in key.lower() or 'balance' in key.lower() or 'value' in key.lower():
                    print(f"  {key}: ${value:,.2f}")
                else:
                    print(f"  {key}: {value:,.2f}")
    
    # Reward analysis
    rewards = results['rewards']
    if rewards:
        positive_rewards = [r for r in rewards if r > 0]
        negative_rewards = [r for r in rewards if r < 0]
        
        print(f"\n📈 Reward Analysis:")
        print(f"  Average Reward/Step: ${np.mean(rewards):.6f}")
        print(f"  Reward Std Dev: ${np.std(rewards):.6f}")
        print(f"  Max Single Reward: ${np.max(rewards):.4f}")
        print(f"  Min Single Reward: ${np.min(rewards):.4f}")
        print(f"  Profitable Steps: {len(positive_rewards):,} ({len(positive_rewards)/len(rewards)*100:.1f}%)")
        print(f"  Losing Steps: {len(negative_rewards):,} ({len(negative_rewards)/len(rewards)*100:.1f}%)")
    
    # Portfolio performance
    portfolio_values = [pv for pv in results['portfolio_values'] if pv is not None]
    if portfolio_values and len(portfolio_values) > 1:
        final_value = portfolio_values[-1]
        initial_value = portfolio_values[0]
        total_return = (final_value - initial_value) / initial_value * 100
        
        print(f"\n💼 Portfolio Performance:")
        print(f"  Initial Value: ${initial_value:,.2f}")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")
        print(f"  Max Value: ${np.max(portfolio_values):,.2f}")
        print(f"  Min Value: ${np.min(portfolio_values):,.2f}")
        
        # Calculate max drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
    
    # Trading activity
    actions = results['actions']
    if actions:
        # Count active trading actions
        active_trades = sum(1 for action in actions if np.any(np.abs(action) > 0.01))
        
        print(f"\n🔄 Trading Activity:")
        print(f"  Total Decision Points: {len(actions):,}")
        print(f"  Active Trades: {active_trades:,} ({active_trades/len(actions)*100:.1f}%)")
        print(f"  Hold Periods: {len(actions)-active_trades:,} ({(len(actions)-active_trades)/len(actions)*100:.1f}%)")
    
    print("="*80)

def main():
    """Main execution for 5-minute crypto trading"""
    
    print("🚀 FinRL Crypto Trading - 5 Minute Data")
    print("="*50)
    
    # Load 5-minute data
    crypto_df = load_crypto_5min_data('crypto_5min_2years.csv')
    
    if crypto_df is None:
        print("❌ Cannot proceed without data. Please run the download script first.")
        return
    
    # Prepare data (try with indicators, fallback to without)
    processed_df = prepare_crypto_data_for_training(crypto_df, add_indicators=True)
    
    # Create train/test split
    train_df, test_df = create_train_test_split(processed_df, test_size=0.2)
    
    # Get symbols
    symbols = sorted(crypto_df['tic'].unique())
    
    # Setup environment
    env_kwargs = setup_trading_environment(processed_df, symbols)
    
    # Train model
    print(f"\n🔥 Starting training...")
    model = train_crypto_model_5min(train_df, env_kwargs, timesteps=100000)
    
    # Test model
    print(f"\n🧪 Starting testing...")
    results = test_crypto_model_5min(model, test_df, env_kwargs)
    
    # Generate report
    generate_5min_performance_report(results, symbols)
    
    print(f"\n✅ 5-minute crypto trading experiment completed!")
    print(f"📁 Check crypto_5min_ppo_model.zip for the trained model")
    print(f"📊 Check ./crypto_5min_ppo_logs/ for TensorBoard logs")

if __name__ == "__main__":
    main()