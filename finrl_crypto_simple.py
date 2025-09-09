import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime

def load_and_prepare_crypto_data(filename='crypto_test_data.csv'):
    """Load and prepare crypto data for FinRL"""
    
    print("Loading crypto data...")
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date and ticker
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Crypto symbols: {df['tic'].unique()}")
    
    return df

def create_crypto_environment(df, symbols, initial_amount=1000000):
    """Create FinRL environment for crypto trading (no technical indicators)"""
    
    stock_dim = len(symbols)
    # Simple state space: balance + stock prices + shares (no technical indicators)
    state_space = 1 + stock_dim + stock_dim
    action_space = stock_dim
    
    env_kwargs = {
        "stock_dim": stock_dim,
        "hmax": 100,  # Maximum shares to hold per asset
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,  # 0.1% trading fee
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": action_space,
        "tech_indicator_list": [],  # No technical indicators for now
    }
    
    return env_kwargs

def train_crypto_model(train_df, env_kwargs, timesteps=20000):
    """Train PPO model on crypto data"""
    
    print(f"Training PPO model for {timesteps} timesteps...")
    
    # Create training environment
    train_env = DummyVecEnv([lambda: StockTradingEnv(df=train_df, **env_kwargs)])
    
    # Initialize PPO model
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=3e-4)
    
    # Train the model
    model.learn(total_timesteps=timesteps)
    
    # Save the model
    model.save("crypto_ppo_simple")
    
    return model

def test_crypto_model(model, test_df, env_kwargs):
    """Test the trained model"""
    
    print("Testing trained model...")
    
    # Create test environment
    test_env = DummyVecEnv([lambda: StockTradingEnv(df=test_df, **env_kwargs)])
    
    # Reset environment and run episode
    obs = test_env.reset()
    total_reward = 0.0
    actions_taken = []
    rewards_received = []
    portfolio_values = []
    
    step = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        
        total_reward += float(reward)
        actions_taken.append(action[0].copy() if hasattr(action[0], 'copy') else action[0])
        rewards_received.append(float(reward))
        
        # Try to get portfolio value from info
        if info and len(info) > 0:
            portfolio_value = info[0].get('total_asset', info[0].get('portfolio_value', None))
            if portfolio_value is not None:
                portfolio_values.append(float(portfolio_value))
        
        step += 1
        if done:
            break
    
    print(f"Testing completed after {step} steps")
    print(f"Total cumulative reward: {total_reward:.2f}")
    
    # Get final results from info
    final_info = info[0] if info and len(info) > 0 else {}
    
    return {
        'total_reward': total_reward,
        'final_info': final_info,
        'actions': actions_taken,
        'rewards': rewards_received,
        'portfolio_values': portfolio_values,
        'steps': step
    }

def generate_performance_report(results, symbols, initial_amount=1000000):
    """Generate comprehensive performance report"""
    
    print("\n" + "="*60)
    print("CRYPTO TRADING PERFORMANCE REPORT")
    print("="*60)
    
    # Basic metrics
    print(f"Trading Symbols: {symbols}")
    print(f"Initial Amount: ${initial_amount:,.2f}")
    print(f"Total Cumulative Reward: {results['total_reward']:,.2f}")
    print(f"Total Steps: {results['steps']}")
    
    # Extract info from final info
    final_info = results['final_info']
    if final_info:
        print("\nFinal Trading Session Info:")
        for key, value in final_info.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:,.2f}")
            else:
                print(f"  {key}: {value}")
    
    # Calculate additional metrics
    rewards = results['rewards']
    if rewards:
        print(f"\nReward Statistics:")
        print(f"  Average Reward per Step: {np.mean(rewards):.4f}")
        print(f"  Reward Std Dev: {np.std(rewards):.4f}")
        print(f"  Max Single Reward: {np.max(rewards):.4f}")
        print(f"  Min Single Reward: {np.min(rewards):.4f}")
        
        # Count profitable vs losing steps
        profitable_steps = sum(1 for r in rewards if r > 0)
        print(f"  Profitable Steps: {profitable_steps}/{len(rewards)} ({profitable_steps/len(rewards)*100:.1f}%)")
    
    # Portfolio value analysis
    portfolio_values = [pv for pv in results['portfolio_values'] if pv is not None]
    if portfolio_values:
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_amount) / initial_amount * 100
        max_portfolio = np.max(portfolio_values)
        min_portfolio = np.min(portfolio_values)
        
        print(f"\nPortfolio Performance:")
        print(f"  Final Portfolio Value: ${final_value:,.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Max Portfolio Value: ${max_portfolio:,.2f}")
        print(f"  Min Portfolio Value: ${min_portfolio:,.2f}")
        
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
    
    # Action analysis
    actions = results['actions']
    if actions:
        print(f"\nTrading Activity:")
        total_actions = len(actions)
        # Count non-zero actions (actual trading)
        active_actions = sum(1 for action in actions if np.any(np.abs(action) > 0.01))
        print(f"  Total Trading Steps: {total_actions}")
        print(f"  Active Trading Steps: {active_actions} ({active_actions/total_actions*100:.1f}%)")

def main():
    """Main execution function"""
    
    print("Starting Simple Crypto Trading with FinRL")
    print("="*50)
    
    # Load and prepare data
    crypto_df = load_and_prepare_crypto_data('crypto_test_data.csv')
    
    # Split data (80% train, 20% test)
    split_point = int(len(crypto_df) * 0.8)
    # Make sure we split by maintaining symbol groupings
    unique_dates = sorted(crypto_df['date'].unique())
    split_date_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_date_idx]
    
    train_df = crypto_df[crypto_df['date'] <= split_date].copy()
    test_df = crypto_df[crypto_df['date'] > split_date].copy()
    
    print(f"\nData split:")
    print(f"  Training data: {len(train_df)} records ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  Test data: {len(test_df)} records ({test_df['date'].min()} to {test_df['date'].max()})")
    
    # Get unique symbols
    symbols = sorted(crypto_df['tic'].unique())
    print(f"  Crypto symbols: {symbols}")
    
    # Create environment
    env_kwargs = create_crypto_environment(crypto_df, symbols, initial_amount=1000000)
    print(f"  State space: {env_kwargs['state_space']}")
    print(f"  Action space: {env_kwargs['action_space']}")
    
    # Train model
    model = train_crypto_model(train_df, env_kwargs, timesteps=20000)
    
    # Test model
    results = test_crypto_model(model, test_df, env_kwargs)
    
    # Generate report
    generate_performance_report(results, symbols, initial_amount=1000000)
    
    print("\n" + "="*60)
    print("Crypto trading experiment completed!")

if __name__ == "__main__":
    main()