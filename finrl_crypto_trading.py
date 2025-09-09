import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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

def add_technical_indicators(df):
    """Add technical indicators using FinRL's FeatureEngineer"""
    
    print("Adding technical indicators...")
    
    # Use a smaller set of indicators for crypto (they tend to be more volatile)
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=['rsi_30', 'macd', 'cci_30', 'dx_30'],
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False
    )
    
    processed_df = fe.preprocess_data(df)
    
    print(f"Data with indicators shape: {processed_df.shape}")
    print("Technical indicators added successfully")
    
    return processed_df

def create_crypto_environment(df, symbols, initial_amount=1000000):
    """Create FinRL environment for crypto trading"""
    
    stock_dim = len(symbols)
    tech_indicators = ['rsi_30', 'macd', 'cci_30', 'dx_30']
    
    # Calculate state space: balance + stock prices + shares + indicators
    state_space = 1 + stock_dim + stock_dim + len(tech_indicators) * stock_dim
    action_space = stock_dim
    
    env_kwargs = {
        "stock_dim": stock_dim,
        "hmax": 100,  # Maximum shares to hold per asset
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,  # 0.1% trading fee (typical for crypto)
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": action_space,
        "tech_indicator_list": tech_indicators,
    }
    
    return env_kwargs

def train_crypto_model(train_df, env_kwargs, timesteps=50000):
    """Train PPO model on crypto data"""
    
    print(f"Training PPO model for {timesteps} timesteps...")
    
    # Create training environment
    train_env = DummyVecEnv([lambda: StockTradingEnv(df=train_df, **env_kwargs)])
    
    # Initialize PPO model
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./crypto_ppo_logs/")
    
    # Train the model
    model.learn(total_timesteps=timesteps)
    
    # Save the model
    model.save("crypto_ppo_model")
    
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
        actions_taken.append(action[0])
        rewards_received.append(float(reward))
        
        # Try to get portfolio value from info
        if info and len(info) > 0:
            portfolio_value = info[0].get('total_asset', info[0].get('portfolio_value', None))
            portfolio_values.append(portfolio_value)
        
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

def generate_performance_report(results, initial_amount=1000000):
    """Generate comprehensive performance report"""
    
    print("\n" + "="*60)
    print("CRYPTO TRADING PERFORMANCE REPORT")
    print("="*60)
    
    # Basic metrics
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
    
    # Portfolio value analysis
    portfolio_values = [pv for pv in results['portfolio_values'] if pv is not None]
    if portfolio_values:
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_amount) / initial_amount * 100
        print(f"\nPortfolio Performance:")
        print(f"  Final Portfolio Value: ${final_value:,.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Max Portfolio Value: ${np.max(portfolio_values):,.2f}")
        print(f"  Min Portfolio Value: ${np.min(portfolio_values):,.2f}")

def main():
    """Main execution function"""
    
    print("Starting Crypto Trading with FinRL")
    print("="*50)
    
    # Load and prepare data
    crypto_df = load_and_prepare_crypto_data('crypto_test_data.csv')
    
    # Add technical indicators
    processed_df = add_technical_indicators(crypto_df)
    
    # Split data (80% train, 20% test)
    split_date = processed_df['date'].quantile(0.8)
    train_df = processed_df[processed_df['date'] <= split_date]
    test_df = processed_df[processed_df['date'] > split_date]
    
    print(f"\nData split:")
    print(f"  Training data: {len(train_df)} records ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  Test data: {len(test_df)} records ({test_df['date'].min()} to {test_df['date'].max()})")
    
    # Get unique symbols
    symbols = sorted(crypto_df['tic'].unique())
    print(f"  Crypto symbols: {symbols}")
    
    # Create environment
    env_kwargs = create_crypto_environment(processed_df, symbols, initial_amount=1000000)
    
    # Train model
    model = train_crypto_model(train_df, env_kwargs, timesteps=30000)
    
    # Test model
    results = test_crypto_model(model, test_df, env_kwargs)
    
    # Generate report
    generate_performance_report(results, initial_amount=1000000)
    
    print("\n" + "="*60)
    print("Crypto trading experiment completed!")

if __name__ == "__main__":
    main()