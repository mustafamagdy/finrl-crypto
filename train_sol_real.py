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

def main():
    print("🚀 TRAINING SOLUSDT - REAL DATA - GPU")
    
    # Load real data
    df = pd.read_csv('crypto_5currencies_2years.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for SOLUSDT
    sol_df = df[df['tic'] == 'SOLUSDT'].copy().reset_index(drop=True)
    print(f"📊 SOLUSDT records: {len(sol_df):,}")
    
    # Skip FeatureEngineer to avoid filtering issues - use raw data
    processed_df = sol_df.copy()
    
    # Split data
    split_point = int(len(processed_df) * 0.8)
    train_df = processed_df.iloc[:split_point].copy()
    test_df = processed_df.iloc[split_point:].copy()
    
    print(f"📊 Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Simple environment (no tech indicators to avoid issues)
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "buy_cost_pct": [0.001],
        "sell_cost_pct": [0.001],
        "reward_scaling": 1e-4,
        "state_space": 3,  # balance + price + shares
        "action_space": 1,
        "tech_indicator_list": [],
        "stock_dim": 1,
        "num_stock_shares": [0],
    }
    
    # Train environment
    train_env = DummyVecEnv([lambda: StockTradingEnv(df=train_df, **env_kwargs)])
    
    # PPO model with GPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🚀 Training on {device}")
    
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        device=device
    )
    
    start_time = datetime.now()
    model.learn(total_timesteps=100_000)
    training_time = datetime.now() - start_time
    
    # Save model
    model.save("crypto_sol_real_gpu_model")
    print(f"💾 SOLUSDT model saved in {training_time}")
    
    # Test
    test_env = DummyVecEnv([lambda: StockTradingEnv(df=test_df, **env_kwargs)])
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
            pv = info[0].get('total_asset', 1000000)
            portfolio_values.append(float(pv))
        
        if done:
            break
    
    # Results
    if portfolio_values:
        initial = portfolio_values[0]
        final = portfolio_values[-1]
        return_pct = (final - initial) / initial * 100
        profit = final - initial
        
        print(f"\n🏆 SOLUSDT RESULTS:")
        print(f"   Return: {return_pct:+.2f}%")
        print(f"   Profit: ${profit:+,.0f}")
        print(f"   Final Value: ${final:,.0f}")
        print(f"   Steps: {steps:,}")
        print(f"   Training Time: {training_time}")
    
if __name__ == "__main__":
    main()