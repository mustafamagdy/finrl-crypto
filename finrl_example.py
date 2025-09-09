import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import pandas as pd
import yfinance as yf
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Download data
TICS = ["AAPL", "MSFT", "SPY"]
print("Downloading stock data...")
df = yf.download(TICS, start="2019-01-01", 
                 end=datetime.today().strftime("%Y-%m-%d"),
                 auto_adjust=True, progress=False)
df = df.stack(level=1).rename_axis(index=["date", "tic"]).reset_index()
df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
df = df[["date", "tic", "open", "high", "low", "close", "volume"]].sort_values(["date", "tic"])

print(f"Data shape: {df.shape}")

# Add required technical indicators
fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30'],
                     use_vix=False,
                     use_turbulence=False)

processed_df = fe.preprocess_data(df)

# Split data
train_df = data_split(processed_df, "2019-01-01", "2021-12-31")
test_df = data_split(processed_df, "2022-01-01", datetime.today().strftime("%Y-%m-%d"))

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Environment parameters
stock_dim = len(TICS)
# State space: account balance + stock prices + stock shares + tech indicators
tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']
state_space = 1 + stock_dim + stock_dim + len(tech_indicators) * stock_dim
action_space = stock_dim

env_kwargs = {
    "stock_dim": stock_dim,
    "hmax": 100,
    "initial_amount": 1_000_000,
    "num_stock_shares": [0] * stock_dim,
    "buy_cost_pct": [1e-3] * stock_dim,
    "sell_cost_pct": [1e-3] * stock_dim,
    "reward_scaling": 1e-4,
    "state_space": state_space,
    "action_space": action_space,
    "tech_indicator_list": ['macd', 'rsi_30', 'cci_30', 'dx_30'],
}

# Create environments
train_env = DummyVecEnv([lambda: StockTradingEnv(df=train_df, **env_kwargs)])
test_env = DummyVecEnv([lambda: StockTradingEnv(df=test_df, **env_kwargs)])

print("Training PPO model...")
model = PPO("MlpPolicy", train_env, verbose=0)
model.learn(total_timesteps=20_000)

print("Testing model...")
obs = test_env.reset()
total_reward = 0.0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    total_reward += float(reward)
    if done:
        break

print("Test cumulative reward:", round(total_reward, 2))
print("Final net worth:", info[0].get("final_balance_reward"))