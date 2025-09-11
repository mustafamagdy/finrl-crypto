import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

# Import our patched StockTradingEnv
from finrl_patch import PatchedStockTradingEnv

def quick_btc_training():
    """Quick BTC training to get real performance data"""
    print("🚀 QUICK BTC TRAINING - REAL DATA")
    print("="*50)
    
    # Load data
    df = pd.read_csv('crypto_5min_2years.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for BTC only
    btc_df = df[df['tic'] == 'BTCUSDT'].copy().reset_index(drop=True)
    print(f"📊 BTC Dataset: {len(btc_df):,} records")
    print(f"📅 Date range: {btc_df['date'].min()} to {btc_df['date'].max()}")
    
    # Calculate actual returns
    start_price = btc_df['close'].iloc[0]
    end_price = btc_df['close'].iloc[-1]
    actual_return = ((end_price - start_price) / start_price) * 100
    print(f"💰 BTC actual return: {actual_return:+.2f}%")
    
    # Feature engineering
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30'],
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False
    )
    
    processed_df = fe.preprocess_data(btc_df)
    print(f"📊 Features added: {len(processed_df.columns)} columns")
    
    # Train/test split (80/20)
    split_point = int(len(processed_df) * 0.8)
    train_df = processed_df.iloc[:split_point].copy()
    test_df = processed_df.iloc[split_point:].copy()
    
    print(f"📊 Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Environment configuration for single asset (BTC)
    state_space = 1 + 1 + 1 + len(['macd', 'rsi_30', 'cci_30', 'dx_30'])
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "buy_cost_pct": [0.001],
        "sell_cost_pct": [0.001],
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": 1,
        "tech_indicator_list": ['macd', 'rsi_30', 'cci_30', 'dx_30'],
        "stock_dim": 1,
        "num_stock_shares": [0],
    }
    
    # Create training environment
    train_env = DummyVecEnv([lambda: PatchedStockTradingEnv(df=train_df, **env_kwargs)])
    
    # Device setup
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🚀 Training BTC on {device}")
    
    # PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,  # Smaller for faster training
        batch_size=64,
        n_epochs=5,    # Fewer epochs for speed
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        device=device
    )
    
    start_time = datetime.now()
    print("🎯 Starting quick BTC training...")
    
    # Quick training (reduced timesteps)
    model.learn(total_timesteps=50_000)
    training_time = datetime.now() - start_time
    
    # Save model
    model_name = "btc_quick_model"
    model.save(model_name)
    print(f"💾 BTC model saved: {model_name}")
    
    # Testing phase
    test_env = DummyVecEnv([lambda: PatchedStockTradingEnv(df=test_df, **env_kwargs)])
    
    print("🧪 Testing BTC model...")
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
            actions_taken.append(int(action[0]) if hasattr(action, '__len__') else int(action))
        
        if done:
            break
    
    # Results analysis
    if portfolio_values:
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        algorithm_return = (final_value - initial_value) / initial_value * 100
        profit = final_value - initial_value
        
        # Performance metrics
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            sharpe = (algorithm_return / 100) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdowns = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdowns) * 100
        else:
            volatility = 0
            sharpe = 0
            max_drawdown = 0
        
        # Action analysis
        buy_count = sum(1 for a in actions_taken if a < 0)
        hold_count = sum(1 for a in actions_taken if a == 0)
        sell_count = sum(1 for a in actions_taken if a > 0)
        
        print(f"\n🏆 BTC QUICK TRAINING RESULTS:")
        print(f"💰 Algorithm Return: {algorithm_return:+.2f}%")
        print(f"💰 Actual BTC Return: {actual_return:+.2f}%")
        print(f"💵 Profit: ${profit:+,.0f}")
        print(f"🏦 Final Portfolio: ${final_value:,.0f}")
        print(f"📊 vs Buy&Hold: {algorithm_return - actual_return:+.2f}% difference")
        print(f"📈 Sharpe Ratio: {sharpe:.3f}")
        print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        print(f"🎯 Actions - Buy: {buy_count}, Hold: {hold_count}, Sell: {sell_count}")
        print(f"⏱️ Training Time: {training_time}")
        
        return {
            'symbol': 'BTCUSDT',
            'algorithm_return': algorithm_return,
            'actual_return': actual_return,
            'profit': profit,
            'final_value': final_value,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': volatility * 100,
            'training_time': str(training_time),
            'model_name': model_name,
            'actions': {'buy': buy_count, 'hold': hold_count, 'sell': sell_count},
            'steps': steps,
            'portfolio_values': portfolio_values
        }
    else:
        print("❌ BTC: No portfolio values recorded")
        return None

if __name__ == "__main__":
    results = quick_btc_training()
    if results:
        print(f"\n✅ BTC Quick Training Complete!")
        print(f"Algorithm: {results['algorithm_return']:+.2f}% vs Actual: {results['actual_return']:+.2f}%")