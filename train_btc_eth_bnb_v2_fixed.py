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

def main():
    print("🚀 TRAINING BTC+ETH+BNB MODEL v2 - FIXED WITH PATCH")
    print("="*80)
    
    # Load BTC+ETH+BNB data
    df = pd.read_csv('crypto_5min_2years.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"📊 Dataset loaded: {len(df):,} records")
    print(f"🪙 Cryptocurrencies: {sorted(df['tic'].unique())}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Detailed data analysis
    for symbol in sorted(df['tic'].unique()):
        symbol_df = df[df['tic'] == symbol]
        start_price = symbol_df['close'].iloc[0]
        end_price = symbol_df['close'].iloc[-1]
        total_return = ((end_price - start_price) / start_price) * 100
        print(f"   {symbol}: {len(symbol_df):,} records, {total_return:+.2f}% return")
    
    # Feature engineering
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30'],
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False
    )
    
    processed_df = fe.preprocess_data(df)
    print(f"📊 Features added: {len(processed_df.columns)} columns")
    
    # Train/test split (80/20)
    split_point = int(len(processed_df) * 0.8)
    train_df = processed_df.iloc[:split_point].copy()
    test_df = processed_df.iloc[split_point:].copy()
    
    print(f"📊 Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Environment configuration optimized for multi-asset trading
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "buy_cost_pct": [0.001, 0.001, 0.001],  # Low transaction costs for 3 assets
        "sell_cost_pct": [0.001, 0.001, 0.001],
        "reward_scaling": 1e-4,
        "state_space": len(processed_df.columns) - 2,  # Exclude date and tic
        "action_space": 3,  # Actions for 3 cryptocurrencies
        "tech_indicator_list": ['macd', 'rsi_30', 'cci_30', 'dx_30'],
        "stock_dim": 3,     # 3 cryptocurrencies
        "num_stock_shares": [0, 0, 0],  # Start with 0 shares
    }
    
    # Create training environment using PATCHED version
    train_env = DummyVecEnv([lambda: PatchedStockTradingEnv(df=train_df, **env_kwargs)])
    
    # PPO model optimized for multi-crypto trending markets
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🚀 Training on {device}")
    
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,     # Standard learning rate
        n_steps=2048,           # Standard rollout
        batch_size=64,          # Good balance for 3-asset portfolio
        n_epochs=10,            # Standard epochs
        gamma=0.99,             # High discount for long-term trends
        gae_lambda=0.95,        # Standard GAE
        clip_range=0.2,         # Standard PPO clipping
        ent_coef=0.01,          # Moderate exploration
        max_grad_norm=0.5,      # Gradient clipping
        device=device
    )
    
    start_time = datetime.now()
    print("🎯 Starting optimized training for BTC+ETH+BNB (FIXED VERSION)...")
    
    # Extended training for better convergence
    model.learn(total_timesteps=200_000)
    training_time = datetime.now() - start_time
    
    # Save model with new version
    model_name = "crypto_btc_eth_bnb_v2_fixed_model"
    model.save(model_name)
    print(f"💾 BTC+ETH+BNB v2 FIXED model saved as: {model_name}")
    
    # Testing phase
    test_env = DummyVecEnv([lambda: PatchedStockTradingEnv(df=test_df, **env_kwargs)])
    
    print("\\n🧪 TESTING BTC+ETH+BNB v2 FIXED MODEL...")
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
    
    # Results analysis
    if portfolio_values:
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        algorithm_return = (final_value - initial_value) / initial_value * 100
        profit = final_value - initial_value
        
        # Buy and hold comparison for each cryptocurrency
        test_symbols = sorted(test_df['tic'].unique())
        buy_hold_returns = []
        
        for symbol in test_symbols:
            symbol_data = test_df[test_df['tic'] == symbol]
            if len(symbol_data) > 0:
                start_price = symbol_data['close'].iloc[0]
                end_price = symbol_data['close'].iloc[-1]
                bh_return = (end_price - start_price) / start_price * 100
                buy_hold_returns.append((symbol, bh_return))
        
        avg_buy_hold = np.mean([ret for _, ret in buy_hold_returns])
        
        print(f"\\n🏆 BTC+ETH+BNB v2 FIXED MODEL RESULTS:")
        print(f"="*70)
        print(f"💰 Algorithm Return: {algorithm_return:+.2f}%")
        print(f"💵 Profit: ${profit:+,.0f}")
        print(f"🏦 Final Portfolio Value: ${final_value:,.0f}")
        print(f"📊 Trading Steps: {steps:,}")
        print(f"⏱️ Training Time: {training_time}")
        
        print(f"\\n📈 PERFORMANCE COMPARISON:")
        print(f"   Algorithm: {algorithm_return:+.2f}%")
        for symbol, bh_return in buy_hold_returns:
            print(f"   {symbol} Buy&Hold: {bh_return:+.2f}%")
        print(f"   Avg Buy&Hold: {avg_buy_hold:+.2f}%")
        print(f"   Algorithm vs Avg B&H: {algorithm_return - avg_buy_hold:+.2f}% difference")
        
        # Action analysis
        if actions_taken and len(actions_taken[0]) == 3:
            btc_actions = [a[0] for a in actions_taken]
            eth_actions = [a[1] for a in actions_taken]
            bnb_actions = [a[2] for a in actions_taken]
            
            print(f"\\n🎯 MULTI-ASSET ACTION ANALYSIS:")
            for i, (symbol, actions) in enumerate([('BTC', btc_actions), ('ETH', eth_actions), ('BNB', bnb_actions)]):
                buy_count = sum(1 for a in actions if a < 0)  # Negative = buy
                hold_count = sum(1 for a in actions if a == 0)
                sell_count = sum(1 for a in actions if a > 0)  # Positive = sell
                
                print(f"   {symbol}: Buy {buy_count} | Hold {hold_count} | Sell {sell_count}")
        
        # Performance metrics
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            sharpe = algorithm_return / (volatility * 100) if volatility > 0 else 0
            
            # Maximum drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdowns = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdowns) * 100
            
            print(f"\\n📊 RISK METRICS:")
            print(f"   Portfolio Volatility: {volatility*100:.2f}%")
            print(f"   Sharpe Ratio: {sharpe:.2f}")
            print(f"   Maximum Drawdown: {max_drawdown:.2f}%")
        
        print(f"\\n✅ BTC+ETH+BNB v2 FIXED model training completed successfully!")
        print(f"🔧 Model uses patched FinRL environment to avoid numpy.float64 bug")
        print(f"🎯 Model optimized for multi-cryptocurrency trending markets")
        print(f"🚀 Ready for deployment on BTC, ETH, and BNB trading")
        
        return model_name, algorithm_return, profit
    else:
        print("❌ No portfolio values recorded during testing")
        return model_name, 0, 0

if __name__ == "__main__":
    main()