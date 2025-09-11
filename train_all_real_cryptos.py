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

def train_single_crypto(symbol, df, results_dict):
    """Train a model on individual cryptocurrency using real data"""
    print(f"\n🚀 TRAINING {symbol} MODEL - REAL DATA WITH PATCH")
    print("="*80)
    
    # Filter data for this specific symbol
    symbol_df = df[df['tic'] == symbol].copy().reset_index(drop=True)
    
    if len(symbol_df) < 1000:
        print(f"❌ {symbol}: Insufficient data ({len(symbol_df)} records)")
        results_dict[symbol] = {
            'status': 'insufficient_data',
            'records': len(symbol_df),
            'error': 'Less than 1000 records'
        }
        return
    
    print(f"📊 {symbol} Dataset: {len(symbol_df):,} records")
    print(f"📅 Date range: {symbol_df['date'].min()} to {symbol_df['date'].max()}")
    
    # Calculate returns
    start_price = symbol_df['close'].iloc[0]
    end_price = symbol_df['close'].iloc[-1]
    total_return = ((end_price - start_price) / start_price) * 100
    print(f"💰 {symbol} 2-year return: {total_return:+.2f}%")
    
    try:
        # Feature engineering
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30'],
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False
        )
        
        processed_df = fe.preprocess_data(symbol_df)
        print(f"📊 Features added: {len(processed_df.columns)} columns")
        
        # Train/test split (80/20)
        split_point = int(len(processed_df) * 0.8)
        train_df = processed_df.iloc[:split_point].copy()
        test_df = processed_df.iloc[split_point:].copy()
        
        print(f"📊 Train: {len(train_df):,}, Test: {len(test_df):,}")
        
        # Environment configuration for single asset
        # State space = 1 (balance) + 1 (price) + 1 (shares) + 4 (tech indicators)
        state_space = 1 + 1 + 1 + len(['macd', 'rsi_30', 'cci_30', 'dx_30'])
        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1_000_000,
            "buy_cost_pct": [0.001],
            "sell_cost_pct": [0.001],
            "reward_scaling": 1e-4,
            "state_space": state_space,  # Correct calculation for single asset
            "action_space": 1,  # Single asset
            "tech_indicator_list": ['macd', 'rsi_30', 'cci_30', 'dx_30'],
            "stock_dim": 1,     # Single cryptocurrency
            "num_stock_shares": [0],  # Start with 0 shares
        }
        
        # Create training environment using PATCHED version
        train_env = DummyVecEnv([lambda: PatchedStockTradingEnv(df=train_df, **env_kwargs)])
        
        # Device setup
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"🚀 Training {symbol} on {device}")
        
        # PPO model optimized for single crypto
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
            max_grad_norm=0.5,
            device=device
        )
        
        start_time = datetime.now()
        print(f"🎯 Starting training for {symbol}...")
        
        # Train the model (reduced timesteps for faster completion)
        model.learn(total_timesteps=100_000)
        training_time = datetime.now() - start_time
        
        # Save model
        model_name = f"crypto_{symbol.lower()}_real_fixed_model"
        model.save(model_name)
        print(f"💾 {symbol} model saved: {model_name}")
        
        # Testing phase
        test_env = DummyVecEnv([lambda: PatchedStockTradingEnv(df=test_df, **env_kwargs)])
        
        print(f"🧪 Testing {symbol} model...")
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
            
            # Buy and hold comparison
            buy_hold_return = total_return
            
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
            
            print(f"\n🏆 {symbol} RESULTS:")
            print(f"💰 Algorithm Return: {algorithm_return:+.2f}%")
            print(f"💵 Profit: ${profit:+,.0f}")
            print(f"🏦 Final Portfolio: ${final_value:,.0f}")
            print(f"📊 vs Buy&Hold: {algorithm_return - buy_hold_return:+.2f}% difference")
            print(f"📈 Sharpe Ratio: {sharpe:.3f}")
            print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
            print(f"🎯 Actions - Buy: {buy_count}, Hold: {hold_count}, Sell: {sell_count}")
            print(f"⏱️ Training Time: {training_time}")
            
            # Store results
            results_dict[symbol] = {
                'status': 'success',
                'records': len(symbol_df),
                'data_return': total_return,
                'algorithm_return': algorithm_return,
                'profit': profit,
                'final_value': final_value,
                'buy_hold_return': buy_hold_return,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'volatility': volatility * 100,
                'training_time': str(training_time),
                'model_name': model_name,
                'actions': {'buy': buy_count, 'hold': hold_count, 'sell': sell_count},
                'steps': steps
            }
            
        else:
            print(f"❌ {symbol}: No portfolio values recorded")
            results_dict[symbol] = {
                'status': 'no_portfolio_values',
                'records': len(symbol_df),
                'error': 'No portfolio values recorded during testing'
            }
            
    except Exception as e:
        print(f"❌ {symbol}: Training failed with error: {str(e)}")
        results_dict[symbol] = {
            'status': 'failed',
            'records': len(symbol_df) if 'symbol_df' in locals() else 0,
            'error': str(e)
        }

def main():
    print("🚀 TRAINING ALL REAL CRYPTOCURRENCY MODELS WITH FINRL PATCH")
    print("="*80)
    
    # Load all available real cryptocurrency data
    datasets = [
        ('crypto_5min_2years.csv', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']),
        ('crypto_5currencies_2years.csv', ['ADAUSDT', 'SOLUSDT', 'MATICUSDT', 'DOTUSDT', 'LINKUSDT'])
    ]
    
    all_results = {}
    
    for dataset_file, expected_symbols in datasets:
        try:
            print(f"\n📁 Loading dataset: {dataset_file}")
            df = pd.read_csv(dataset_file)
            df['date'] = pd.to_datetime(df['date'])
            
            available_symbols = sorted(df['tic'].unique())
            print(f"💰 Available symbols: {available_symbols}")
            
            # Train each symbol individually
            for symbol in available_symbols:
                if symbol in expected_symbols:
                    train_single_crypto(symbol, df, all_results)
                    
        except FileNotFoundError:
            print(f"⚠️ Dataset {dataset_file} not found, skipping...")
            for symbol in expected_symbols:
                all_results[symbol] = {
                    'status': 'dataset_not_found',
                    'error': f'Dataset {dataset_file} not found'
                }
        except Exception as e:
            print(f"❌ Error loading {dataset_file}: {str(e)}")
            for symbol in expected_symbols:
                all_results[symbol] = {
                    'status': 'dataset_error',
                    'error': f'Error loading dataset: {str(e)}'
                }
    
    # Generate comprehensive results table
    print(f"\n🏆 FINAL RESULTS SUMMARY - ALL REAL CRYPTOCURRENCY MODELS")
    print("="*120)
    
    successful_results = {k: v for k, v in all_results.items() if v.get('status') == 'success'}
    
    if successful_results:
        print(f"{'Symbol':<10} {'Records':<10} {'Data Return':<12} {'Algo Return':<12} {'Profit':<12} {'Sharpe':<8} {'Max DD':<8} {'Status':<12}")
        print("-" * 120)
        
        for symbol, results in all_results.items():
            if results['status'] == 'success':
                print(f"{symbol:<10} {results['records']:<10,} {results['data_return']:>+10.2f}% {results['algorithm_return']:>+10.2f}% ${results['profit']:>+9,.0f} {results['sharpe']:>6.3f} {results['max_drawdown']:>6.2f}% {'✅ Success':<12}")
            else:
                status_text = f"❌ {results['status']}"
                print(f"{symbol:<10} {results.get('records', 0):<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<8} {'N/A':<8} {status_text:<12}")
    
    print(f"\n📈 PERFORMANCE SUMMARY:")
    if successful_results:
        avg_algo_return = np.mean([r['algorithm_return'] for r in successful_results.values()])
        avg_data_return = np.mean([r['data_return'] for r in successful_results.values()])
        avg_sharpe = np.mean([r['sharpe'] for r in successful_results.values()])
        
        print(f"   Average Algorithm Return: {avg_algo_return:+.2f}%")
        print(f"   Average Data Return: {avg_data_return:+.2f}%")
        print(f"   Average Sharpe Ratio: {avg_sharpe:.3f}")
        print(f"   Successful Models: {len(successful_results)}/{len(all_results)}")
        
        # Best and worst performers
        best_algo = max(successful_results.items(), key=lambda x: x[1]['algorithm_return'])
        worst_algo = min(successful_results.items(), key=lambda x: x[1]['algorithm_return'])
        
        print(f"   🏆 Best Performer: {best_algo[0]} ({best_algo[1]['algorithm_return']:+.2f}%)")
        print(f"   📉 Worst Performer: {worst_algo[0]} ({worst_algo[1]['algorithm_return']:+.2f}%)")
    
    print(f"\n✅ Training completed for all available real cryptocurrency datasets!")
    print(f"🔧 All models use PatchedStockTradingEnv to fix FinRL framework bugs")
    
    return all_results

if __name__ == "__main__":
    results = main()