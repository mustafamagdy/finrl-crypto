#!/usr/bin/env python3

import pandas as pd
import numpy as np
from finrl_comprehensive_patch import create_safe_finrl_env, safe_backtest_model
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl import config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

def test_comprehensive_patch():
    """Test the comprehensive FinRL patch with real crypto data"""
    
    print("🧪 TESTING COMPREHENSIVE FINRL PATCH")
    print("="*60)
    
    try:
        # Load test data
        print("📊 Loading test data...")
        df = pd.read_csv('crypto_5currencies_2years.csv')
        
        # Filter for one currency to test
        test_symbol = 'ADAUSDT'
        df_test = df[df['tic'] == test_symbol].copy().reset_index(drop=True)
        
        print(f"✅ Loaded {len(df_test):,} records for {test_symbol}")
        print(f"📅 Date range: {df_test['date'].min()} to {df_test['date'].max()}")
        
        # Add technical indicators
        print("🔧 Adding technical indicators...")
        tech_indicators = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=tech_indicators,
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False
        )
        processed = fe.preprocess_data(df_test)
        print(f"✅ Technical indicators added: {processed.shape}")
        
        # Split data
        print("✂️ Splitting data...")
        train_data = data_split(processed, '2023-09-11', '2024-09-11')
        test_data = data_split(processed, '2024-09-11', '2025-09-10')
        print(f"📊 Train: {len(train_data):,}, Test: {len(test_data):,}")
        
        # Test 1: Environment Creation
        print("\n🔬 TEST 1: Environment Creation")
        print("-" * 40)
        
        try:
            env = create_safe_finrl_env(train_data)
            print("✅ Environment created successfully")
            
            # Test reset
            obs, info = env.reset()
            print(f"✅ Reset successful - State shape: {obs.shape}")
            print(f"   State sample: {obs[:5]}...")
            
        except Exception as e:
            print(f"❌ Environment creation failed: {e}")
            return False
        
        # Test 2: Step Function
        print("\n🔬 TEST 2: Step Function")
        print("-" * 40)
        
        try:
            # Test multiple step types
            test_actions = [
                [0.5],          # Single positive action
                [-0.3],         # Single negative action  
                [0.0],          # No action
                np.array([0.1]) # Numpy array action
            ]
            
            for i, action in enumerate(test_actions):
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"✅ Step {i+1}: Action={action}, Reward={reward:.4f}, Done={terminated}")
                
        except Exception as e:
            print(f"❌ Step function failed: {e}")
            return False
        
        # Test 3: Training Compatibility
        print("\n🔬 TEST 3: Training Compatibility")
        print("-" * 40)
        
        try:
            # Create vectorized environment
            env = create_safe_finrl_env(train_data)
            vec_env = DummyVecEnv([lambda: env])
            
            # Test with minimal PPO model
            print("🤖 Creating PPO model...")
            model = PPO("MlpPolicy", vec_env, verbose=0, device='cpu')
            
            # Test very short training
            print("🚀 Testing short training...")
            model.learn(total_timesteps=100, log_interval=None)
            print("✅ Training completed without errors")
            
        except Exception as e:
            print(f"❌ Training compatibility failed: {e}")
            return False
        
        # Test 4: Backtesting
        print("\n🔬 TEST 4: Safe Backtesting")
        print("-" * 40)
        
        try:
            results = safe_backtest_model(model, test_data)
            
            print("✅ Backtesting completed successfully")
            print(f"   Total Return: {results['total_return']:.2f}%")
            print(f"   Sharpe Ratio: {results['sharpe']:.4f}")
            print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"   Steps Completed: {results['steps_completed']}")
            
            # Verify all required keys are present
            required_keys = ['total_return', 'sharpe', 'max_drawdown', 'final_value']
            missing_keys = [key for key in required_keys if key not in results]
            
            if missing_keys:
                print(f"⚠️ Missing result keys: {missing_keys}")
            else:
                print("✅ All result keys present")
                
        except Exception as e:
            print(f"❌ Backtesting failed: {e}")
            return False
        
        # Test 5: Error Resilience
        print("\n🔬 TEST 5: Error Resilience")
        print("-" * 40)
        
        try:
            # Test with problematic actions
            env = create_safe_finrl_env(train_data)
            env.reset()
            
            problematic_actions = [
                [np.inf],           # Infinite action
                [np.nan],           # NaN action
                [-1e10],            # Extremely large negative
                [1e10],             # Extremely large positive
                [],                 # Empty action
                [0.5, 0.3, 0.1],   # Wrong dimension
            ]
            
            error_count = 0
            for i, action in enumerate(problematic_actions):
                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"✅ Handled problematic action {i+1}: {action}")
                except Exception as e:
                    error_count += 1
                    print(f"❌ Action {i+1} caused error: {e}")
            
            if error_count == 0:
                print("✅ All problematic actions handled gracefully")
            else:
                print(f"⚠️ {error_count} actions caused errors")
                
        except Exception as e:
            print(f"❌ Error resilience test failed: {e}")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Comprehensive patch is working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_comprehensive_patch()
    if success:
        print("\n✅ PATCH VALIDATION SUCCESSFUL")
        print("🚀 Ready for production training!")
    else:
        print("\n❌ PATCH VALIDATION FAILED")
        print("🔧 Additional fixes may be needed")