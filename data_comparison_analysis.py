import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def analyze_data_characteristics():
    """Analyze and compare different crypto datasets"""
    
    print("📊 CRYPTO DATA ANALYSIS COMPARISON")
    print("="*70)
    
    datasets = {}
    
    # Check available datasets
    files_to_check = [
        ('crypto_test_data.csv', '30-day 1-minute test data'),
        ('crypto_5year_hourly.csv', '5-year hourly data'),
        ('crypto_5min_2years.csv', '2-year 5-minute data (if available)')
    ]
    
    for filename, description in files_to_check:
        if os.path.exists(filename):
            print(f"✅ Found: {filename} - {description}")
            try:
                df = pd.read_csv(filename)
                df['date'] = pd.to_datetime(df['date'])
                datasets[description] = {
                    'df': df,
                    'filename': filename
                }
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
        else:
            print(f"❌ Missing: {filename} - {description}")
    
    print("\n" + "="*70)
    print("DATASET COMPARISON SUMMARY")
    print("="*70)
    
    for name, data in datasets.items():
        df = data['df']
        
        print(f"\n📈 {name.upper()}")
        print("-" * 50)
        print(f"  File: {data['filename']}")
        print(f"  Total Records: {len(df):,}")
        print(f"  Symbols: {sorted(df['tic'].unique())}")
        print(f"  Records per Symbol: {len(df) // len(df['tic'].unique()):,}")
        print(f"  Date Range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Time Span: {(df['date'].max() - df['date'].min()).days} days")
        
        # Calculate frequency
        if len(df) > 1:
            dates_per_symbol = df[df['tic'] == df['tic'].iloc[0]]['date'].sort_values()
            if len(dates_per_symbol) > 1:
                avg_interval = (dates_per_symbol.iloc[1] - dates_per_symbol.iloc[0])
                print(f"  Data Frequency: ~{avg_interval}")
        
        # Price statistics
        print(f"  Price Range:")
        for symbol in sorted(df['tic'].unique()):
            symbol_data = df[df['tic'] == symbol]
            min_price = symbol_data['close'].min()
            max_price = symbol_data['close'].max()
            print(f"    {symbol}: ${min_price:,.2f} - ${max_price:,.2f}")
    
    return datasets

def calculate_expected_performance_differences():
    """Calculate expected performance differences between timeframes"""
    
    print("\n" + "="*70)
    print("EXPECTED PERFORMANCE DIFFERENCES BY TIMEFRAME")
    print("="*70)
    
    timeframes = {
        '1-minute': {
            'data_points_per_day': 1440,
            'advantages': [
                'Captures intraday volatility patterns',
                'More trading opportunities',
                'Better entry/exit precision',
                'Can exploit short-term arbitrage'
            ],
            'challenges': [
                'Much higher computational load',
                'More noise in signals',
                'Higher transaction costs impact',
                'Requires faster execution'
            ],
            'best_for': 'Day trading, scalping strategies'
        },
        '5-minute': {
            'data_points_per_day': 288,
            'advantages': [
                'Good balance of detail vs noise',
                'Captures key intraday movements',
                'Manageable computational load',
                'Reduces market noise'
            ],
            'challenges': [
                'May miss very short-term opportunities',
                'Still requires good execution speed',
                'Medium computational requirements'
            ],
            'best_for': 'Swing trading, short-term strategies'
        },
        '1-hour': {
            'data_points_per_day': 24,
            'advantages': [
                'Low noise, clear trends',
                'Efficient computational processing',
                'Good for trend following',
                'Lower transaction costs impact'
            ],
            'challenges': [
                'Misses intraday opportunities',
                'Slower reaction to market changes',
                'Less precise entry/exit points'
            ],
            'best_for': 'Position trading, trend following'
        }
    }
    
    for timeframe, info in timeframes.items():
        print(f"\n🕐 {timeframe.upper()} DATA")
        print("-" * 30)
        print(f"  Data Points/Day: {info['data_points_per_day']}")
        print(f"  Best For: {info['best_for']}")
        
        print("  ✅ Advantages:")
        for advantage in info['advantages']:
            print(f"    • {advantage}")
        
        print("  ⚠️ Challenges:")
        for challenge in info['challenges']:
            print(f"    • {challenge}")

def estimate_training_requirements():
    """Estimate computational and time requirements for different datasets"""
    
    print("\n" + "="*70)
    print("TRAINING REQUIREMENTS ESTIMATION")
    print("="*70)
    
    scenarios = [
        {
            'name': '30-day 1-minute test',
            'records': 129_602,  # From our test data
            'timesteps': 50_000,
            'estimated_time': '10-15 minutes',
            'memory_usage': 'Low (~1GB)',
            'complexity': 'Low'
        },
        {
            'name': '5-year hourly',
            'records': 131_340,  # From our hourly data
            'timesteps': 100_000,
            'estimated_time': '15-25 minutes',
            'memory_usage': 'Medium (~2GB)',
            'complexity': 'Medium'
        },
        {
            'name': '2-year 5-minute',
            'records': 210_240,  # Estimated (288 * 730)
            'timesteps': 150_000,
            'estimated_time': '30-45 minutes',
            'memory_usage': 'Medium-High (~4GB)',
            'complexity': 'High'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🚀 {scenario['name'].upper()}")
        print("-" * 40)
        print(f"  Records: ~{scenario['records']:,}")
        print(f"  Suggested Timesteps: {scenario['timesteps']:,}")
        print(f"  Estimated Training Time: {scenario['estimated_time']}")
        print(f"  Memory Usage: {scenario['memory_usage']}")
        print(f"  Model Complexity: {scenario['complexity']}")

def generate_recommendations():
    """Generate recommendations for crypto trading with different timeframes"""
    
    print("\n" + "="*70)
    print("🎯 RECOMMENDATIONS FOR CRYPTO TRADING")
    print("="*70)
    
    recommendations = {
        'For Beginners': {
            'suggested_timeframe': '1-hour',
            'dataset': '5-year hourly data',
            'reasons': [
                'Lower computational requirements',
                'Clearer trends, less noise',
                'Good for learning RL concepts',
                'Faster iteration cycles'
            ],
            'model_params': {
                'timesteps': '50,000 - 100,000',
                'batch_size': '64',
                'learning_rate': '3e-4'
            }
        },
        'For Intermediate': {
            'suggested_timeframe': '5-minute',
            'dataset': '2-year 5-minute data',
            'reasons': [
                'Good balance of detail and efficiency',
                'Captures most important patterns',
                'Realistic for actual trading',
                'Manageable complexity'
            ],
            'model_params': {
                'timesteps': '100,000 - 200,000',
                'batch_size': '128',
                'learning_rate': '1e-4 to 3e-4'
            }
        },
        'For Advanced': {
            'suggested_timeframe': '1-minute or multi-timeframe',
            'dataset': 'Multiple timeframes combined',
            'reasons': [
                'Maximum detail and opportunities',
                'Can exploit micro-movements',
                'Professional-grade precision',
                'Complex strategy development'
            ],
            'model_params': {
                'timesteps': '200,000+',
                'batch_size': '256+',
                'learning_rate': 'Adaptive/scheduled'
            }
        }
    }
    
    for level, rec in recommendations.items():
        print(f"\n🎓 {level.upper()}")
        print("-" * 30)
        print(f"  Timeframe: {rec['suggested_timeframe']}")
        print(f"  Dataset: {rec['dataset']}")
        print("  Reasons:")
        for reason in rec['reasons']:
            print(f"    • {reason}")
        print("  Model Parameters:")
        for param, value in rec['model_params'].items():
            print(f"    • {param}: {value}")

def main():
    """Main analysis function"""
    
    # Analyze available datasets
    datasets = analyze_data_characteristics()
    
    # Calculate performance differences
    calculate_expected_performance_differences()
    
    # Estimate training requirements
    estimate_training_requirements()
    
    # Generate recommendations
    generate_recommendations()
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print("The 5-minute data download is optimal for most crypto trading applications.")
    print("Once crypto_5min_2years.csv is ready, run: python finrl_crypto_5min_trading.py")

if __name__ == "__main__":
    main()