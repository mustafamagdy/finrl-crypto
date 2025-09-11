#!/usr/bin/env python3
"""
QuantConnect LEAN Local Backtest Runner
=======================================

This script demonstrates how to run the FinRL crypto strategy locally using
QuantConnect LEAN engine. This is useful for local testing before deploying
to the QuantConnect cloud platform.

Prerequisites:
- QuantConnect LEAN installed locally
- Docker (for LEAN runtime)
- Python environment with required packages

Usage:
    python run_lean_backtest.py
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def setup_lean_environment():
    """Setup QuantConnect LEAN environment for backtesting"""
    
    print("🔧 Setting up QuantConnect LEAN environment...")
    
    # Check if LEAN is installed
    try:
        result = subprocess.run(['lean', '--version'], capture_output=True, text=True)
        print(f"✅ LEAN CLI version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ LEAN CLI not found. Please install QuantConnect LEAN:")
        print("   npm install -g @quantconnect/lean-cli")
        return False
    
    return True

def create_lean_config():
    """Create LEAN configuration for the backtest"""
    
    config = {
        "environment": "backtesting",
        "algorithm-type-name": "FinRLCryptoTradingAlgorithm",
        "algorithm-language": "Python", 
        "algorithm-location": "FinRLCryptoStrategy/main.py",
        
        "data-folder": "./data",
        "cache-location": "./cache",
        "log-handler": "CompositeLogHandler",
        
        "job-organization-id": "",
        "api-access-token": "",
        
        "parameters": {
            "start-date": "20240101",
            "end-date": "20241231",
            "cash": "100000",
            "symbol": "BTCUSD"
        },
        
        "data-provider": "QuantConnect",
        "map-file-provider": "LocalDiskMapFileProvider",
        "factor-file-provider": "LocalDiskFactorFileProvider",
        
        "results-destination-folder": "./results",
        "debugging": False,
        "desktop-http-port": 1234,
        
        "composer-dll-directory": "",
        "messaging-handler": "QuantConnect.Messaging.Messaging",
        "job-queue-handler": "QuantConnect.Queues.JobQueue",
        "api-handler": "QuantConnect.Api.Api",
        
        "close-automatically": True
    }
    
    config_path = "lean.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created LEAN configuration: {config_path}")
    return config_path

def run_backtest():
    """Run the FinRL crypto strategy backtest"""
    
    print("🚀 Starting FinRL Crypto Strategy Backtest...")
    print("=" * 60)
    
    try:
        # Run LEAN backtest
        cmd = [
            'lean', 'backtest', 
            '--project', 'FinRLCryptoStrategy',
            '--start', '20240101',
            '--end', '20241231'
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        
        # Start the backtest process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return_code = process.poll()
        
        if return_code == 0:
            print("\n✅ Backtest completed successfully!")
            print("📊 Check results in ./results/ folder")
        else:
            print(f"\n❌ Backtest failed with return code: {return_code}")
            
        return return_code == 0
        
    except Exception as e:
        print(f"❌ Backtest execution failed: {str(e)}")
        return False

def analyze_results():
    """Analyze and display backtest results"""
    
    print("\n📊 BACKTEST RESULTS ANALYSIS")
    print("=" * 60)
    
    results_dir = Path("./results")
    
    if not results_dir.exists():
        print("⚠️ Results directory not found")
        return
    
    # Look for result files
    log_files = list(results_dir.glob("*.log"))
    json_files = list(results_dir.glob("*.json"))
    
    print(f"📄 Log files found: {len(log_files)}")
    print(f"📄 JSON files found: {len(json_files)}")
    
    # Display latest log file content (last 50 lines)
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f"\n📋 Latest log file: {latest_log.name}")
        print("-" * 40)
        
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                # Show last 50 lines
                for line in lines[-50:]:
                    print(line.rstrip())
        except Exception as e:
            print(f"Error reading log file: {e}")
    
    # Parse performance metrics from JSON results
    if json_files:
        latest_json = max(json_files, key=os.path.getctime)
        print(f"\n📈 Performance metrics: {latest_json.name}")
        print("-" * 40)
        
        try:
            with open(latest_json, 'r') as f:
                results = json.load(f)
            
            # Extract key metrics
            if 'Statistics' in results:
                stats = results['Statistics']
                print(f"Total Return: {stats.get('Total Performance', 'N/A')}")
                print(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}")
                print(f"Max Drawdown: {stats.get('Drawdown', 'N/A')}")
                print(f"Total Trades: {stats.get('Total Trades', 'N/A')}")
                
        except Exception as e:
            print(f"Error parsing results: {e}")

def main():
    """Main execution function"""
    
    print("🤖 FinRL Crypto Strategy - QuantConnect LEAN Backtest Runner")
    print("=" * 70)
    print("📅 Backtest Period: 2024")
    print("💰 Starting Capital: $100,000")
    print("🪙 Symbol: BTCUSD")
    print("🧠 Model: FinRL PPO + Technical Analysis")
    print()
    
    # Check if we're in the right directory
    if not Path("FinRLCryptoStrategy").exists():
        print("❌ FinRLCryptoStrategy directory not found")
        print("   Please run this script from the backtesting/ folder")
        return False
    
    # Setup LEAN environment
    if not setup_lean_environment():
        return False
    
    # Create configuration
    config_path = create_lean_config()
    
    # Run backtest
    success = run_backtest()
    
    if success:
        # Analyze results
        analyze_results()
        
        print("\n🎉 Backtest completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Review performance metrics above")
        print("   2. Check detailed logs in ./results/")
        print("   3. Optimize strategy parameters if needed") 
        print("   4. Deploy to QuantConnect cloud for live trading")
        
    else:
        print("\n💥 Backtest failed. Please check the logs for errors.")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Ensure LEAN is properly installed")
        print("   2. Check that data feeds are available")
        print("   3. Verify algorithm syntax and imports")
        print("   4. Review model file paths and permissions")
    
    # Cleanup
    if Path(config_path).exists():
        os.remove(config_path)
        print(f"🧹 Cleaned up: {config_path}")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Backtest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)