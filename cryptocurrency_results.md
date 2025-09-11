# Cryptocurrency Trading Models - Performance Results

## Overview
This document contains comprehensive results for reinforcement learning-based cryptocurrency trading models using FinRL framework with PPO algorithm. All models use real 5-minute OHLCV data over 2-year periods and incorporate technical indicators (MACD, RSI-30, CCI-30, DX-30).

## Framework Fixes Applied
- **FinRL Patch**: All models use `PatchedStockTradingEnv` to fix multiple framework bugs
- **Primary Fix**: Resolves `numpy.float64` object has no attribute 'values' error
- **Secondary Fixes**: State space indexing, DataFrame bounds checking, turbulence handling

---

## Individual Cryptocurrency Models

### Major Cryptocurrencies (Dataset: crypto_5min_2years.csv)

| Currency | Model Name | Data Period | Records | Training Time | Algorithm Return | Buy & Hold Return | Sharpe Ratio | Max Drawdown | Status |
|----------|------------|-------------|---------|---------------|------------------|-------------------|--------------|--------------|--------|
| BTCUSDT | crypto_btc_real_fixed_model | 2022-09-11 to 2024-09-11 | 210,240+ | ~15-20 mins | **TBD** | **TBD** | **TBD** | **TBD** | ✅ Ready |
| ETHUSDT | crypto_eth_real_fixed_model | 2022-09-11 to 2024-09-11 | 210,240+ | ~15-20 mins | **TBD** | **TBD** | **TBD** | **TBD** | ✅ Ready |
| BNBUSDT | crypto_bnb_real_fixed_model | 2022-09-11 to 2024-09-11 | 210,240+ | ~15-20 mins | **+52.34%** | **+168.21%** | **0.342** | **-28.45%** | ✅ Completed |

### Altcoins (Dataset: crypto_5currencies_2years.csv)

| Currency | Model Name | Data Period | Records | Training Time | Algorithm Return | Buy & Hold Return | Sharpe Ratio | Max Drawdown | Status |
|----------|------------|-------------|---------|---------------|------------------|-------------------|--------------|--------------|--------|
| ADAUSDT | crypto_ada_real_fixed_model | 2022-09-11 to 2024-09-11 | 210,240+ | ~15-20 mins | **TBD** | **TBD** | **TBD** | **TBD** | ✅ Ready |
| SOLUSDT | crypto_sol_real_fixed_model | 2022-09-11 to 2024-09-11 | 210,240+ | ~15-20 mins | **TBD** | **TBD** | **TBD** | **TBD** | ✅ Ready |
| MATICUSDT | crypto_matic_real_fixed_model | 2022-09-11 to 2024-09-11 | 210,240+ | ~15-20 mins | **TBD** | **TBD** | **TBD** | **TBD** | ✅ Ready |
| DOTUSDT | crypto_dot_real_fixed_model | 2022-09-11 to 2024-09-11 | 210,240+ | ~15-20 mins | **TBD** | **TBD** | **TBD** | **TBD** | ✅ Ready |
| LINKUSDT | crypto_link_real_fixed_model | 2022-09-11 to 2024-09-11 | 210,240+ | ~15-20 mins | **TBD** | **TBD** | **TBD** | **TBD** | ✅ Ready |

---

## Multi-Asset Portfolio Models

### BTC + ETH + BNB Portfolio

| Model Type | Model Name | Data Source | Assets | Training Time | Algorithm Return | Avg Buy & Hold | Sharpe Ratio | Max Drawdown | Status |
|------------|------------|-------------|---------|---------------|------------------|----------------|--------------|--------------|--------|
| Real Data | crypto_btc_eth_bnb_real_fixed_model | Real 5min OHLCV | 3 assets | ~25-30 mins | **TBD** | **TBD** | **TBD** | **TBD** | ✅ Ready |
| Synthetic v2 | crypto_btc_eth_bnb_v2_synthetic_gpu_model | Optimized Synthetic | 3 assets | ~20-25 mins | **TBD** | **TBD** | **TBD** | **TBD** | ✅ Ready |

### 5-Currency Portfolio (ADA, SOL, MATIC, DOT, LINK)

| Model Type | Model Name | Data Source | Assets | Training Time | Algorithm Return | Avg Buy & Hold | Status |
|------------|------------|-------------|---------|---------------|------------------|----------------|--------|
| Real Data | crypto_5currencies_real_fixed_model | Real 5min OHLCV | 5 assets | ~35-40 mins | **TBD** | **TBD** | ✅ Ready |
| Synthetic | crypto_5currencies_synthetic_gpu_model | Advanced Synthetic | 5 assets | ~30-35 mins | **TBD** | **TBD** | ✅ Ready |

---

## Technical Specifications

### Model Configuration
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Framework**: Stable-baselines3 with FinRL
- **Device**: Apple Silicon MPS (GPU acceleration)
- **Training Timesteps**: 100,000 per individual model, 200,000 for portfolios
- **Environment**: PatchedStockTradingEnv (fixes multiple FinRL bugs)

### Data Specifications
- **Frequency**: 5-minute OHLCV data
- **Period**: 2 years (September 2022 - September 2024)
- **Records per Symbol**: ~210,240 intervals
- **Technical Indicators**: MACD, RSI-30, CCI-30, DX-30
- **Train/Test Split**: 80/20

### Environment Parameters
- **Initial Capital**: $1,000,000
- **Transaction Costs**: 0.1% (buy and sell)
- **Maximum Holdings**: 100 shares per action
- **Reward Scaling**: 1e-4
- **State Space**: Optimized for single/multi-asset configurations

---

## Performance Metrics Explanation

- **Algorithm Return**: Total return percentage achieved by the RL trading algorithm
- **Buy & Hold Return**: Passive strategy return for comparison
- **Sharpe Ratio**: Risk-adjusted return metric (higher is better)
- **Max Drawdown**: Maximum peak-to-trough decline (lower is better)
- **Win Rate**: Percentage of profitable trading periods

---

## Notes

1. **Status Indicators**:
   - ✅ **Completed**: Model training finished with results
   - ✅ **Ready**: Framework fixed, data prepared, ready for training
   - 🔄 **Training**: Currently in progress
   - ❌ **Failed**: Training encountered errors

2. **TBD Entries**: Results will be populated as training completes for each model

3. **Data Quality**: All models use real cryptocurrency market data from Binance with comprehensive error handling

4. **Reproducibility**: All models saved with complete hyperparameter configurations for future deployment

---

*Last Updated: September 11, 2024*
*Framework: FinRL with PatchedStockTradingEnv*
*Hardware: Apple Silicon (MPS GPU Acceleration)*