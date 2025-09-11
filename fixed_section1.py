# Section 1: Environment Setup and Dependencies - FIXED VERSION
import sys
import os
from pathlib import Path

# Add paths safely
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))
sys.path.insert(0, str(current_dir.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Statistical analysis (with error handling)
try:
    from scipy import stats
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    print("✅ Statistical libraries loaded")
except ImportError as e:
    print(f"⚠️ Optional statistical library missing: {e}")
    stats = None

# Optional optimization library
try:
    import optuna
    print("✅ Optuna optimization available")
except ImportError:
    print("⚠️ Optuna not available - hyperparameter optimization disabled")
    optuna = None

# FinRL preprocessor (only what we need)
try:
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
    print("✅ FinRL preprocessors loaded")
except ImportError as e:
    print(f"❌ FinRL import error: {e}")

# MOST IMPORTANT: Use our comprehensive patch instead of buggy original FinRL
from finrl_comprehensive_patch import create_safe_finrl_env, safe_backtest_model

# Configure plotting safely
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        print("⚠️ Using default matplotlib style")

sns.set_palette("plasma")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

# Check PyTorch device
if torch.backends.mps.is_available():
    device = 'mps'
    print("🚀 Using Apple Silicon MPS GPU")
elif torch.cuda.is_available():
    device = 'cuda'
    print("🚀 Using NVIDIA CUDA GPU")
else:
    device = 'cpu'
    print("🖥️ Using CPU")

print("✅ Environment setup complete for Cardano (ADA) trading")
print("🔧 Using comprehensive FinRL patch for error-free training")
print(f"⚡ Compute device: {device}")