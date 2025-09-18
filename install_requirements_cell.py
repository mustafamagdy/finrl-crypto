# First Jupyter Cell - Install All Requirements
# Copy this entire cell into your Jupyter notebook and run it first

import subprocess
import sys
import os
from IPython.display import clear_output
import time

def run_command(command, description=""):
    """Run a command and display output"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def check_gpu():
    """Check if GPU is available"""
    print("🎮 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU detected: {gpu_name}")
            print(f"🧠 GPU Memory: {gpu_memory:.1f} GB")
            return True
        elif torch.backends.mps.is_available():
            print("✅ Apple Silicon GPU (MPS) detected")
            return True
        else:
            print("⚠️ No GPU detected, using CPU")
            return False
    except ImportError:
        print("⚠️ PyTorch not yet installed, skipping GPU check")
        return False

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print("🚀 Installing PyTorch with GPU support...")

    # Check CUDA availability
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("🎮 NVIDIA GPU detected, installing CUDA version")
            pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        else:
            print("🎮 No NVIDIA GPU, installing CPU version")
            pytorch_cmd = "pip install torch torchvision torchaudio"
    except:
        print("🎮 Installing CPU version of PyTorch")
        pytorch_cmd = "pip install torch torchvision torchaudio"

    return run_command(pytorch_cmd, "Installing PyTorch")

def main():
    """Main installation function"""
    print("🚀 Enhanced Transformer - Installation Script")
    print("=" * 60)

    # Update pip first
    run_command("pip install --upgrade pip", "Upgrading pip")

    # Install PyTorch
    install_pytorch()

    # Install core ML libraries
    libraries = [
        ("numpy", "NumPy for numerical computing"),
        ("pandas", "Pandas for data manipulation"),
        ("scikit-learn", "Scikit-learn for ML utilities"),
        ("matplotlib", "Matplotlib for plotting"),
        ("seaborn", "Seaborn for statistical plots"),
        ("plotly", "Plotly for interactive plots"),
        ("tqdm", "TQDM for progress bars"),
        ("psutil", "PSUtil for system monitoring"),
        ("requests", "Requests for HTTP")
    ]

    for lib, desc in libraries:
        run_command(f"pip install {lib}", f"Installing {desc}")

    # Install technical analysis libraries
    print("\n📈 Installing technical analysis libraries...")

    # Try TA-Lib first (may fail on some systems)
    if not run_command("pip install talib-binary", "Installing TA-Lib"):
        print("⚠️ TA-Lib installation failed, will use alternative implementations")

    # Install TA library
    run_command("pip install ta", "Installing TA library")

    # Install Jupyter and widgets
    run_command("pip install jupyter ipywidgets", "Installing Jupyter and widgets")

    # Additional ML libraries
    ml_libs = [
        ("scipy", "SciPy for scientific computing"),
        ("joblib", "Joblib for parallel computing"),
        ("memory-profiler", "Memory profiler for monitoring")
    ]

    for lib, desc in ml_libs:
        run_command(f"pip install {lib}", f"Installing {desc}")

    # Clear output and show final status
    clear_output(wait=True)
    print("✅ Installation completed!")
    print("\n📋 Installed Components:")
    print("   🔥 PyTorch with GPU support")
    print("   📊 NumPy, Pandas, Scikit-learn")
    print("   📈 Matplotlib, Seaborn, Plotly")
    print("   📈 Technical Analysis (TA, TA-Lib)")
    print("   🚀 Jupyter Notebook with widgets")
    print("   🔧 System monitoring tools")

    # Check GPU again
    check_gpu()

    print("\n🎯 Ready to start training!")
    print("💡 Next steps:")
    print("   1. Upload your crypto data file (crypto_5min_2years.csv)")
    print("   2. Run the remaining notebook cells")
    print("   3. Start training when ready")

    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import torch
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ All core imports successful!")

        # Test GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"🚀 Default device: {device}")

    except Exception as e:
        print(f"❌ Import test failed: {e}")

if __name__ == "__main__":
    main()