#!/bin/bash

echo "=========================================="
echo "GPU Server Training Environment Setup"
echo "Cryptocurrency Transformer Trading Model"
echo "=========================================="

# Check if running on GPU server
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No NVIDIA GPU detected. Training will use CPU (very slow)."
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv gpu_training_env
source gpu_training_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    # Check CUDA version
    cuda_version=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -c2-4)
    echo "CUDA version detected: $cuda_version"
    
    if [[ "$cuda_version" == "11."* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$cuda_version" == "12."* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "Installing default CUDA version..."
        pip install torch torchvision torchaudio
    fi
else
    echo "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install additional development tools
echo "Installing development tools..."
pip install ipython jupyterlab tensorboard

# Setup TA-Lib (if not already installed)
echo "Setting up TA-Lib..."
if ! python -c "import talib" &> /dev/null; then
    echo "TA-Lib not found. Installing..."
    
    # Try different installation methods
    if command -v conda &> /dev/null; then
        conda install -c conda-forge ta-lib
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y ta-lib-dev
        pip install TA-Lib
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ta-lib
            pip install TA-Lib
        else
            echo "Please install Homebrew and run: brew install ta-lib"
        fi
    else
        echo "Please install TA-Lib manually for your system"
    fi
fi

# Create necessary directories
echo "Creating directory structure..."
mkdir -p checkpoints
mkdir -p data
mkdir -p logs
mkdir -p results

# Test GPU availability
echo "Testing GPU availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.version.cuda}')
    print(f'✓ GPU devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  - GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✓ Apple Silicon GPU (MPS) available')
else:
    print('⚠️  Using CPU only')
"

# Test key imports
echo "Testing key imports..."
python3 -c "
try:
    import torch
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn
    import tqdm
    import wandb
    import ccxt
    import yfinance
    import ta
    import talib
    print('✓ All key packages imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"

# Set up Weights & Biases (optional)
echo "Setting up Weights & Biases..."
echo "Please run 'wandb login' if you want to use experiment tracking"

# Create activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
echo "Activating GPU training environment..."
source gpu_training_env/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "Environment activated. Ready for training!"
echo "Run: python gpu_training_script.py --help for usage"
EOF

chmod +x activate_env.sh

echo "=========================================="
echo "✓ Environment setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source activate_env.sh"
echo "2. Login to W&B (optional): wandb login"
echo "3. Download data: python gpu_data_processing.py"
echo "4. Start training: python gpu_training_script.py --data-path crypto_production_dataset.csv"
echo ""
echo "For advanced usage, see README.md"
echo "=========================================="