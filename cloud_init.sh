#!/bin/bash
# Enhanced Transformer Cloud Initialization Script

echo "🚀 Setting up Enhanced Transformer Training Environment"

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv transformer_env
source transformer_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install pandas numpy scikit-learn matplotlib ta talib-binary tqdm psutil jupyter

# Create working directory
mkdir -p ~/transformer_training
cd ~/transformer_training

echo "✅ Environment setup complete!"
echo "📋 Next steps:"
echo "   1. Upload your deployment package to ~/transformer_training/"
echo "   2. Unzip: unzip enhanced_transformer_deployment.zip"
echo "   3. Start notebook: jupyter notebook --ip=0.0.0.0 --port=8888"
echo "   4. Access via browser with provided token"
