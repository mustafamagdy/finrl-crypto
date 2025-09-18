#!/usr/bin/env python3
"""
Script to upload enhanced transformer training files to GPU cloud services
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path
import json

def create_deployment_package():
    """Create a zip package with all required files"""
    print("📦 Creating deployment package...")

    # List of files to include
    required_files = [
        "enhanced_transformer_training.ipynb",
        "transformer_enhanced_v2.py",
        "enhanced_features.py",
        "requirements.txt",
        "transformers_enhance_plan.md",
        "PHASE1_IMPLEMENTATION_SUMMARY.md"
    ]

    # Check if files exist
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return None

    # Create zip file
    zip_filename = "enhanced_transformer_deployment.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in required_files:
            zipf.write(file)
            print(f"   Added: {file}")

    print(f"✅ Created {zip_filename}")
    return zip_filename

def create_setup_instructions():
    """Create setup instructions for different cloud platforms"""
    instructions = {
        "lambda_labs": {
            "name": "Lambda Labs",
            "setup": """
# Lambda Labs Setup Instructions:
1. Go to lambdalabs.com and create account
2. Choose instance type: RTX A6000 (48GB VRAM recommended)
3. Select image: PyTorch 2.0+ with CUDA 12.1
4. Launch instance and SSH into it
5. Upload the deployment package:
   scp enhanced_transformer_deployment.zip user@instance_ip:/home/user/
6. Unzip and install:
   unzip enhanced_transformer_deployment.zip
   pip install -r requirements.txt
7. Start Jupyter notebook:
   jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
8. Access notebook in browser
            """,
            "expected_cost": "$0.60/hour = ~$6-12 total",
            "training_time": "6-8 hours"
        },
        "vast_ai": {
            "name": "Vast.ai",
            "setup": """
# Vast.ai Setup Instructions:
1. Go to vast.ai and create account
2. Browse marketplace for RTX 4090 or A6000
3. Filter for: PyTorch, Docker, >=24GB VRAM
4. Rent instance and get SSH credentials
5. Upload deployment package:
   scp enhanced_transformer_deployment.zip user@instance_ip:~/
6. Connect and setup:
   ssh user@instance_ip
   unzip enhanced_transformer_deployment.zip
   pip install -r requirements.txt
7. Start notebook:
   jupyter notebook --allow-root --ip=0.0.0.0 --port=8888
            """,
            "expected_cost": "$0.30/hour = ~$3-6 total",
            "training_time": "8-12 hours"
        },
        "colab": {
            "name": "Google Colab Pro",
            "setup": """
# Google Colab Setup Instructions:
1. Upload deployment package to Google Drive
2. Open Colab and mount Drive:
   from google.colab import drive
   drive.mount('/content/drive')
3. Copy files to Colab environment:
   !cp /content/drive/MyDrive/enhanced_transformer_deployment.zip /content/
   !unzip enhanced_transformer_deployment.zip
4. Install requirements:
   !pip install -r requirements.txt
5. Upload and open the notebook
6. Set runtime to GPU (Runtime -> Change runtime type)
            """,
            "expected_cost": "$10/month unlimited",
            "training_time": "8-12 hours"
        }
    }

    return instructions

def print_platform_comparison():
    """Print comparison of cloud platforms"""
    print("🌟 GPU Cloud Platform Comparison")
    print("=" * 80)

    platforms = [
        {
            "name": "Lambda Labs",
            "gpu": "RTX A6000",
            "vram": "48GB",
            "cost_hour": "$0.60",
            "cost_total": "$6-12",
            "time": "6-8h",
            "ease": "⭐⭐⭐⭐",
            "performance": "⭐⭐⭐⭐⭐"
        },
        {
            "name": "Vast.ai",
            "gpu": "RTX 4090",
            "vram": "24GB",
            "cost_hour": "$0.30",
            "cost_total": "$3-6",
            "time": "8-12h",
            "ease": "⭐⭐⭐",
            "performance": "⭐⭐⭐⭐"
        },
        {
            "name": "Google Colab Pro",
            "gpu": "A100/T4",
            "vram": "16-40GB",
            "cost_hour": "~$0.15",
            "cost_total": "$10/month",
            "time": "8-12h",
            "ease": "⭐⭐⭐⭐⭐",
            "performance": "⭐⭐⭐"
        }
    ]

    # Header
    print(f"{'Platform':<15} {'GPU':<12} {'VRAM':<6} {'Cost/Hr':<8} {'Total':<8} {'Time':<8} {'Ease':<5} {'Perf':<5}")
    print("-" * 80)

    for platform in platforms:
        print(f"{platform['name']:<15} {platform['gpu']:<12} {platform['vram']:<6} "
              f"{platform['cost_hour']:<8} {platform['cost_total']:<8} {platform['time']:<8} "
              f"{platform['ease']:<5} {platform['performance']:<5}")

    print("\n🏆 Recommendation: Lambda Labs for best balance of cost and performance")
    print("💰 Cheapest: Vast.ai if budget is primary concern")
    print("🎯 Easiest: Google Colab Pro for beginners")

def create_cloud_init_script():
    """Create cloud initialization script"""
    script_content = """#!/bin/bash
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
"""

    with open("cloud_init.sh", "w") as f:
        f.write(script_content)

    os.chmod("cloud_init.sh", 0o755)
    print("✅ Created cloud_init.sh")

def main():
    """Main deployment script"""
    print("🚀 Enhanced Transformer GPU Cloud Deployment Script")
    print("=" * 60)

    # Create deployment package
    zip_file = create_deployment_package()
    if not zip_file:
        return

    # Create setup instructions
    instructions = create_setup_instructions()

    # Print platform comparison
    print_platform_comparison()

    # Create cloud init script
    create_cloud_init_script()

    # Save instructions to file
    with open("deployment_instructions.json", "w") as f:
        json.dump(instructions, f, indent=2)

    print(f"\n✅ Deployment package created: {zip_file}")
    print(f"📋 Setup instructions saved: deployment_instructions.json")
    print(f"🔧 Cloud init script created: cloud_init.sh")

    print("\n📦 Files to upload to cloud:")
    print(f"   - {zip_file}")
    print("   - crypto_5min_2years.csv (your training data)")

    print("\n🎯 Recommended GPU instances:")
    print("   - RTX A6000 (48GB VRAM) - Best performance")
    print("   - RTX 4090 (24GB VRAM) - Good value")
    print("   - A100 (40GB VRAM) - Fastest training")

    print("\n💰 Expected total cost: $3-12 USD")
    print("⏱️  Expected training time: 6-12 hours")

    print("\n🚀 Ready for cloud deployment!")

if __name__ == "__main__":
    main()