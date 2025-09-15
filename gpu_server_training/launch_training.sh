#!/bin/bash

echo "============================================="
echo "  GPU Server Training Launcher"
echo "  Cryptocurrency Transformer Trading Model"
echo "============================================="

# Configuration
DATA_PATH="crypto_production_dataset.csv"
EXPERIMENT_NAME="crypto_transformer_production_$(date +%Y%m%d_%H%M%S)"
EPOCHS=500
BATCH_SIZE=128
LEARNING_RATE=5e-5

# Check if environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not detected. Activating..."
    if [ -f "gpu_training_env/bin/activate" ]; then
        source gpu_training_env/bin/activate
        echo "✓ Environment activated"
    else
        echo "❌ Environment not found. Please run setup_environment.sh first"
        exit 1
    fi
fi

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA GPU available: {torch.cuda.get_device_name()}')
    print(f'✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    # Adjust batch size based on GPU memory
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_mem_gb < 12:
        print('⚠️  Limited GPU memory detected. Reducing batch size to 64')
        import sys
        sys.exit(64)  # Return batch size as exit code
    elif gpu_mem_gb < 24:
        print('✓ Standard GPU memory. Using batch size 128')
        import sys
        sys.exit(128)
    else:
        print('✓ High GPU memory. Using batch size 256')
        import sys
        sys.exit(256)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✓ Apple Silicon GPU (MPS) available')
    print('⚠️  Using reduced batch size for MPS')
    import sys
    sys.exit(32)
else:
    print('❌ No GPU detected. Training will be very slow!')
    print('⚠️  Using very small batch size for CPU')
    import sys
    sys.exit(16)
"

# Get optimal batch size from GPU check
optimal_batch_size=$?
if [ $optimal_batch_size -ne 0 ]; then
    BATCH_SIZE=$optimal_batch_size
    echo "✓ Using optimized batch size: $BATCH_SIZE"
fi

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Data file not found: $DATA_PATH"
    echo "Running data processing..."
    
    python3 gpu_data_processing.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Data processing failed!"
        exit 1
    fi
    
    if [ ! -f "$DATA_PATH" ]; then
        echo "❌ Data file still not found after processing!"
        exit 1
    fi
fi

echo "✓ Data file found: $DATA_PATH"

# Check data size
data_size=$(wc -l < "$DATA_PATH")
echo "✓ Dataset contains $data_size rows"

# Setup experiment tracking
echo "Setting up experiment tracking..."
echo "Experiment name: $EXPERIMENT_NAME"

# Create output directory
OUTPUT_DIR="experiments/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR"

# Log system information
echo "Logging system information..."
cat > "$OUTPUT_DIR/system_info.txt" << EOF
Training Session: $EXPERIMENT_NAME
Started: $(date)
Hostname: $(hostname)
Python: $(python3 --version)
PyTorch: $(python3 -c "import torch; print(torch.__version__)")

GPU Information:
$(nvidia-smi 2>/dev/null || echo "No NVIDIA GPU detected")

Configuration:
- Data Path: $DATA_PATH
- Batch Size: $BATCH_SIZE
- Learning Rate: $LEARNING_RATE
- Epochs: $EPOCHS
- Data Size: $data_size rows
EOF

# Setup monitoring
echo "Starting system monitoring..."
nohup bash -c '
while true; do
    echo "$(date): GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "N/A"), CPU: $(top -bn1 | grep "Cpu(s)" | awk "{print \$2}" | cut -d"%" -f1)"
    sleep 30
done' > "$OUTPUT_DIR/system_monitor.log" 2>&1 &

MONITOR_PID=$!
echo "✓ System monitoring started (PID: $MONITOR_PID)"

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    kill $MONITOR_PID 2>/dev/null
    echo "Training interrupted at $(date)" >> "$OUTPUT_DIR/training.log"
}

trap cleanup EXIT INT TERM

# Launch training with error handling
echo "============================================="
echo "🚀 Starting training..."
echo "============================================="
echo "Configuration:"
echo "  - Experiment: $EXPERIMENT_NAME"
echo "  - Data: $DATA_PATH ($data_size rows)"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Epochs: $EPOCHS"
echo "  - Output: $OUTPUT_DIR"
echo "============================================="

# Start training
python3 gpu_training_script.py \
    --data-path "$DATA_PATH" \
    --experiment-name "$EXPERIMENT_NAME" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Check training result
TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "============================================="
    echo "✅ Training completed successfully!"
    echo "============================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo "Training logs: $OUTPUT_DIR/training.log"
    echo "System monitor: $OUTPUT_DIR/system_monitor.log"
    
    # Generate training summary
    echo "Generating training summary..."
    python3 -c "
import json
import os
import glob

output_dir = '$OUTPUT_DIR'
checkpoints = glob.glob(os.path.join(output_dir, '../checkpoints_$EXPERIMENT_NAME', '*.pth'))
if checkpoints:
    print(f'✓ Checkpoints saved: {len(checkpoints)}')
    best_checkpoint = os.path.join(output_dir, '../checkpoints_$EXPERIMENT_NAME', 'best_model.pth')
    if os.path.exists(best_checkpoint):
        print(f'✓ Best model: {best_checkpoint}')
        import torch
        checkpoint = torch.load(best_checkpoint, map_location='cpu')
        val_metrics = checkpoint.get('val_metrics', {})
        print(f'✓ Best validation loss: {val_metrics.get(\"total_loss\", \"N/A\")}')
        print(f'✓ R² Score: {val_metrics.get(\"r2_score\", \"N/A\")}')
else:
    print('❌ No checkpoints found')
"
    
else
    echo "============================================="
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "============================================="
    echo "Check logs: $OUTPUT_DIR/training.log"
    echo "Check system monitor: $OUTPUT_DIR/system_monitor.log"
fi

# Cleanup
kill $MONITOR_PID 2>/dev/null

echo "Training session completed: $(date)" >> "$OUTPUT_DIR/system_info.txt"
echo "Exit code: $TRAINING_EXIT_CODE" >> "$OUTPUT_DIR/system_info.txt"

exit $TRAINING_EXIT_CODE