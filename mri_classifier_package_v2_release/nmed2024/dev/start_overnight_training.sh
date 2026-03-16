#!/bin/bash

# Overnight training startup script with robust process management
# This script ensures training continues even if terminal disconnects

set -e

echo "================================================================================"
echo "OVERNIGHT TRAINING STARTUP - OPTIMIZED FOR MAXIMUM PERFORMANCE"
echo "================================================================================"
echo ""
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Navigate to correct directory
cd /home/vatsal/MRI_with_additional_features/nmed2024/dev

# Create logs directory
mkdir -p ../../logs_overnight

# Set log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="../../logs_overnight/training_${TIMESTAMP}.log"
PID_FILE="../../logs_overnight/training.pid"
STATUS_FILE="../../logs_overnight/training_status.txt"

echo "Configuration:"
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo "  Status file: $STATUS_FILE"
echo ""

# Save startup info
cat > $STATUS_FILE << EOF
Training Status
===============
Start Time: $(date '+%Y-%m-%d %H:%M:%S')
Status: STARTING
Log File: $LOG_FILE
PID File: $PID_FILE
Checkpoint Dir: checkpoints_overnight
Expected Duration: 12-14 hours
Target Epochs: 150
Model: d_model=128, nhead=8 (optimized)
EOF

echo "Starting training with nohup (will continue if terminal disconnects)..."
echo ""

# Start training with nohup
nohup /mnt/c/common_folder/miniconda3/envs/abstract/bin/python -u train_overnight.py \
    --num_epochs 150 \
    --d_model 128 \
    --nhead 8 \
    --lr 5e-5 \
    --weight_decay 1e-5 \
    --batch_size 32 \
    --ckpt_interval 10 \
    > $LOG_FILE 2>&1 &

# Get PID
TRAIN_PID=$!
echo $TRAIN_PID > $PID_FILE

echo "✓ Training started successfully!"
echo ""
echo "Process Information:"
echo "  PID: $TRAIN_PID"
echo "  Log file: $LOG_FILE"
echo ""
echo "Training is now running in the background and will continue even if you"
echo "disconnect from the terminal."
echo ""
echo "================================================================================"
echo "MONITORING COMMANDS"
echo "================================================================================"
echo ""
echo "Check if training is running:"
echo "  ps aux | grep $TRAIN_PID | grep -v grep"
echo ""
echo "View live training output:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Check checkpoints:"
echo "  ls -lh checkpoints_overnight/"
echo ""
echo "Check training progress log:"
echo "  cat checkpoints_overnight/training_progress.log"
echo ""
echo "Stop training gracefully:"
echo "  kill -SIGINT $TRAIN_PID"
echo ""
echo "================================================================================"
echo ""
echo "Waiting 10 seconds before starting monitoring phase..."
sleep 10

# Update status
cat >> $STATUS_FILE << EOF

Process Started
===============
PID: $TRAIN_PID
Status: RUNNING
Time: $(date '+%Y-%m-%d %H:%M:%S')
EOF

echo ""
echo "✓ Training process is running"
echo "✓ Beginning 15-minute monitoring phase..."
echo ""

