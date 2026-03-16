#!/bin/bash

# Full training script with monitoring and error handling
# This script runs the complete training pipeline with proper checkpoint management

set -e  # Exit on error

echo "================================================================================"
echo "NMED2024 TABULAR-ONLY BASELINE TRAINING"
echo "================================================================================"
echo ""
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Navigate to correct directory
cd /home/vatsal/MRI_with_additional_features/nmed2024/dev

# Activate conda environment
echo "Activating conda environment 'abstract'..."
eval "$(conda shell.bash hook)"
conda activate abstract

# Verify environment
echo ""
echo "Verifying environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "================================================================================"
echo "STEP 1: RUNNING 5-EPOCH TEST"
echo "================================================================================"
echo ""

# Run 5-epoch test
python test_training_run.py 2>&1 | tee ../../logs/test_run_$(date +%Y%m%d_%H%M%S).log

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "ERROR: Test run failed with exit code $TEST_EXIT_CODE"
    echo "================================================================================"
    exit $TEST_EXIT_CODE
fi

echo ""
echo "================================================================================"
echo "TEST RUN COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "Waiting 10 seconds before starting full training..."
sleep 10

echo ""
echo "================================================================================"
echo "STEP 2: RUNNING FULL BASELINE TRAINING (50 EPOCHS)"
echo "================================================================================"
echo ""

# Create logs directory
mkdir -p ../../logs

# Run full training with enhanced checkpointing
python train_baseline_enhanced.py \
    --num_epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --d_model 64 \
    --nhead 4 \
    --gamma 2.0 \
    --ckpt_dir checkpoints \
    --ckpt_interval 10 \
    2>&1 | tee ../../logs/full_training_$(date +%Y%m%d_%H%M%S).log

TRAIN_EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "TRAINING COMPLETED SUCCESSFULLY"
else
    echo "TRAINING FAILED WITH EXIT CODE $TRAIN_EXIT_CODE"
fi
echo "================================================================================"
echo ""
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Show checkpoint files
echo "Checkpoint files created:"
ls -lh checkpoints/

echo ""
echo "Log files created:"
ls -lh ../../logs/

exit $TRAIN_EXIT_CODE

