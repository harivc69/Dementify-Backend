#!/bin/bash
#
# Training script for tabular-only baseline model
# Run from nmed2024/dev directory
#

# Activate conda environment
# source activate abstract  # Uncomment if needed

# Training parameters
TRAIN_PATH="data/processed/train.csv"
VAL_PATH="data/processed/val.csv"
TEST_PATH="data/processed/test.csv"
CNF_FILE="data/toml_files/custom_nacc_config_no_img.toml"

# Model hyperparameters
D_MODEL=64
NHEAD=4
NUM_EPOCHS=50
BATCH_SIZE=32
LR=0.0001
GAMMA=2.0

# Checkpoint path
CKPT_PATH="checkpoints/baseline_model.pt"

# Training options
# Add --wandb to enable Weights & Biases logging
# Add --balanced_sampling to use balanced sampling
# Add --load_from_ckpt to resume from checkpoint

python train_baseline.py \
    --train_path $TRAIN_PATH \
    --val_path $VAL_PATH \
    --test_path $TEST_PATH \
    --cnf_file $CNF_FILE \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --gamma $GAMMA \
    --ckpt_path $CKPT_PATH \
    --save_intermediate_ckpts

