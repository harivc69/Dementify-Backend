#!/bin/bash
################################################################################
# Automated Training Queue Manager
# Purpose: Maximize epochs trained before 4:00 PM CST meeting deadline
# Created: November 3, 2025
################################################################################

set -e  # Exit on error

# Configuration
SCRIPT_DIR="/home/vatsal/MRI_with_additional_features/nmed2024/dev"
CONDA_PATH="/mnt/c/common_folder/miniconda3"
CONDA_ENV="abstract"
PYTHON_SCRIPT="train_expanded.py"
BATCH_SIZE=32
AVG_EPOCH_TIME_MINUTES=3.83
AVG_EPOCH_TIME_SECONDS=230
MIN_EPOCHS=5
BUFFER_MINUTES=10

# Deadline: 3:50 PM CST (4:00 PM - 10 min buffer)
DEADLINE_HOUR=15
DEADLINE_MINUTE=50

# Current training PID to monitor
CURRENT_PID=15811

# Log files
MASTER_LOG="${SCRIPT_DIR}/training_master_log_20251103.txt"
SCRIPT_LOG="${SCRIPT_DIR}/auto_training_queue_20251103.log"
SUMMARY_FILE="${SCRIPT_DIR}/training_summary_all_runs_20251103.txt"

# Run counter (starting at 2 since run 1 is currently running)
RUN_NUMBER=2

################################################################################
# Helper Functions
################################################################################

log_message() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" | tee -a "$SCRIPT_LOG"
}

get_current_time_minutes() {
    # Returns current time in minutes since midnight
    date '+%H * 60 + %M' | bc
}

get_deadline_minutes() {
    # Returns deadline time in minutes since midnight
    echo "$DEADLINE_HOUR * 60 + $DEADLINE_MINUTE" | bc
}

calculate_remaining_minutes() {
    local current_minutes=$(get_current_time_minutes)
    local deadline_minutes=$(get_deadline_minutes)
    echo "$deadline_minutes - $current_minutes" | bc
}

calculate_max_epochs() {
    local remaining_minutes=$1
    # Calculate max epochs that can fit in remaining time
    echo "scale=0; $remaining_minutes / $AVG_EPOCH_TIME_MINUTES" | bc
}

wait_for_process() {
    local pid=$1
    log_message "Waiting for process $pid to complete..."
    
    while kill -0 $pid 2>/dev/null; do
        sleep 30
    done
    
    log_message "Process $pid has completed"
}

extract_final_losses() {
    local log_file=$1
    
    # Extract last training and validation losses from log
    local train_loss=$(grep -i "train.*loss" "$log_file" | tail -1 | grep -oP '\d+\.\d+' | head -1 || echo "N/A")
    local val_loss=$(grep -i "val.*loss" "$log_file" | tail -1 | grep -oP '\d+\.\d+' | head -1 || echo "N/A")
    
    echo "$train_loss,$val_loss"
}

start_training_run() {
    local epochs=$1
    local run_num=$2
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local log_file="${SCRIPT_DIR}/training_meeting_run${run_num}_${timestamp}.log"
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    log_message "=========================================="
    log_message "Starting Training Run #${run_num}"
    log_message "Epochs: $epochs"
    log_message "Start Time: $start_time"
    log_message "Log File: $log_file"
    log_message "=========================================="
    
    # Start training in background
    cd "$SCRIPT_DIR"
    nohup ${CONDA_PATH}/envs/${CONDA_ENV}/bin/python -u ${PYTHON_SCRIPT} \
        --epochs $epochs \
        --batch_size $BATCH_SIZE \
        > "$log_file" 2>&1 &
    
    local new_pid=$!
    log_message "Training started with PID: $new_pid"
    
    # Wait for training to complete
    wait_for_process $new_pid
    
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    log_message "Training Run #${run_num} completed at $end_time"
    
    # Extract final losses
    sleep 2  # Give time for log file to be fully written
    local losses=$(extract_final_losses "$log_file")
    local train_loss=$(echo "$losses" | cut -d',' -f1)
    local val_loss=$(echo "$losses" | cut -d',' -f2)
    
    # Record to master log
    echo "========================================" >> "$MASTER_LOG"
    echo "Run Number: $run_num" >> "$MASTER_LOG"
    echo "Start Time: $start_time" >> "$MASTER_LOG"
    echo "End Time: $end_time" >> "$MASTER_LOG"
    echo "Epochs: $epochs" >> "$MASTER_LOG"
    echo "Final Training Loss: $train_loss" >> "$MASTER_LOG"
    echo "Final Validation Loss: $val_loss" >> "$MASTER_LOG"
    echo "Log File: $log_file" >> "$MASTER_LOG"
    echo "Checkpoint: ${SCRIPT_DIR}/checkpoints_corrected/final_model_epoch_${epochs}.pt" >> "$MASTER_LOG"
    echo "" >> "$MASTER_LOG"
    
    log_message "Run #${run_num} recorded to master log"
    
    return 0
}

generate_final_summary() {
    log_message "Generating final summary..."
    
    {
        echo "================================================================================"
        echo "AUTOMATED TRAINING QUEUE - FINAL SUMMARY"
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S CST')"
        echo "================================================================================"
        echo ""
        
        # Count total runs (including the initial run)
        local total_runs=$(grep -c "Run Number:" "$MASTER_LOG" 2>/dev/null || echo "0")
        echo "Total Training Runs Completed: $total_runs"
        echo ""
        
        # Calculate total epochs
        local total_epochs=0
        while IFS= read -r line; do
            if [[ $line =~ Epochs:\ ([0-9]+) ]]; then
                total_epochs=$((total_epochs + ${BASH_REMATCH[1]}))
            fi
        done < "$MASTER_LOG"
        echo "Total Epochs Trained: $total_epochs"
        echo ""
        
        # Find best validation loss
        echo "Best Validation Loss:"
        grep "Final Validation Loss:" "$MASTER_LOG" | grep -oP '\d+\.\d+' | sort -n | head -1 || echo "N/A"
        echo ""
        
        echo "================================================================================"
        echo "DETAILED RUN BREAKDOWN"
        echo "================================================================================"
        cat "$MASTER_LOG"
        echo ""
        
        echo "================================================================================"
        echo "CHECKPOINT FILES"
        echo "================================================================================"
        ls -lh "${SCRIPT_DIR}/checkpoints_corrected/"*.pt 2>/dev/null || echo "No checkpoints found"
        echo ""
        
        echo "================================================================================"
        echo "TIMING BREAKDOWN"
        echo "================================================================================"
        grep -E "(Run Number:|Start Time:|End Time:|Epochs:)" "$MASTER_LOG"
        
    } > "$SUMMARY_FILE"
    
    log_message "Final summary saved to: $SUMMARY_FILE"
}

################################################################################
# Main Execution
################################################################################

main() {
    log_message "================================================================================"
    log_message "AUTOMATED TRAINING QUEUE MANAGER STARTED"
    log_message "================================================================================"
    log_message "Script Directory: $SCRIPT_DIR"
    log_message "Conda Environment: $CONDA_ENV"
    log_message "Deadline: ${DEADLINE_HOUR}:${DEADLINE_MINUTE} CST (with ${BUFFER_MINUTES} min buffer)"
    log_message "Average Epoch Time: ${AVG_EPOCH_TIME_MINUTES} minutes"
    log_message "Minimum Epochs per Run: $MIN_EPOCHS"
    log_message "================================================================================"
    
    # Initialize master log
    {
        echo "================================================================================"
        echo "TRAINING MASTER LOG - November 3, 2025"
        echo "Goal: Maximize epochs before 4:00 PM CST meeting"
        echo "================================================================================"
        echo ""
    } > "$MASTER_LOG"
    
    # Step 1: Wait for current training (Run #1, PID 15811) to complete
    log_message "Step 1: Monitoring current training run (PID: $CURRENT_PID, 35 epochs)"
    
    if kill -0 $CURRENT_PID 2>/dev/null; then
        wait_for_process $CURRENT_PID
        log_message "Initial training run (35 epochs) has completed"
        
        # Record Run #1 to master log
        echo "========================================" >> "$MASTER_LOG"
        echo "Run Number: 1" >> "$MASTER_LOG"
        echo "Start Time: 2025-11-03 12:34:26" >> "$MASTER_LOG"
        echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$MASTER_LOG"
        echo "Epochs: 35" >> "$MASTER_LOG"
        echo "Log File: ${SCRIPT_DIR}/training_meeting_4hr_20251103_123236.log" >> "$MASTER_LOG"
        echo "Checkpoint: ${SCRIPT_DIR}/checkpoints_corrected/final_model_epoch_35.pt" >> "$MASTER_LOG"
        echo "" >> "$MASTER_LOG"
    else
        log_message "WARNING: Process $CURRENT_PID not found. It may have already completed."
    fi
    
    # Step 2: Start sequential training runs
    log_message "Step 2: Starting sequential training runs"
    
    while true; do
        # Calculate remaining time
        local remaining_minutes=$(calculate_remaining_minutes)
        log_message "Time remaining until deadline: $remaining_minutes minutes"
        
        if [ $remaining_minutes -le 0 ]; then
            log_message "Deadline reached. Stopping training queue."
            break
        fi
        
        # Calculate max epochs for next run
        local max_epochs=$(calculate_max_epochs $remaining_minutes)
        log_message "Maximum epochs that can fit: $max_epochs"
        
        if [ $max_epochs -lt $MIN_EPOCHS ]; then
            log_message "Insufficient time for minimum $MIN_EPOCHS epochs. Stopping queue."
            break
        fi
        
        # Start next training run
        start_training_run $max_epochs $RUN_NUMBER
        
        # Increment run counter
        RUN_NUMBER=$((RUN_NUMBER + 1))
        
        # Brief pause between runs
        sleep 5
    done
    
    # Step 3: Generate final summary
    log_message "Step 3: Generating final summary"
    generate_final_summary
    
    log_message "================================================================================"
    log_message "AUTOMATED TRAINING QUEUE MANAGER COMPLETED"
    log_message "Total runs executed: $((RUN_NUMBER - 1))"
    log_message "Summary file: $SUMMARY_FILE"
    log_message "Master log: $MASTER_LOG"
    log_message "================================================================================"
}

# Execute main function
main

exit 0

