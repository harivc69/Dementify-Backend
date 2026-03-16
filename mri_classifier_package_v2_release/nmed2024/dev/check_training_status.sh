#!/bin/bash

# Quick training status check script
# Use this in the morning to check if training completed

echo "================================================================================"
echo "OVERNIGHT TRAINING STATUS CHECK"
echo "================================================================================"
echo ""
echo "Current Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Get PID
PID_FILE="../../logs_overnight/training.pid"
if [ -f "$PID_FILE" ]; then
    TRAIN_PID=$(cat $PID_FILE)
    echo "Training PID: $TRAIN_PID"
else
    echo "ERROR: PID file not found!"
    echo "Training may not have started properly."
    exit 1
fi

echo ""
echo "================================================================================"
echo "1. PROCESS STATUS"
echo "================================================================================"
echo ""

# Check if process is running
if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "✓ Process Status: RUNNING"
    echo ""
    
    # Get process stats
    PROC_STATS=$(ps -p $TRAIN_PID -o %cpu,%mem,etime,rss --no-headers)
    CPU=$(echo $PROC_STATS | awk '{print $1}')
    MEM=$(echo $PROC_STATS | awk '{print $2}')
    ETIME=$(echo $PROC_STATS | awk '{print $3}')
    RSS=$(echo $PROC_STATS | awk '{print $4}')
    RSS_GB=$(echo "scale=2; $RSS / 1024 / 1024" | bc)
    
    echo "  CPU Usage: ${CPU}%"
    echo "  Memory: ${MEM}% (${RSS_GB} GB)"
    echo "  Running Time: ${ETIME}"
    echo ""
    echo "Training is still in progress..."
    
else
    echo "✗ Process Status: NOT RUNNING"
    echo ""
    echo "Training has completed or stopped."
fi

echo ""
echo "================================================================================"
echo "2. GPU STATUS"
echo "================================================================================"
echo ""

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv
else
    echo "nvidia-smi not available"
fi

echo ""
echo "================================================================================"
echo "3. CHECKPOINTS"
echo "================================================================================"
echo ""

if [ -d "checkpoints_overnight" ]; then
    echo "Checkpoint files:"
    ls -lh checkpoints_overnight/*.pt 2>/dev/null || echo "  No .pt files yet"
    echo ""
    echo "All files:"
    ls -lh checkpoints_overnight/
else
    echo "Checkpoint directory not found!"
fi

echo ""
echo "================================================================================"
echo "4. TRAINING PROGRESS"
echo "================================================================================"
echo ""

if [ -f "checkpoints_overnight/training_progress.log" ]; then
    echo "Last 10 log entries:"
    tail -10 checkpoints_overnight/training_progress.log
else
    echo "Progress log not found!"
fi

echo ""
echo "================================================================================"
echo "5. TRAINING SUMMARY"
echo "================================================================================"
echo ""

if [ -f "checkpoints_overnight/training_summary.json" ]; then
    echo "✓ Training completed!"
    echo ""
    cat checkpoints_overnight/training_summary.json
else
    echo "Training summary not yet available (training may still be running)"
fi

echo ""
echo "================================================================================"
echo "6. LOG FILE"
echo "================================================================================"
echo ""

LOG_FILE=$(ls -t ../../logs_overnight/training_*.log 2>/dev/null | head -1)
if [ -f "$LOG_FILE" ]; then
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Last 50 lines:"
    tail -50 "$LOG_FILE"
    echo ""
    echo "Errors/Warnings:"
    ERROR_COUNT=$(grep -i "error\|exception\|failed" "$LOG_FILE" 2>/dev/null | wc -l)
    if [ $ERROR_COUNT -gt 0 ]; then
        echo "  ⚠ Found $ERROR_COUNT error/exception lines"
        echo "  Last 5 errors:"
        grep -i "error\|exception\|failed" "$LOG_FILE" | tail -5
    else
        echo "  ✓ No errors detected"
    fi
else
    echo "Log file not found!"
fi

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""

# Determine overall status
if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "Status: TRAINING IN PROGRESS"
    echo ""
    echo "Training is still running. Check back later or let it continue."
    echo ""
    echo "To monitor live:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "To stop gracefully:"
    echo "  kill -SIGINT $TRAIN_PID"
    
elif [ -f "checkpoints_overnight/training_summary.json" ]; then
    echo "Status: TRAINING COMPLETED SUCCESSFULLY ✓"
    echo ""
    echo "Check the training summary above for results."
    echo ""
    echo "Next steps:"
    echo "  1. Review training_summary.json"
    echo "  2. Load the best model: checkpoints_overnight/best_model.pt"
    echo "  3. Evaluate on test set"
    
else
    echo "Status: TRAINING STOPPED (may have failed)"
    echo ""
    echo "Training stopped but no summary found."
    echo "Check the log file for errors:"
    echo "  $LOG_FILE"
    echo ""
    echo "To resume training:"
    echo "  python train_overnight.py --resume_from checkpoints_overnight/best_model.pt"
fi

echo ""
echo "================================================================================"

