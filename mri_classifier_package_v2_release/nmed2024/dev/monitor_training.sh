#!/bin/bash

# Training monitoring script
# Monitors training for 15 minutes and generates status report

echo "================================================================================"
echo "TRAINING MONITORING - 15 MINUTE CHECK"
echo "================================================================================"
echo ""

# Get PID from file
PID_FILE="../../logs_overnight/training.pid"
if [ -f "$PID_FILE" ]; then
    TRAIN_PID=$(cat $PID_FILE)
    echo "Training PID: $TRAIN_PID"
else
    echo "ERROR: PID file not found!"
    exit 1
fi

# Function to check if process is running
check_process() {
    if ps -p $TRAIN_PID > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to get GPU stats
get_gpu_stats() {
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
}

# Function to get process stats
get_process_stats() {
    ps -p $TRAIN_PID -o %cpu,%mem,etime,rss --no-headers
}

echo ""
echo "Starting 15-minute monitoring..."
echo "Checks will be performed at: 1, 5, 10, and 15 minutes"
echo ""

REPORT_FILE="../../logs_overnight/monitoring_report_$(date +%Y%m%d_%H%M%S).txt"

# Initialize report
cat > $REPORT_FILE << EOF
OVERNIGHT TRAINING - 15 MINUTE MONITORING REPORT
================================================
Generated: $(date '+%Y-%m-%d %H:%M:%S')
Training PID: $TRAIN_PID

EOF

# Monitoring loop
for MINUTE in 1 5 10 15; do
    echo "--- Check at ${MINUTE} minute(s) ---"
    
    # Wait until the target minute
    if [ $MINUTE -gt 1 ]; then
        PREV_MINUTE=$((MINUTE - $([ $MINUTE -eq 5 ] && echo 4 || [ $MINUTE -eq 10 ] && echo 5 || echo 5)))
        WAIT_TIME=$((60 * (MINUTE - $([ $MINUTE -eq 1 ] && echo 0 || [ $MINUTE -eq 5 ] && echo 1 || [ $MINUTE -eq 10 ] && echo 5 || echo 10))))
        sleep $WAIT_TIME
    else
        sleep 60
    fi
    
    echo ""
    echo "=== Minute $MINUTE Check ===" | tee -a $REPORT_FILE
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $REPORT_FILE
    echo "" | tee -a $REPORT_FILE
    
    # Check if process is still running
    if check_process; then
        echo "✓ Process Status: RUNNING" | tee -a $REPORT_FILE
        
        # Get process stats
        PROC_STATS=$(get_process_stats)
        CPU=$(echo $PROC_STATS | awk '{print $1}')
        MEM=$(echo $PROC_STATS | awk '{print $2}')
        ETIME=$(echo $PROC_STATS | awk '{print $3}')
        RSS=$(echo $PROC_STATS | awk '{print $4}')
        RSS_GB=$(echo "scale=2; $RSS / 1024 / 1024" | bc)
        
        echo "  CPU Usage: ${CPU}%" | tee -a $REPORT_FILE
        echo "  Memory: ${MEM}% (${RSS_GB} GB)" | tee -a $REPORT_FILE
        echo "  Running Time: ${ETIME}" | tee -a $REPORT_FILE
        
        # Get GPU stats
        if command -v nvidia-smi &> /dev/null; then
            GPU_STATS=$(get_gpu_stats)
            GPU_UTIL=$(echo $GPU_STATS | cut -d',' -f1)
            GPU_MEM_USED=$(echo $GPU_STATS | cut -d',' -f2)
            GPU_MEM_TOTAL=$(echo $GPU_STATS | cut -d',' -f3)
            GPU_TEMP=$(echo $GPU_STATS | cut -d',' -f4)
            GPU_POWER=$(echo $GPU_STATS | cut -d',' -f5)
            
            echo "" | tee -a $REPORT_FILE
            echo "✓ GPU Status:" | tee -a $REPORT_FILE
            echo "  Utilization: ${GPU_UTIL}%" | tee -a $REPORT_FILE
            echo "  Memory: ${GPU_MEM_USED} MB / ${GPU_MEM_TOTAL} MB" | tee -a $REPORT_FILE
            echo "  Temperature: ${GPU_TEMP}°C" | tee -a $REPORT_FILE
            echo "  Power: ${GPU_POWER}W" | tee -a $REPORT_FILE
        fi
        
        # Check checkpoints
        if [ -d "checkpoints_overnight" ]; then
            CKPT_COUNT=$(ls -1 checkpoints_overnight/*.pt 2>/dev/null | wc -l)
            echo "" | tee -a $REPORT_FILE
            echo "✓ Checkpoints: ${CKPT_COUNT} files" | tee -a $REPORT_FILE
            if [ -f "checkpoints_overnight/training_progress.log" ]; then
                LAST_LOG=$(tail -1 checkpoints_overnight/training_progress.log)
                echo "  Last log: ${LAST_LOG}" | tee -a $REPORT_FILE
            fi
        fi
        
        # Check log file for errors
        LOG_FILE=$(ls -t ../../logs_overnight/training_*.log 2>/dev/null | head -1)
        if [ -f "$LOG_FILE" ]; then
            ERROR_COUNT=$(grep -i "error\|exception\|failed" $LOG_FILE 2>/dev/null | wc -l)
            if [ $ERROR_COUNT -gt 0 ]; then
                echo "" | tee -a $REPORT_FILE
                echo "⚠ Errors detected: ${ERROR_COUNT}" | tee -a $REPORT_FILE
                echo "  Check log file: $LOG_FILE" | tee -a $REPORT_FILE
            else
                echo "" | tee -a $REPORT_FILE
                echo "✓ No errors detected in log" | tee -a $REPORT_FILE
            fi
        fi
        
    else
        echo "✗ Process Status: NOT RUNNING" | tee -a $REPORT_FILE
        echo "" | tee -a $REPORT_FILE
        echo "Training process has stopped!" | tee -a $REPORT_FILE
        echo "Check log file for details:" | tee -a $REPORT_FILE
        LOG_FILE=$(ls -t ../../logs_overnight/training_*.log 2>/dev/null | head -1)
        echo "  $LOG_FILE" | tee -a $REPORT_FILE
        break
    fi
    
    echo "" | tee -a $REPORT_FILE
    echo "---" | tee -a $REPORT_FILE
    echo ""
done

# Final summary
echo "" | tee -a $REPORT_FILE
echo "================================================================================" | tee -a $REPORT_FILE
echo "15-MINUTE MONITORING COMPLETE" | tee -a $REPORT_FILE
echo "================================================================================" | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

if check_process; then
    echo "✓ TRAINING IS RUNNING SUCCESSFULLY" | tee -a $REPORT_FILE
    echo "" | tee -a $REPORT_FILE
    echo "Training is stable and will continue overnight." | tee -a $REPORT_FILE
    echo "You can safely disconnect from the terminal." | tee -a $REPORT_FILE
    echo "" | tee -a $REPORT_FILE
    echo "To check progress in the morning:" | tee -a $REPORT_FILE
    echo "  1. Check if still running: ps aux | grep $TRAIN_PID" | tee -a $REPORT_FILE
    echo "  2. View log: tail -100 $LOG_FILE" | tee -a $REPORT_FILE
    echo "  3. Check checkpoints: ls -lh checkpoints_overnight/" | tee -a $REPORT_FILE
    echo "  4. View progress: cat checkpoints_overnight/training_progress.log" | tee -a $REPORT_FILE
else
    echo "✗ TRAINING HAS STOPPED" | tee -a $REPORT_FILE
    echo "" | tee -a $REPORT_FILE
    echo "Please check the log file for errors:" | tee -a $REPORT_FILE
    echo "  $LOG_FILE" | tee -a $REPORT_FILE
fi

echo "" | tee -a $REPORT_FILE
echo "Full monitoring report saved to: $REPORT_FILE" | tee -a $REPORT_FILE
echo ""

# Display report location
echo "================================================================================"
echo "Monitoring report saved to: $REPORT_FILE"
echo "================================================================================"

