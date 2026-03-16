#!/bin/bash

# Real-time training progress monitor
# Shows epoch count updating in real-time

echo "================================================================================"
echo "REAL-TIME TRAINING PROGRESS MONITOR"
echo "================================================================================"
echo ""
echo "Monitoring training progress... (Press Ctrl+C to stop)"
echo ""

LOG_FILE="../../logs_overnight/training_20251012_230733.log"
PID=2151444
TARGET_EPOCHS=150

while true; do
    # Clear screen
    clear
    
    echo "================================================================================"
    echo "OVERNIGHT TRAINING - REAL-TIME PROGRESS"
    echo "================================================================================"
    echo ""
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check if process is running
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Process Status: RUNNING (PID: $PID)"
        
        # Get process stats
        PROC_STATS=$(ps -p $PID -o etime,%cpu,%mem,rss --no-headers)
        ETIME=$(echo $PROC_STATS | awk '{print $1}')
        CPU=$(echo $PROC_STATS | awk '{print $2}')
        MEM=$(echo $PROC_STATS | awk '{print $3}')
        RSS=$(echo $PROC_STATS | awk '{print $4}')
        RSS_GB=$(awk "BEGIN {printf \"%.2f\", $RSS / 1024 / 1024}")
        
        echo "  Running Time: $ETIME"
        echo "  CPU Usage: ${CPU}%"
        echo "  Memory: ${MEM}% (${RSS_GB} GB)"
    else
        echo "✗ Process Status: NOT RUNNING"
        echo ""
        echo "Training has completed or stopped."
        break
    fi
    
    echo ""
    echo "================================================================================"
    echo "TRAINING PROGRESS"
    echo "================================================================================"
    echo ""
    
    # Count epochs
    if [ -f "$LOG_FILE" ]; then
        EPOCHS=$(grep -c "best_model.pt" "$LOG_FILE")
        PROGRESS=$((EPOCHS * 100 / TARGET_EPOCHS))
        REMAINING=$((TARGET_EPOCHS - EPOCHS))

        echo "  Epochs Completed: $EPOCHS / $TARGET_EPOCHS"
        echo "  Progress: ${PROGRESS}%"
        echo "  Remaining: $REMAINING epochs"

        # Calculate time estimates
        if [ $EPOCHS -gt 0 ]; then
            # Convert elapsed time to seconds
            ETIME_SEC=$(echo $ETIME | awk -F: '{ if (NF==3) print ($1 * 3600) + ($2 * 60) + $3; else if (NF==2) print ($1 * 60) + $2; else print $1 }')
            TIME_PER_EPOCH=$((ETIME_SEC / EPOCHS))
            REMAINING_SEC=$((REMAINING * TIME_PER_EPOCH))
            REMAINING_MIN=$((REMAINING_SEC / 60))

            echo ""
            echo "  Time per Epoch: ${TIME_PER_EPOCH} seconds"
            echo "  Estimated Remaining: ${REMAINING_MIN} minutes"

            # Calculate completion time
            COMPLETION_TIME=$(date -d "+${REMAINING_MIN} minutes" '+%H:%M' 2>/dev/null || echo "N/A")
            echo "  Expected Completion: ${COMPLETION_TIME}"
        fi

        # Progress bar
        echo ""
        PROGRESS_INT=$PROGRESS
        BAR_LENGTH=50
        FILLED=$((PROGRESS_INT * BAR_LENGTH / 100))
        EMPTY=$((BAR_LENGTH - FILLED))
        
        printf "  ["
        for i in $(seq 1 $FILLED); do printf "="; done
        printf ">"
        for i in $(seq 1 $EMPTY); do printf " "; done
        printf "] ${PROGRESS}%%\n"
        
    else
        echo "  Log file not found!"
    fi
    
    echo ""
    echo "================================================================================"
    echo "GPU STATUS"
    echo "================================================================================"
    echo ""
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
        if [ -n "$GPU_STATS" ]; then
            GPU_UTIL=$(echo $GPU_STATS | cut -d',' -f1)
            GPU_MEM_USED=$(echo $GPU_STATS | cut -d',' -f2)
            GPU_MEM_TOTAL=$(echo $GPU_STATS | cut -d',' -f3)
            GPU_TEMP=$(echo $GPU_STATS | cut -d',' -f4)
            GPU_POWER=$(echo $GPU_STATS | cut -d',' -f5)
            
            echo "  GPU Utilization: ${GPU_UTIL}%"
            echo "  GPU Memory: ${GPU_MEM_USED} MB / ${GPU_MEM_TOTAL} MB"
            echo "  Temperature: ${GPU_TEMP}°C"
            echo "  Power: ${GPU_POWER}W"
        fi
    fi
    
    echo ""
    echo "================================================================================"
    echo "Refreshing every 10 seconds... (Press Ctrl+C to stop)"
    echo "================================================================================"
    
    # Wait 10 seconds
    sleep 10
done

echo ""
echo "Monitoring stopped."

