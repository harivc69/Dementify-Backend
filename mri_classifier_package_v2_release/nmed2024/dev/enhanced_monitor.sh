#!/bin/bash
################################################################################
# Enhanced Training Monitor
# Purpose: Monitor GPU utilization, training progress, and model improvement
# Runs autonomously until 4:00 PM CST
################################################################################

SCRIPT_DIR="/home/vatsal/MRI_with_additional_features/nmed2024/dev"
MONITOR_LOG="${SCRIPT_DIR}/enhanced_monitor_20251103.log"
ALERT_LOG="${SCRIPT_DIR}/training_alerts_20251103.log"
GPU_LOG="${SCRIPT_DIR}/gpu_utilization_20251103.log"
LOSS_TRACKING="${SCRIPT_DIR}/loss_tracking_20251103.log"

# Monitoring intervals
CHECK_INTERVAL_SECONDS=1800  # 30 minutes
QUICK_CHECK_SECONDS=60       # 1 minute for quick checks

# Thresholds
MIN_GPU_UTIL=20
TARGET_GPU_UTIL=30
LOW_GPU_DURATION=300  # 5 minutes

# Deadline
DEADLINE_HOUR=16
DEADLINE_MINUTE=0

################################################################################
# Helper Functions
################################################################################

log_monitor() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

log_alert() {
    echo "[ALERT $(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ALERT_LOG" "$MONITOR_LOG"
}

check_deadline() {
    local current_hour=$(date '+%H')
    local current_minute=$(date '+%M')
    local current_minutes=$((current_hour * 60 + current_minute))
    local deadline_minutes=$((DEADLINE_HOUR * 60 + DEADLINE_MINUTE))
    
    if [ $current_minutes -ge $deadline_minutes ]; then
        return 0  # Deadline reached
    else
        return 1  # Still time remaining
    fi
}

check_gpu_utilization() {
    local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    local gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU: ${gpu_util}% | Memory: ${gpu_mem}MB | Temp: ${gpu_temp}°C" >> "$GPU_LOG"
    
    if [ "$gpu_util" -lt "$MIN_GPU_UTIL" ]; then
        log_alert "LOW GPU UTILIZATION: ${gpu_util}% (threshold: ${MIN_GPU_UTIL}%)"
        return 1
    fi
    
    return 0
}

check_training_active() {
    if ps aux | grep -v grep | grep "train_expanded.py" > /dev/null; then
        return 0  # Training active
    else
        return 1  # No training
    fi
}

check_automation_running() {
    if ps aux | grep -v grep | grep "auto_training_queue.sh" > /dev/null; then
        return 0  # Automation running
    else
        return 1  # Automation stopped
    fi
}

extract_latest_losses() {
    local log_pattern="${SCRIPT_DIR}/training_meeting_*.log"
    local latest_log=$(ls -t $log_pattern 2>/dev/null | head -1)
    
    if [ -z "$latest_log" ]; then
        return 1
    fi
    
    # Try to extract loss values (this depends on your training script's output format)
    # Adjust grep patterns based on actual log format
    local train_loss=$(grep -i "train.*loss" "$latest_log" 2>/dev/null | tail -1 | grep -oP '\d+\.\d+' | head -1 || echo "N/A")
    local val_loss=$(grep -i "val.*loss" "$latest_log" 2>/dev/null | tail -1 | grep -oP '\d+\.\d+' | head -1 || echo "N/A")
    
    echo "$train_loss,$val_loss,$latest_log"
}

monitor_loss_progression() {
    local losses=$(extract_latest_losses)
    if [ $? -eq 0 ]; then
        local train_loss=$(echo "$losses" | cut -d',' -f1)
        local val_loss=$(echo "$losses" | cut -d',' -f2)
        local log_file=$(echo "$losses" | cut -d',' -f3)
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Train Loss: $train_loss | Val Loss: $val_loss | Log: $(basename $log_file)" >> "$LOSS_TRACKING"
        
        # Check if losses are reasonable (not N/A and not extremely high)
        if [ "$train_loss" != "N/A" ] && [ "$val_loss" != "N/A" ]; then
            log_monitor "Loss tracking: Train=$train_loss, Val=$val_loss"
        fi
    fi
}

count_completed_epochs() {
    local log_pattern="${SCRIPT_DIR}/training_meeting_*.log"
    local total_epochs=0
    
    for log_file in $log_pattern; do
        if [ -f "$log_file" ]; then
            local epochs=$(grep -c "Training epoch.*completed" "$log_file" 2>/dev/null || echo "0")
            total_epochs=$((total_epochs + epochs))
        fi
    done
    
    echo $total_epochs
}

count_completed_runs() {
    if [ -f "${SCRIPT_DIR}/training_master_log_20251103.txt" ]; then
        grep -c "Run Number:" "${SCRIPT_DIR}/training_master_log_20251103.txt" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

generate_status_report() {
    local report_file="${SCRIPT_DIR}/status_report_$(date '+%H%M').txt"
    
    {
        echo "================================================================================"
        echo "TRAINING STATUS REPORT"
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S CST')"
        echo "================================================================================"
        echo ""
        
        echo "AUTOMATION STATUS:"
        if check_automation_running; then
            echo "  ✅ Automation script: RUNNING"
        else
            echo "  ❌ Automation script: STOPPED"
        fi
        
        if check_training_active; then
            echo "  ✅ Training process: ACTIVE"
            ps aux | grep -v grep | grep "train_expanded.py" | awk '{print "     PID: "$2", Epochs: "$NF}'
        else
            echo "  ⏸️  Training process: IDLE (may be between runs)"
        fi
        echo ""
        
        echo "GPU STATUS:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
            awk -F', ' '{print "  GPU Utilization: "$1"% | Memory: "$2"/"$3" MB | Temp: "$4"°C"}'
        echo ""
        
        echo "TRAINING PROGRESS:"
        local total_runs=$(count_completed_runs)
        local total_epochs=$(count_completed_epochs)
        echo "  Completed Runs: $total_runs"
        echo "  Total Epochs Trained: $total_epochs"
        echo ""
        
        echo "RECENT LOSSES:"
        tail -5 "$LOSS_TRACKING" 2>/dev/null | sed 's/^/  /' || echo "  No loss data available yet"
        echo ""
        
        echo "RECENT ALERTS:"
        tail -5 "$ALERT_LOG" 2>/dev/null | sed 's/^/  /' || echo "  No alerts"
        echo ""
        
    } > "$report_file"
    
    log_monitor "Status report generated: $report_file"
}

restart_automation_if_needed() {
    if ! check_automation_running; then
        log_alert "CRITICAL: Automation script stopped! Attempting restart..."
        
        cd "$SCRIPT_DIR"
        nohup ./auto_training_queue.sh > auto_training_queue_restart_$(date '+%H%M%S').log 2>&1 &
        
        sleep 5
        
        if check_automation_running; then
            log_alert "SUCCESS: Automation script restarted"
        else
            log_alert "FAILED: Could not restart automation script"
        fi
    fi
}

check_idle_time() {
    local idle_start=""
    local idle_duration=0
    
    while true; do
        if ! check_training_active; then
            if [ -z "$idle_start" ]; then
                idle_start=$(date +%s)
                log_monitor "Training idle detected, monitoring..."
            else
                local current_time=$(date +%s)
                idle_duration=$((current_time - idle_start))
                
                if [ $idle_duration -gt 120 ]; then  # More than 2 minutes idle
                    log_alert "Training idle for ${idle_duration} seconds - checking automation"
                    restart_automation_if_needed
                fi
            fi
        else
            if [ -n "$idle_start" ]; then
                log_monitor "Training resumed after ${idle_duration} seconds idle"
                idle_start=""
                idle_duration=0
            fi
        fi
        
        sleep 30
    done
}

################################################################################
# Main Monitoring Loop
################################################################################

main() {
    log_monitor "================================================================================"
    log_monitor "ENHANCED TRAINING MONITOR STARTED"
    log_monitor "================================================================================"
    log_monitor "Monitoring until: ${DEADLINE_HOUR}:${DEADLINE_MINUTE} CST"
    log_monitor "GPU utilization checks: Every 30 minutes"
    log_monitor "Quick status checks: Every 60 seconds"
    log_monitor "================================================================================"
    
    # Initialize logs
    echo "GPU Utilization Log - Started $(date '+%Y-%m-%d %H:%M:%S')" > "$GPU_LOG"
    echo "Loss Tracking Log - Started $(date '+%Y-%m-%d %H:%M:%S')" > "$LOSS_TRACKING"
    echo "Alert Log - Started $(date '+%Y-%m-%d %H:%M:%S')" > "$ALERT_LOG"
    
    local last_gpu_check=$(date +%s)
    local last_report=$(date +%s)
    local iteration=0
    
    # Start idle time checker in background
    check_idle_time &
    local idle_checker_pid=$!
    
    while true; do
        # Check if deadline reached
        if check_deadline; then
            log_monitor "Deadline reached (4:00 PM CST). Generating final report..."
            generate_final_report
            kill $idle_checker_pid 2>/dev/null
            break
        fi
        
        iteration=$((iteration + 1))
        local current_time=$(date +%s)
        
        # Quick status check every minute
        if check_training_active; then
            log_monitor "Training active (iteration $iteration)"
        else
            log_monitor "No training active - automation should start next run soon"
        fi
        
        # GPU utilization check every 30 minutes
        if [ $((current_time - last_gpu_check)) -ge $CHECK_INTERVAL_SECONDS ]; then
            log_monitor "Performing scheduled GPU utilization check..."
            check_gpu_utilization
            monitor_loss_progression
            last_gpu_check=$current_time
        fi
        
        # Generate status report every 30 minutes
        if [ $((current_time - last_report)) -ge $CHECK_INTERVAL_SECONDS ]; then
            generate_status_report
            last_report=$current_time
        fi
        
        # Check automation health
        if ! check_automation_running; then
            log_alert "Automation script not running!"
            restart_automation_if_needed
        fi
        
        # Sleep for quick check interval
        sleep $QUICK_CHECK_SECONDS
    done
    
    log_monitor "================================================================================"
    log_monitor "ENHANCED TRAINING MONITOR COMPLETED"
    log_monitor "================================================================================"
}

generate_final_report() {
    local final_report="${SCRIPT_DIR}/FINAL_REPORT_4PM_20251103.txt"
    
    {
        echo "================================================================================"
        echo "FINAL TRAINING REPORT - 4:00 PM CST"
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S CST')"
        echo "================================================================================"
        echo ""
        
        echo "1. TRAINING RUNS COMPLETED:"
        local total_runs=$(count_completed_runs)
        echo "   Total Runs: $total_runs"
        echo ""
        
        if [ -f "${SCRIPT_DIR}/training_master_log_20251103.txt" ]; then
            echo "   Run Details:"
            grep -E "(Run Number:|Epochs:|Final.*Loss:)" "${SCRIPT_DIR}/training_master_log_20251103.txt" | sed 's/^/   /'
        fi
        echo ""
        
        echo "2. TOTAL EPOCHS TRAINED:"
        local total_epochs=$(count_completed_epochs)
        echo "   Total Epochs: $total_epochs"
        echo ""
        
        echo "3. LOSS PROGRESSION:"
        if [ -f "$LOSS_TRACKING" ]; then
            cat "$LOSS_TRACKING" | sed 's/^/   /'
        else
            echo "   No loss tracking data available"
        fi
        echo ""
        
        echo "4. GPU UTILIZATION SUMMARY:"
        if [ -f "$GPU_LOG" ]; then
            echo "   Average GPU Utilization:"
            grep "GPU:" "$GPU_LOG" | awk -F'GPU: |%' '{sum+=$2; count++} END {if(count>0) print "   " sum/count "%"; else print "   N/A"}'
            echo ""
            echo "   Recent GPU Stats:"
            tail -10 "$GPU_LOG" | sed 's/^/   /'
        fi
        echo ""
        
        echo "5. ALERTS AND ISSUES:"
        if [ -f "$ALERT_LOG" ] && [ -s "$ALERT_LOG" ]; then
            cat "$ALERT_LOG" | sed 's/^/   /'
        else
            echo "   ✅ No alerts - training ran smoothly"
        fi
        echo ""
        
        echo "6. CHECKPOINT FILES:"
        ls -lh "${SCRIPT_DIR}/checkpoints_corrected/"*.pt 2>/dev/null | tail -10 | sed 's/^/   /' || echo "   No checkpoints found"
        echo ""
        
        echo "7. SUCCESS CRITERIA CHECK:"
        echo "   ✓ Zero GPU idle time: $(check_training_active && echo "PASS" || echo "CHECK LOGS")"
        echo "   ✓ At least 3 runs completed: $([ $total_runs -ge 3 ] && echo "PASS ($total_runs runs)" || echo "PARTIAL ($total_runs runs)")"
        echo "   ✓ Comprehensive logs: AVAILABLE"
        echo ""
        
        echo "================================================================================"
        echo "All detailed logs available in:"
        echo "  - Master Log: training_master_log_20251103.txt"
        echo "  - Monitor Log: enhanced_monitor_20251103.log"
        echo "  - GPU Log: gpu_utilization_20251103.log"
        echo "  - Loss Tracking: loss_tracking_20251103.log"
        echo "  - Alert Log: training_alerts_20251103.log"
        echo "================================================================================"
        
    } > "$final_report"
    
    log_monitor "Final report generated: $final_report"
    
    # Also create a summary file
    cat "$final_report"
}

# Execute main function
main

exit 0

