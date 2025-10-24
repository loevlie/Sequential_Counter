#!/bin/bash
# =============================================================================
# Check Hyperparameter Search Status
# =============================================================================
# This script provides a quick overview of your hyperparameter search progress
# =============================================================================

echo "=============================================="
echo "VLM Counting Hyperparameter Search Status"
echo "=============================================="
echo ""

# Check for running/pending jobs
echo "SLURM Job Status:"
echo "-------------------"
squeue -u dloevl01 -o "%.8i %.9P %.20j %.8T %.10M %.6D %.4C" | grep -E "vlm_counting|sequential" || echo "No active jobs found"
echo ""

# Check completed jobs
if [ -d "slurm_logs/out" ]; then
    completed=$(ls slurm_logs/out/*.out 2>/dev/null | wc -l)
    failed=$(grep -l "error\|Error\|ERROR\|fail\|Fail" slurm_logs/err/*.err 2>/dev/null | wc -l)

    echo "Completed Experiments: $completed"
    echo "Failed Experiments: $failed"
    echo ""
fi

# Check saved models
if [ -d "hparam_results" ]; then
    num_models=$(ls -d hparam_results/*/ 2>/dev/null | wc -l)
    echo "Saved Model Checkpoints: $num_models"
    echo ""

    if [ $num_models -gt 0 ]; then
        echo "Recent checkpoints:"
        ls -lt hparam_results/ | head -6 | tail -5
        echo ""
    fi
fi

# Check for W&B sync status
echo "=============================================="
echo "W&B Status:"
echo "-------------------"
if command -v wandb &> /dev/null; then
    wandb status 2>/dev/null || echo "W&B not logged in. Run: wandb login"
else
    echo "W&B not installed. Run: pip install wandb"
fi
echo ""

# Show recent errors (if any)
if [ -d "slurm_logs/err" ]; then
    recent_errors=$(find slurm_logs/err -name "*.err" -mmin -60 -size +0 2>/dev/null)
    if [ ! -z "$recent_errors" ]; then
        echo "=============================================="
        echo "Recent Errors (last hour):"
        echo "-------------------"
        for err_file in $recent_errors; do
            echo "File: $err_file"
            tail -5 "$err_file"
            echo ""
        done
    fi
fi

# Disk usage check
echo "=============================================="
echo "Disk Usage:"
echo "-------------------"
if [ -d "hparam_results" ]; then
    du -sh hparam_results 2>/dev/null || echo "No results directory"
fi
if [ -d "slurm_logs" ]; then
    du -sh slurm_logs 2>/dev/null || echo "No logs directory"
fi
echo ""

echo "=============================================="
echo "Quick Commands:"
echo "-------------------"
echo "View specific log:       tail -f slurm_logs/out/<job_id>_<array_id>.out"
echo "Cancel all jobs:         scancel -u dloevl01"
echo "Analyze results:         python analyze_hparam_results.py"
echo "W&B dashboard:           https://wandb.ai/<username>/sequential-counting-hparam-search"
echo "=============================================="
