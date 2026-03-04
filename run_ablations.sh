#!/bin/bash
# ============================================================
# Causeway Ablation Suite — RunPod Runner
#
# Features:
#   - nohup: training survives terminal disconnects
#   - Periodic checkpoints every 25 epochs (crash recovery)
#   - Results saved to ablation_results.json after each experiment
#   - Completed experiments skipped on restart
#
# Usage:
#   chmod +x run_ablations.sh
#   ./run_ablations.sh               # run in background, survives disconnect
#
# Monitor progress:
#   tail -f ablation.log
#   tail -20 ablation.log
#
# Resume after crash/disconnect (just re-run):
#   ./run_ablations.sh
#
# Check results so far:
#   cat ablation_results.json | python3 -m json.tool
# ============================================================

set -e

LOG="ablation.log"
RESULTS="ablation_results.json"

echo "============================================"
echo "  Causeway Ablation Suite"
echo "============================================"
echo "  Log:     $LOG"
echo "  Results: $RESULTS"
echo ""

# Check if already running
if pgrep -f "train_ablations.py" > /dev/null 2>&1; then
    echo "WARNING: train_ablations.py is already running!"
    echo "  PIDs: $(pgrep -f train_ablations.py)"
    echo "  Monitor with: tail -f $LOG"
    exit 1
fi

# Show current results if any
if [ -f "$RESULTS" ]; then
    echo "Existing results found in $RESULTS:"
    python3 -c "
import json
with open('$RESULTS') as f:
    r = json.load(f)
print(f'  Completed: {list(r.keys())}')
print(f'  Remaining: {[x for x in [\"ablation_mlp_deployment\",\"ablation_no_graph\",\"ablation_no_engine\",\"ablation_no_sparsity\",\"ablation_confounded\",\"ablation_mlp_confounded\"] if x not in r]}')
" 2>/dev/null || true
    echo ""
fi

echo "Starting ablation suite in background..."
echo "Logs -> $LOG"
echo ""

# Launch with nohup so it survives terminal disconnect
# -u: unbuffered output so logs appear in real-time
nohup python -u train_ablations.py \
    --epochs 200 \
    --num_samples 50000 \
    --d_causal 48 \
    --batch_size 128 \
    --lr 3e-4 \
    --warmup_epochs 10 \
    > "$LOG" 2>&1 &

PID=$!
echo "Launched with PID: $PID"
echo ""
echo "Commands:"
echo "  Monitor:  tail -f $LOG"
echo "  Status:   ps aux | grep train_ablations"
echo "  Results:  cat $RESULTS | python3 -m json.tool"
echo "  Resume:   (just re-run ./run_ablations.sh if it crashes)"
echo ""
echo "The training will continue even if you close this terminal."
echo "============================================"

# Show first few lines of log
sleep 3
echo ""
echo "--- First output ---"
tail -20 "$LOG" 2>/dev/null || true
