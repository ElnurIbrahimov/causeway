#!/bin/bash
# ============================================================
# Causeway Confounded Domain: Full Pipeline on Llama 3.1 8B
#
# Run on RunPod with:
#   nohup ./run_confounded.sh > confounded_out.log 2>&1 &
# ============================================================

set -e

MODEL="meta-llama/Llama-3.1-8B"
MODEL_SHORT="Llama-3.1-8B"
DOMAIN="confounded"
D_CAUSAL=64
NUM_SAMPLES=50000
EPOCHS=200
BRIDGE_EPOCHS=30

echo "============================================"
echo "  Causeway Confounded Pipeline"
echo "  Model: $MODEL"
echo "  Domain: $DOMAIN"
echo "============================================"
echo "Start time: $(date)"
echo ""

# Step 1: Encode confounded SCM → cache
echo "[1/4] Encoding confounded SCM through $MODEL_SHORT..."
echo "  This encodes $NUM_SAMPLES samples and caches to disk."
echo "  Expected time: ~20-30 min on H100"
echo ""

python train_on_transformer.py \
    --domain $DOMAIN \
    --model $MODEL \
    --d_causal $D_CAUSAL \
    --epochs $EPOCHS \
    --num_samples $NUM_SAMPLES

echo ""
echo "  Step 1 complete: Causeway trained on confounded domain."
echo "  Checkpoint: causeway_${DOMAIN}_${MODEL_SHORT}.pt"
echo ""

# Step 2: Train bridge (V2, pairwise ranking)
echo "[2/4] Training bridge (V2, pairwise ranking)..."
echo "  Expected time: ~30-60 min on H100"
echo ""

python train_bridge.py \
    --model $MODEL \
    --domain $DOMAIN \
    --bridge_version v2 \
    --epochs $BRIDGE_EPOCHS

echo ""
echo "  Step 2 complete: Bridge trained."
echo "  Checkpoint: bridge_v2_${DOMAIN}_${MODEL_SHORT}.pt"
echo ""

# Step 3: Full evaluation
echo "[3/4] Running closed-loop evaluation..."
echo ""

python eval_closed_loop.py \
    --domain $DOMAIN \
    --checkpoint causeway_${DOMAIN}_${MODEL_SHORT}.pt \
    --bridge_checkpoint bridge_v2_${DOMAIN}_${MODEL_SHORT}.pt \
    --bridge_version v2

echo ""
echo "  Step 3 complete."
echo ""

# Step 4: Also run existing domains for comparison
echo "[4/4] Running existing domains on $MODEL_SHORT for comparison..."
echo ""

for EXISTING_DOMAIN in deployment clinical; do
    echo "--- $EXISTING_DOMAIN domain ---"
    python train_on_transformer.py \
        --domain $EXISTING_DOMAIN \
        --model $MODEL \
        --d_causal $D_CAUSAL \
        --epochs $EPOCHS \
        --num_samples $NUM_SAMPLES

    echo "  $EXISTING_DOMAIN Causeway training complete."
done

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  End time: $(date)"
echo "============================================"
