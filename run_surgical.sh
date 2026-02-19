#!/bin/bash
# RunPod pipeline: Oracle -> Distillation -> End-to-End -> Evaluation
#
# Usage:
#   bash run_surgical.sh                    # Full pipeline with GPT-2
#   bash run_surgical.sh --model gpt2       # Explicit GPT-2
#   bash run_surgical.sh --model meta-llama/Llama-3.1-8B  # Llama 8B
#
# Prerequisites:
#   pip install -r requirements.txt
#   # For Llama: huggingface-cli login

set -euo pipefail

MODEL="${1:-gpt2}"
DOMAIN="${2:-confounded}"
D_CAUSAL="${3:-48}"
EPOCHS_ORACLE="${4:-200}"

MODEL_SHORT=$(echo "$MODEL" | rev | cut -d'/' -f1 | rev)

echo "============================================================"
echo "  SURGICAL CAUSEWAY PIPELINE"
echo "  Model: $MODEL  Domain: $DOMAIN  d_causal: $D_CAUSAL"
echo "============================================================"

# Determine d_causal and layer_idx based on model
if [[ "$MODEL_SHORT" == *"8B"* ]] || [[ "$MODEL_SHORT" == *"8b"* ]]; then
    D_CAUSAL=64
    LAYER_IDX=16
elif [[ "$MODEL_SHORT" == *"7B"* ]] || [[ "$MODEL_SHORT" == *"7b"* ]]; then
    D_CAUSAL=64
    LAYER_IDX=16
else
    # GPT-2 or small models
    D_CAUSAL=48
    LAYER_IDX=6
fi

echo "d_causal=$D_CAUSAL  layer_idx=$LAYER_IDX"

# ---- Phase 1: Oracle pre-training ----
ORACLE_CKPT="causeway_${DOMAIN}_${MODEL_SHORT}.pt"
if [ "$DOMAIN" = "deployment" ]; then
    ORACLE_CKPT="causeway_${MODEL_SHORT}.pt"
fi

if [ ! -f "$ORACLE_CKPT" ]; then
    echo ""
    echo "---- Phase 1: Oracle Pre-training ----"
    python train_on_transformer.py \
        --domain "$DOMAIN" \
        --model "$MODEL" \
        --d_causal "$D_CAUSAL" \
        --epochs "$EPOCHS_ORACLE" \
        --num_samples 50000
else
    echo ""
    echo "---- Phase 1: Skipping (checkpoint exists: $ORACLE_CKPT) ----"
fi

# ---- Phase 2: Distillation ----
DISTILL_CKPT="surgical_distill_${DOMAIN}_${MODEL_SHORT}.pt"
if [ ! -f "$DISTILL_CKPT" ]; then
    echo ""
    echo "---- Phase 2: Distillation Adaptation ----"
    python train_surgical.py \
        --phase 2 \
        --model "$MODEL" \
        --domain "$DOMAIN" \
        --causeway_checkpoint "$ORACLE_CKPT" \
        --layer_idx "$LAYER_IDX" \
        --p2_epochs 100 \
        --p2_freeze_epochs 30 \
        --batch_size 64
else
    echo ""
    echo "---- Phase 2: Skipping (checkpoint exists: $DISTILL_CKPT) ----"
fi

# ---- Phase 3: End-to-End ----
E2E_CKPT="surgical_e2e_${DOMAIN}_${MODEL_SHORT}.pt"
if [ ! -f "$E2E_CKPT" ]; then
    echo ""
    echo "---- Phase 3: End-to-End Pairwise Ranking ----"
    python train_surgical.py \
        --phase 3 \
        --model "$MODEL" \
        --domain "$DOMAIN" \
        --surgical_checkpoint "$DISTILL_CKPT" \
        --layer_idx "$LAYER_IDX" \
        --p3_epochs 40 \
        --batch_size 32
else
    echo ""
    echo "---- Phase 3: Skipping (checkpoint exists: $E2E_CKPT) ----"
fi

# ---- Evaluation ----
echo ""
echo "---- Evaluation ----"

# Try bridge checkpoint if it exists
BRIDGE_ARGS=""
BRIDGE_CKPT="bridge_v2_${DOMAIN}_${MODEL_SHORT}.pt"
if [ -f "$BRIDGE_CKPT" ]; then
    BRIDGE_ARGS="--bridge_checkpoint $BRIDGE_CKPT --bridge_version v2"
fi

python eval_closed_loop.py \
    --checkpoint "$ORACLE_CKPT" \
    --domain "$DOMAIN" \
    --surgical_checkpoint "$E2E_CKPT" \
    --n_scenarios 200 \
    $BRIDGE_ARGS

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================"
