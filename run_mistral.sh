#!/bin/bash
# ============================================================
# Train Causeway on Mistral 7B v0.3 — Both Domains
#
# Trains deployment domain first, then clinical domain.
# Each domain encodes 50K samples through frozen Mistral 7B,
# then trains Causeway on the cached hidden states.
#
# Expected timeline (per domain):
#   - Encoding 50K samples: ~15-30 min (7B model in fp16)
#   - Training 200 epochs:  ~10-20 min (cached data, no model)
#   - Total per domain:     ~30-50 min
#   - Total both domains:   ~60-90 min (second domain reuses model cache)
#
# Expected VRAM:
#   - During encoding: ~16-18 GB (Mistral 7B fp16 + batch overhead)
#   - During training:  ~1 GB (Causeway only, Mistral freed after encoding)
#
# GPU requirement: 24 GB minimum (RTX 4090, A10, A100)
#
# Usage:
#   chmod +x run_mistral.sh && ./run_mistral.sh
# ============================================================

set -e

echo "============================================"
echo "  Causeway x Mistral 7B v0.3"
echo "============================================"
echo ""

# ---- Domain 1: Software Deployment ----
echo "============================================"
echo "  [1/2] Training: Deployment Domain"
echo "============================================"
echo ""

# d_causal=64: Mistral's 4096-dim hidden states encode richer
# structure than GPT-2's 768-dim, so more causal variables help.
python train_on_transformer.py \
    --domain deployment \
    --model mistralai/Mistral-7B-v0.3 \
    --d_causal 64 \
    --epochs 200 \
    --num_samples 50000 \
    --lr 3e-4 \
    --warmup_epochs 10 \
    --batch_size 128

echo ""
echo "  Deployment done! Saved: causeway_Mistral-7B-v0.3.pt"
echo ""

# ---- Domain 2: Clinical Treatment ----
echo "============================================"
echo "  [2/2] Training: Clinical Domain"
echo "============================================"
echo ""

python train_on_transformer.py \
    --domain clinical \
    --model mistralai/Mistral-7B-v0.3 \
    --d_causal 64 \
    --epochs 200 \
    --num_samples 50000 \
    --lr 3e-4 \
    --warmup_epochs 10 \
    --batch_size 128

echo ""
echo "============================================"
echo "  All training complete!"
echo ""
echo "  Checkpoints:"
echo "    causeway_Mistral-7B-v0.3.pt           (deployment)"
echo "    causeway_clinical_Mistral-7B-v0.3.pt   (clinical)"
echo ""
echo "  Cached datasets (reusable):"
echo "    cache_deployment_Mistral-7B-v0.3_50000_v2.pt"
echo "    cache_clinical_Mistral-7B-v0.3_50000_v2.pt"
echo "============================================"
