#!/bin/bash
# ============================================================
# Train Causeway on Mistral 7B v0.3
#
# Expected timeline:
#   - Encoding 50K samples: ~15-30 min (7B model in fp16)
#   - Training 200 epochs:  ~10-20 min (cached data, no model)
#   - Total:                ~30-50 min
#
# Expected VRAM:
#   - During encoding: ~16-18 GB (Mistral 7B fp16 + batch overhead)
#   - During training:  ~1 GB (Causeway only, model freed)
#
# Usage:
#   chmod +x run_mistral.sh && ./run_mistral.sh
# ============================================================

set -e

echo "============================================"
echo "  Training Causeway on Mistral 7B v0.3"
echo "============================================"
echo ""

# d_causal=64: Mistral's 4096-dim hidden states encode richer
# structure than GPT-2's 768-dim, so more causal variables help.
python train_on_transformer.py \
    --model mistralai/Mistral-7B-v0.3 \
    --d_causal 64 \
    --epochs 200 \
    --num_samples 50000 \
    --lr 3e-4 \
    --warmup_epochs 10 \
    --batch_size 128

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Download causeway_Mistral-7B-v0.3.pt"
echo "============================================"
