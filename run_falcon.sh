#!/bin/bash
# ============================================================
# Causeway x Falcon-H1R-7B Experiment
# Trains Causeway causal adapter on Falcon's hybrid SSM+Attention
# hidden states across all 3 SCM domains.
#
# Usage:
#   chmod +x run_falcon.sh && ./run_falcon.sh
# ============================================================

set -e

FALCON="tiiuae/Falcon-H1R-7B"
LOG_DIR="/workspace/results/falcon"
WORKDIR="/workspace/causeway"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Causeway x Falcon-H1R-7B Experiment"
echo "  Domains: deployment, clinical, confounded"
echo "============================================"

cd "$WORKDIR"

# Domain 1: Deployment (software deployment SCM)
echo ""
echo "[1/3] Deployment domain..."
python train_on_transformer.py \
  --domain deployment \
  --model "$FALCON" \
  --d_causal 64 \
  --epochs 200 \
  --num_samples 50000 \
  --batch_encode_size 16 \
  --batch_size 128 \
  --save_path "$LOG_DIR/causeway_falcon_deployment.pt" \
  2>&1 | tee "$LOG_DIR/deployment.log"

echo ""
echo "[2/3] Clinical domain..."
python train_on_transformer.py \
  --domain clinical \
  --model "$FALCON" \
  --d_causal 64 \
  --epochs 200 \
  --num_samples 50000 \
  --batch_encode_size 16 \
  --batch_size 128 \
  --save_path "$LOG_DIR/causeway_falcon_clinical.pt" \
  2>&1 | tee "$LOG_DIR/clinical.log"

echo ""
echo "[3/3] Confounded domain..."
python train_on_transformer.py \
  --domain confounded \
  --model "$FALCON" \
  --d_causal 64 \
  --epochs 200 \
  --num_samples 50000 \
  --batch_encode_size 16 \
  --batch_size 128 \
  --save_path "$LOG_DIR/causeway_falcon_confounded.pt" \
  2>&1 | tee "$LOG_DIR/confounded.log"

echo ""
echo "============================================"
echo "  All domains done. Running comparison..."
echo "============================================"

python compare_results.py \
  --falcon_dir "$LOG_DIR" \
  --mistral_dir "$WORKDIR" \
  2>&1 | tee "$LOG_DIR/comparison.log"

echo ""
echo "Results in: $LOG_DIR"
