#!/bin/bash
# ============================================================
# Causeway RunPod Setup Script
# Run this once after connecting to your RunPod instance.
#
# Usage:
#   chmod +x runpod_setup.sh && ./runpod_setup.sh
# ============================================================

set -e

echo "============================================"
echo "  Causeway RunPod Setup"
echo "============================================"

# 1. Install dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip install --quiet torch numpy tqdm transformers accelerate sentencepiece protobuf

# 2. Check GPU
echo ""
echo "[2/4] Checking GPU..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# 3. Download model (pre-download so encoding doesn't stall)
echo ""
echo "[3/4] Pre-downloading Mistral 7B v0.3 (this may take a few minutes)..."
python -c "
from transformers import AutoModel, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.3', trust_remote_code=True)
print('Downloading model...')
AutoModel.from_pretrained('mistralai/Mistral-7B-v0.3', trust_remote_code=True, torch_dtype='auto')
print('Model cached successfully.')
"

# 4. Verify project files
echo ""
echo "[4/4] Verifying Causeway project..."
python -c "
import sys, os
sys.path.insert(0, '.')
from causeway.causeway_module import Causeway
print('Causeway module imports OK')
from environments.text_scm import TextSCMDataset
print('TextSCMDataset imports OK')
print('All checks passed!')
"

echo ""
echo "============================================"
echo "  Setup complete! Run training with:"
echo ""
echo "  python train_on_transformer.py \\"
echo "    --model mistralai/Mistral-7B-v0.3 \\"
echo "    --d_causal 64 \\"
echo "    --epochs 200 \\"
echo "    --num_samples 50000"
echo ""
echo "============================================"
