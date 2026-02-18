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
pip install --quiet torch numpy tqdm transformers accelerate sentencepiece protobuf networkx matplotlib

# 2. Check GPU
echo ""
echo "[2/4] Checking GPU..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'GPU: {name}')
    print(f'VRAM: {vram:.1f} GB')
    if vram < 20:
        print('WARNING: <20 GB VRAM — Mistral 7B encoding may OOM.')
        print('Consider using a 24+ GB GPU (RTX 4090, A10, A100).')
    else:
        print('VRAM OK for Mistral 7B fp16 encoding.')
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
print('  Causeway module .............. OK')
from environments.text_scm import TextSCMDataset
print('  TextSCMDataset (deployment) .. OK')
from environments.text_clinical import TextClinicalDataset
print('  TextClinicalDataset (clinical) OK')

# Quick param count for Mistral config
model = Causeway(d_model=4096, d_causal=64, d_action=4096)
n = sum(p.numel() for p in model.parameters())
print(f'  Causeway params (Mistral): {n/1e6:.2f}M')
print('All checks passed!')
"

echo ""
echo "============================================"
echo "  Setup complete! Start training with:"
echo ""
echo "    chmod +x run_mistral.sh && ./run_mistral.sh"
echo ""
echo "  Or run domains individually:"
echo ""
echo "    # Deployment only"
echo "    python train_on_transformer.py \\"
echo "      --domain deployment \\"
echo "      --model mistralai/Mistral-7B-v0.3 \\"
echo "      --d_causal 64 --epochs 200 --num_samples 50000"
echo ""
echo "    # Clinical only"
echo "    python train_on_transformer.py \\"
echo "      --domain clinical \\"
echo "      --model mistralai/Mistral-7B-v0.3 \\"
echo "      --d_causal 64 --epochs 200 --num_samples 50000"
echo ""
echo "============================================"
