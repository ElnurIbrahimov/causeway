"""
RunPod Bootstrap: Causeway x Falcon-H1R-7B Experiment.

Installs mamba-ssm CUDA kernels, clones causeway from GitHub,
downloads Falcon-H1R-7B, and launches training across all 3 SCM domains.

Run from RunPod web terminal or SSH:
    python runpod_bootstrap_falcon.py
"""

import subprocess
import os
import sys
import time

REPO = "https://github.com/ElnurIbrahimov/causeway.git"
WORKDIR = "/workspace/causeway"
RESULTS = "/workspace/results/falcon"
LOG = "/workspace/falcon_causeway.log"


def run(cmd, cwd=None, check=True):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd,
                            capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        print(f"ERROR: exit {result.returncode}")
        sys.exit(1)
    return result


print("=" * 60)
print("  Causeway x Falcon-H1R-7B Bootstrap")
print("=" * 60)

# 1. GPU check
print("\n[1] GPU...")
run("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader")

# 2. Install PyTorch + HuggingFace
print("\n[2] Installing base dependencies...")
run("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
run("pip install -q transformers accelerate sentencepiece protobuf "
    "networkx matplotlib tqdm numpy huggingface_hub")

# 3. Install Mamba2 CUDA kernels (required for Falcon-H1R-7B)
print("\n[3] Installing mamba-ssm and causal-conv1d...")
run("pip install causal-conv1d>=1.4.0")
run("pip install mamba-ssm>=2.2.2")

# Verify mamba-ssm installed
run("python -c \"import mamba_ssm; print('mamba-ssm OK:', mamba_ssm.__version__)\"")

# 4. Clone or update causeway
print(f"\n[4] Setting up causeway repo...")
if os.path.exists(WORKDIR):
    print("Repo exists, pulling latest...")
    run("git pull", cwd=WORKDIR)
else:
    run(f"git clone {REPO} {WORKDIR}")

os.makedirs(RESULTS, exist_ok=True)

# 5. Pre-download Falcon-H1R-7B
print("\n[5] Downloading Falcon-H1R-7B from HuggingFace...")
run("""python -c "
from transformers import AutoTokenizer, AutoModel
import torch
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('tiiuae/Falcon-H1R-7B', trust_remote_code=True)
print('Downloading model weights...')
AutoModel.from_pretrained('tiiuae/Falcon-H1R-7B', trust_remote_code=True, torch_dtype=torch.float16)
print('Falcon-H1R-7B ready.')
" """)

# 6. Verify causeway imports
print("\n[6] Verifying causeway module...")
run("""python -c "
import sys
sys.path.insert(0, '.')
from causeway.causeway_module import Causeway
# Falcon-H1R-7B d_model=4096
model = Causeway(d_model=4096, d_causal=64, d_action=4096)
n = sum(p.numel() for p in model.parameters())
print(f'Causeway OK — {n/1e6:.3f}M params on d_model=4096')
" """, cwd=WORKDIR)

# 7. Launch experiment (background)
print(f"\n[7] Launching experiment -> {LOG}")
run(
    f"nohup bash run_falcon.sh > {LOG} 2>&1 &",
    cwd=WORKDIR,
    check=False,
)

time.sleep(4)
result = run("pgrep -af train_on_transformer", check=False)
if result.returncode == 0:
    print(f"\nRunning: {result.stdout.strip()}")
    print(f"\nMonitor with:")
    print(f"  tail -f {LOG}")
    print(f"\nResults will be in: {RESULTS}")
else:
    print("\nWARNING: Process not found. Checking log...")
    run(f"tail -30 {LOG}", check=False)

print("\n" + "=" * 60)
print("  Bootstrap complete!")
print(f"  Log:     {LOG}")
print(f"  Results: {RESULTS}")
print("=" * 60)
