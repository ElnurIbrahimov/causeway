"""
RunPod Bootstrap Script.
Clones causeway repo, installs deps, and launches ablation suite in background.
Run with: runpod exec python runpod_bootstrap.py --pod_id <ID>
"""
import subprocess
import os
import sys

REPO = "https://github.com/ElnurIbrahimov/causeway.git"
WORKDIR = "/workspace/causeway"
LOG = "/workspace/ablation.log"

def run(cmd, cwd=None, check=True):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        print(f"ERROR: command failed (exit {result.returncode})")
        sys.exit(1)
    return result

print("=" * 60)
print("  Causeway RunPod Bootstrap")
print("=" * 60)

# 1. Check GPU
print("\n[1] Checking GPU...")
run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

# 2. Clone or update repo
if os.path.exists(WORKDIR):
    print(f"\n[2] Repo exists, pulling latest...")
    run("git pull", cwd=WORKDIR)
else:
    print(f"\n[2] Cloning {REPO}...")
    run(f"git clone {REPO} {WORKDIR}")

# 3. Install dependencies
print("\n[3] Installing dependencies...")
run(f"pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
run(f"pip install -q transformers accelerate sentencepiece protobuf networkx matplotlib tqdm numpy")

# 4. Verify imports
print("\n[4] Verifying causeway module...")
run(f"python -c \"import sys; sys.path.insert(0,'.'); from causeway.causeway_module import Causeway; print('Causeway OK')\"", cwd=WORKDIR)

# 5. Launch ablation suite with nohup
print(f"\n[5] Launching ablation suite (nohup -> {LOG})...")
run(
    f"nohup python -u train_ablations.py "
    f"--epochs 200 --num_samples 50000 --d_causal 48 --batch_size 128 "
    f"> {LOG} 2>&1 &",
    cwd=WORKDIR,
    check=False,
)

# 6. Verify it started
import time
time.sleep(3)
result = run("pgrep -f train_ablations.py", check=False)
if result.returncode == 0:
    print(f"\nSUCCESS! Training running with PID: {result.stdout.strip()}")
    print(f"Monitor with: tail -f {LOG}")
else:
    print("\nWARNING: Process not found. Checking log...")
    run(f"tail -20 {LOG}", check=False)

print("\n" + "=" * 60)
print("  Bootstrap complete!")
print(f"  Log: {LOG}")
print(f"  Results: {WORKDIR}/ablation_results.json")
print("=" * 60)
