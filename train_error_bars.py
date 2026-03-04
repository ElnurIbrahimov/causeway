"""
Error Bar Experiment.

Runs 3 seeds each for the two key variants to get confidence intervals:
  1. MLP baseline (no causal structure, 0.922M params)
  2. No-engine Causeway (StateEncoder + concat MLP + DeltaPredictor, 0.382M)

If confidence intervals don't overlap → the causal pipeline advantage is real.
If they overlap → the 0.018 gap is noise.

Results appended to ablation_results.json.
Uses cached deployment dataset.

Usage:
    python -u train_error_bars.py
"""

import json, os, sys, time
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causeway.causeway_module import Causeway
from causeway.losses import CausewayLoss
from causeway.delta_predictor import DEFAULT_DELTA_DIMS
from causeway.state_encoder import StateEncoder
from train_ablations import MLPBaseline, CausewayNoEngine, SimpleDelta, evaluate

RESULTS_FILE = "ablation_results.json"
SEEDS = [42, 123, 777]
EPOCHS = 200
BATCH_SIZE = 128
LR = 3e-4
WARMUP = 10
D_CAUSAL = 48
NUM_SAMPLES = 50000
MODEL = "gpt2"


def run_seed(name, model, train_loader, val_loader, device,
             lambda_sparse=0.0, lambda_edge_count=0.0, seed=42):
    SAVE_EVERY = 25
    best_path = f"errbar_{name}_s{seed}_best.pt"
    periodic_path = f"errbar_{name}_s{seed}_periodic.pt"

    torch.manual_seed(seed)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = CausewayLoss(lambda_sparse=lambda_sparse, lambda_edge_count=lambda_edge_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=WARMUP)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP, eta_min=LR * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP])

    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(periodic_path):
        ckpt = torch.load(periodic_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  [seed={seed}] Resuming from epoch {start_epoch}")

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        if hasattr(model, "causal_graph"):
            progress = epoch / max(EPOCHS - 1, 1)
            model.causal_graph.set_temperature(1.0 + (0.05 - 1.0) * progress)
            criterion.lambda_acyclic = min(1.0, epoch / (EPOCHS * 0.3)) * 10.0
        else:
            criterion.lambda_acyclic = 0.0

        model.train()
        train_loss, n_batches = 0.0, 0
        for h, action, target in train_loader:
            h, action, target = h.to(device), action.to(device), target.to(device)
            delta = model(h, action)
            reg = model.get_regularization_losses()
            losses = criterion(delta, target, reg)
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += losses["total"].item()
            n_batches += 1

        scheduler.step()
        val_results = evaluate(model, criterion, val_loader, device)
        val_loss = val_results["losses"]["total"]

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            elapsed = time.time() - t0
            eta = elapsed / max(epoch - start_epoch + 1, 1) * (EPOCHS - epoch - 1)
            print(f"  [seed={seed}] Epoch {epoch:3d}/{EPOCHS}  "
                  f"val={val_loss:.4f}  corr={val_results['overall_corr']:.4f}  "
                  f"eta={eta/60:.1f}m")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_loss, "val_results": val_results}, best_path)

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_val_loss}, periodic_path)

    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final = evaluate(model, criterion, val_loader, device)

    if os.path.exists(periodic_path):
        os.remove(periodic_path)

    return {
        "seed": seed,
        "n_params": n_params,
        "best_epoch": ckpt["epoch"],
        "best_val_loss": float(ckpt["val_loss"]),
        "overall_corr": float(final["overall_corr"]),
        "overall_mae": float(final["overall_mae"]),
        "correlations": {k: float(v) for k, v in final["correlations"].items()},
        "directional_accuracy": {k: float(v) for k, v in final["directional_accuracy"].items()},
    }


def summarize(results_list, label):
    import statistics
    corrs = [r["overall_corr"] for r in results_list]
    mean = statistics.mean(corrs)
    std = statistics.stdev(corrs) if len(corrs) > 1 else 0.0
    ci = 1.96 * std  # 95% CI half-width (normal approx)
    print(f"  {label}: mean={mean:.4f} ± {std:.4f}  (95% CI: [{mean-ci:.4f}, {mean+ci:.4f}])")
    print(f"    Seeds: {[f'{c:.4f}' for c in corrs]}")
    return {"mean": mean, "std": std, "ci_95": ci, "seeds": corrs}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*65}")
    print(f"  Error Bar Experiment — 3 seeds × 2 variants")
    print(f"  Device: {device}")
    print(f"{'='*65}\n")

    # Load cached dataset
    from environments.text_scm import TextSCMDataset
    cache_path = f"cache_deployment_{MODEL}_{NUM_SAMPLES}_v2.pt"
    dataset = TextSCMDataset(model_name=MODEL, num_samples=NUM_SAMPLES,
                              device=str(device), cache_path=cache_path)
    d_model = dataset.d_model
    print(f"Dataset: {len(dataset)} samples, d_model={d_model}\n")

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    def save():
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # ---- MLP baseline: 3 seeds ----
    print("=" * 65)
    print("  VARIANT 1: MLP Baseline (3 seeds)")
    print("=" * 65)
    mlp_results = results.get("errbar_mlp_seeds", [])
    completed_seeds = {r["seed"] for r in mlp_results}

    for seed in SEEDS:
        if seed in completed_seeds:
            print(f"  [seed={seed}] Already done, skipping.")
            continue
        # Fresh split per seed for variance
        g = torch.Generator().manual_seed(seed)
        n_train = int(0.8 * len(dataset))
        n_val = len(dataset) - n_train
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=g)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                   generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

        print(f"\n  [seed={seed}] Training MLP baseline...")
        model = MLPBaseline(d_model=d_model, d_action=d_model, hidden_dim=512)
        r = run_seed("mlp", model, train_loader, val_loader, device, seed=seed)
        mlp_results.append(r)
        results["errbar_mlp_seeds"] = mlp_results
        save()
        print(f"  [seed={seed}] corr={r['overall_corr']:.4f}")

    # ---- No-engine Causeway: 3 seeds ----
    print("\n" + "=" * 65)
    print("  VARIANT 2: No-Engine Causeway (StateEncoder + concat + DeltaPredictor, 3 seeds)")
    print("=" * 65)
    noeng_results = results.get("errbar_no_engine_seeds", [])
    completed_seeds = {r["seed"] for r in noeng_results}

    for seed in SEEDS:
        if seed in completed_seeds:
            print(f"  [seed={seed}] Already done, skipping.")
            continue
        g = torch.Generator().manual_seed(seed)
        n_train = int(0.8 * len(dataset))
        n_val = len(dataset) - n_train
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=g)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                   generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

        print(f"\n  [seed={seed}] Training no-engine Causeway...")
        model = CausewayNoEngine(d_model=d_model, d_causal=D_CAUSAL, d_action=d_model)
        r = run_seed("noengine", model, train_loader, val_loader, device, seed=seed)
        noeng_results.append(r)
        results["errbar_no_engine_seeds"] = noeng_results
        save()
        print(f"  [seed={seed}] corr={r['overall_corr']:.4f}")

    # ---- Summary ----
    print(f"\n{'='*65}")
    print(f"  ERROR BAR SUMMARY")
    print(f"{'='*65}")
    mlp_summary = summarize(results["errbar_mlp_seeds"], "MLP baseline     ")
    noeng_summary = summarize(results["errbar_no_engine_seeds"], "No-engine Causeway")

    gap = noeng_summary["mean"] - mlp_summary["mean"]
    mlp_ci_high = mlp_summary["mean"] + mlp_summary["ci_95"]
    noeng_ci_low = noeng_summary["mean"] - noeng_summary["ci_95"]
    overlap = mlp_ci_high > noeng_ci_low

    print(f"\n  Gap: {gap:+.4f}")
    print(f"  95% CIs overlap: {overlap}")
    if not overlap:
        print(f"  --> RESULT IS REAL. CIs do not overlap. The causal pipeline advantage is statistically significant.")
    else:
        print(f"  --> CIs overlap. Gap may be noise. Interpret with caution.")

    results["errbar_summary"] = {
        "mlp": mlp_summary, "no_engine": noeng_summary,
        "gap": gap, "ci_overlap": overlap,
    }
    save()
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
