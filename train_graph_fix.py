"""
Graph Fix Experiment.

The original Causeway graph collapses because:
  - edge_prior = -2.0 → edges start at σ(-2) = 12% probability
  - L0 + L1 + NOTEARS all push toward fewer edges
  - Graph converges to ~0.2 edges out of 10 true edges

Fix:
  - edge_prior = +2.0 → edges start at σ(+2) = 88% (dense → must earn removal)
  - lambda_sparse = 0, lambda_edge_count = 0 (no sparsity pressure)
  - Min connectivity loss: penalize if expected edges < floor (d_causal // 4)
  - Only NOTEARS enforces DAG structure

Runs 3 variants for comparison:
  A. original_causeway   — edge_prior=-2, sparsity on  (baseline)
  B. graph_fix           — edge_prior=+2, no sparsity, min connectivity
  C. graph_fix_no_floor  — edge_prior=+2, no sparsity, no floor (control)

Results appended to ablation_results.json.
Uses cached deployment dataset (no re-encoding needed).

Usage:
    python -u train_graph_fix.py
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causeway.causeway_module import Causeway
from causeway.causal_graph import CausalGraphLayer
from causeway.losses import CausewayLoss
from causeway.delta_predictor import DEFAULT_DELTA_DIMS

RESULTS_FILE = "ablation_results.json"
EPOCHS = 200
BATCH_SIZE = 128
LR = 3e-4
WARMUP = 10
D_CAUSAL = 48
NUM_SAMPLES = 50000
MODEL = "gpt2"


# ---------------------------------------------------------------------------
# Causeway with configurable edge_prior (monkey-patch the graph layer)
# ---------------------------------------------------------------------------

def make_causeway_with_prior(d_model, d_causal, d_action, edge_prior):
    """Build Causeway but override the graph's edge_prior."""
    causeway = Causeway(d_model=d_model, d_causal=d_causal, d_action=d_action)
    # Re-init edge logits with desired prior
    nn.init.constant_(causeway.causal_graph.edge_logits, edge_prior)
    return causeway


# ---------------------------------------------------------------------------
# Extended loss with minimum connectivity
# ---------------------------------------------------------------------------

class GraphFixLoss(CausewayLoss):
    """CausewayLoss + minimum connectivity floor."""

    def __init__(self, min_edges: float = 0.0, lambda_min_edges: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.min_edges = min_edges          # floor on expected edge count
        self.lambda_min_edges = lambda_min_edges

    def forward(self, delta_pred, delta_target, reg_losses, causal_graph=None):
        result = super().forward(delta_pred, delta_target, reg_losses)
        if causal_graph is not None and self.min_edges > 0:
            expected = causal_graph.adjacency_probs.sum()
            # Penalize if below floor: max(0, min_edges - expected)^2
            deficit = torch.relu(self.min_edges - expected)
            result["min_connectivity"] = deficit ** 2
            result["total"] = result["total"] + self.lambda_min_edges * deficit ** 2
        return result


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, criterion, loader, device):
    model.eval()
    total_losses = {}
    all_preds, all_targets = [], []
    n = 0

    for h, action, target in loader:
        h, action, target = h.to(device), action.to(device), target.to(device)
        delta = model(h, action)
        reg = model.get_regularization_losses()
        losses = criterion(delta, target, reg)
        all_preds.append(delta.values.cpu())
        all_targets.append(target.cpu())
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()
        n += 1

    avg = {k: v / n for k, v in total_losses.items()}
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    corrs, dir_accs, maes = {}, {}, {}
    for i, name in enumerate(DEFAULT_DELTA_DIMS):
        p, t = preds[:, i], targets[:, i]
        corrs[name] = torch.corrcoef(torch.stack([p, t]))[0, 1].item() if t.std() > 1e-6 and p.std() > 1e-6 else 0.0
        dir_accs[name] = ((p.sign() == t.sign()).float().mean()).item()
        maes[name] = (p - t).abs().mean().item()

    overall_corr = torch.corrcoef(torch.stack([preds.flatten(), targets.flatten()]))[0, 1].item()
    return {
        "losses": avg, "correlations": corrs, "directional_accuracy": dir_accs,
        "mae": maes, "overall_corr": overall_corr,
        "overall_mae": (preds - targets).abs().mean().item(),
    }


def run_variant(name, model, criterion, train_loader, val_loader, device,
                lambda_sparse=0.05, lambda_edge_count=0.05):
    SAVE_EVERY = 25
    best_path = f"graphfix_{name}_best.pt"
    periodic_path = f"graphfix_{name}_periodic.pt"

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*65}")
    print(f"  Variant: {name}  |  Params: {n_params/1e6:.3f}M")
    print(f"{'='*65}")

    # Override criterion lambdas
    criterion.lambda_sparse = lambda_sparse
    criterion.lambda_edge_count = lambda_edge_count

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=WARMUP)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP, eta_min=LR * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP])

    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(periodic_path):
        print(f"  Resuming from {periodic_path}...")
        ckpt = torch.load(periodic_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resuming from epoch {start_epoch}")

    # Print initial graph state
    gs = model.causal_graph.get_graph_stats()
    print(f"  Initial graph: hard_edges={gs['hard_edges']}  expected={gs['expected_edges']}  "
          f"max_prob={gs['max_edge_prob']:.4f}  density={gs['density']:.4f}")

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        progress = epoch / max(EPOCHS - 1, 1)
        temp = 1.0 + (0.05 - 1.0) * progress
        model.causal_graph.set_temperature(temp)
        ramp = min(1.0, epoch / (EPOCHS * 0.3))
        criterion.lambda_acyclic = ramp * 10.0

        model.train()
        train_loss = 0.0
        n_batches = 0
        for h, action, target in train_loader:
            h, action, target = h.to(device), action.to(device), target.to(device)
            delta = model(h, action)
            reg = model.get_regularization_losses()
            # Pass graph for min connectivity loss
            losses = criterion(delta, target, reg,
                               causal_graph=model.causal_graph if hasattr(criterion, 'min_edges') else None)
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += losses["total"].item()
            n_batches += 1

        scheduler.step()
        train_loss /= n_batches
        val_results = evaluate(model, criterion, val_loader, device)
        val_loss = val_results["losses"]["total"]

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            gs = model.causal_graph.get_graph_stats()
            elapsed = time.time() - t0
            eta = elapsed / max(epoch - start_epoch + 1, 1) * (EPOCHS - epoch - 1)
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"val={val_loss:.4f}  corr={val_results['overall_corr']:.4f}  "
                  f"edges={gs['hard_edges']}(hard) {gs['expected_edges']}(exp)  "
                  f"density={gs['density']:.3f}  eta={eta/60:.1f}m")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_loss, "val_results": val_results}, best_path)

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_val_loss}, periodic_path)

    # Final
    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final = evaluate(model, criterion, val_loader, device)
    gs = model.causal_graph.get_graph_stats()
    top_edges = model.causal_graph.get_top_edges(15)

    print(f"\n  RESULT [{name}]")
    print(f"  Corr: {final['overall_corr']:.4f}  MAE: {final['overall_mae']:.4f}")
    print(f"  Graph: {gs['hard_edges']} hard edges, {gs['expected_edges']} expected, density={gs['density']:.4f}")
    print(f"  Top edges (i->j, prob, weight): {top_edges[:10]}")
    print(f"  Per-dim: {', '.join(f'{k[:8]}={v:.4f}' for k,v in final['correlations'].items())}")

    if os.path.exists(periodic_path):
        os.remove(periodic_path)

    return {
        "name": name,
        "n_params": n_params,
        "best_epoch": ckpt["epoch"],
        "best_val_loss": ckpt["val_loss"],
        "overall_corr": final["overall_corr"],
        "overall_mae": final["overall_mae"],
        "correlations": final["correlations"],
        "directional_accuracy": final["directional_accuracy"],
        "mae": final["mae"],
        "graph_stats": gs,
        "top_edges": top_edges[:10],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    print(f"{'='*65}")
    print(f"  Graph Fix Experiment")
    print(f"  Device: {device}")
    print(f"{'='*65}")

    # Load cached deployment dataset
    cache_path = f"cache_deployment_gpt2_{NUM_SAMPLES}_v2.pt"
    print(f"\nLoading dataset from cache: {cache_path}")
    from environments.text_scm import TextSCMDataset
    dataset = TextSCMDataset(
        model_name=MODEL, num_samples=NUM_SAMPLES,
        device=str(device), cache_path=cache_path,
    )
    d_model = dataset.d_model
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    print(f"Dataset: {n_train} train, {n_val} val, d_model={d_model}")

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    def save():
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # ---- Variant A: Original (edge_prior=-2, sparsity on) ----
    name = "graphfix_original"
    if name not in results:
        model = make_causeway_with_prior(d_model, D_CAUSAL, d_model, edge_prior=-2.0)
        criterion = CausewayLoss()
        result = run_variant(name, model, criterion, train_loader, val_loader, device,
                             lambda_sparse=0.05, lambda_edge_count=0.05)
        results[name] = result
        save()
    else:
        print(f"\nSkipping {name} (already done)")

    # ---- Variant B: Fix (edge_prior=+2, no sparsity, min connectivity) ----
    name = "graphfix_positive_prior"
    if name not in results:
        model = make_causeway_with_prior(d_model, D_CAUSAL, d_model, edge_prior=+2.0)
        min_edges = D_CAUSAL // 4  # floor = 12 edges (true graph has 10)
        criterion = GraphFixLoss(
            min_edges=min_edges, lambda_min_edges=0.5,
            lambda_sparse=0.0, lambda_edge_count=0.0,
        )
        print(f"\n  Min connectivity floor: {min_edges} edges")
        result = run_variant(name, model, criterion, train_loader, val_loader, device,
                             lambda_sparse=0.0, lambda_edge_count=0.0)
        results[name] = result
        save()
    else:
        print(f"\nSkipping {name} (already done)")

    # ---- Variant C: edge_prior=+2, no sparsity, no floor (control) ----
    name = "graphfix_positive_prior_no_floor"
    if name not in results:
        model = make_causeway_with_prior(d_model, D_CAUSAL, d_model, edge_prior=+2.0)
        criterion = CausewayLoss(lambda_sparse=0.0, lambda_edge_count=0.0)
        result = run_variant(name, model, criterion, train_loader, val_loader, device,
                             lambda_sparse=0.0, lambda_edge_count=0.0)
        results[name] = result
        save()
    else:
        print(f"\nSkipping {name} (already done)")

    # ---- Summary ----
    print(f"\n{'='*65}")
    print(f"  GRAPH FIX SUMMARY")
    print(f"{'='*65}")
    ref_corr = results.get("ablation_mlp_deployment", {}).get("overall_corr", None)
    if ref_corr:
        print(f"  MLP baseline (reference): {ref_corr:.4f}")

    for n, r in results.items():
        if n.startswith("graphfix_"):
            gs = r.get("graph_stats", {})
            print(f"  {n:<40} corr={r['overall_corr']:.4f}  "
                  f"edges={gs.get('hard_edges','?')}(hard) {gs.get('expected_edges','?')}(exp)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
