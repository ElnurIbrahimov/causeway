"""
Path A: Direct Graph Supervision Experiment.

The SCM has 8 variables and 10 known ground-truth edges.
Causeway normally uses d_causal=48, so the graph (48x48) can't be directly
supervised against the 8x8 SCM adjacency.

This experiment runs with d_causal=8 to match the SCM exactly, then adds
a BCE supervision loss: L_graph = BCE(edge_probs, true_adjacency_binary).

3 variants:
  A. d_causal=8, no supervision  (baseline at same size)
  B. d_causal=8, with BCE graph supervision
  C. d_causal=48, no supervision (original size, for reference)

Key question: Even with oracle graph supervision (correct topology forced),
does the DeltaPredictor use the graph signal to improve prediction?

If B > A: supervision helps, and the graph CAN contribute with the right topology
If B == A: residual bypass confirmed — graph topology is irrelevant regardless

Results appended to ablation_results.json.
Uses cached deployment dataset (no re-encoding needed).

Usage:
    python -u train_graph_supervised.py
"""

import json, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causeway.causeway_module import Causeway
from causeway.losses import CausewayLoss
from causeway.delta_predictor import DEFAULT_DELTA_DIMS
from train_ablations import evaluate

RESULTS_FILE = "ablation_results.json"
EPOCHS = 200
BATCH_SIZE = 128
LR = 3e-4
WARMUP = 10
NUM_SAMPLES = 50000
MODEL = "gpt2"
SEED = 42
SAVE_EVERY = 25


def get_true_adjacency_binary():
    """Return binary 8x8 ground-truth adjacency from the deployment SCM."""
    from environments.synthetic_scm import SyntheticSCM
    scm = SyntheticSCM()
    adj = scm.get_adjacency_tensor()
    return (adj != 0).float()  # binary: 1 where edge exists


class SupervisedCausewayLoss(CausewayLoss):
    """CausewayLoss + BCE supervision on graph edge probabilities."""

    def __init__(self, true_adjacency: torch.Tensor, lambda_graph_sup: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer = None  # not a Module, store directly
        self._true_adj = true_adjacency
        self.lambda_graph_sup = lambda_graph_sup

    def forward(self, delta_pred, delta_target, reg_losses, causal_graph=None):
        result = super().forward(delta_pred, delta_target, reg_losses)
        if causal_graph is not None and self.lambda_graph_sup > 0:
            probs = causal_graph.adjacency_probs  # (d_causal, d_causal)
            true_adj = self._true_adj.to(probs.device)
            # BCE between learned edge probs and binary ground truth
            l_sup = F.binary_cross_entropy(probs, true_adj)
            result["graph_supervision"] = l_sup
            result["total"] = result["total"] + self.lambda_graph_sup * l_sup
        return result


def run_variant(name, model, criterion, train_loader, val_loader, device,
                use_graph_supervision=False):
    best_path = f"graphsup_{name}_best.pt"
    periodic_path = f"graphsup_{name}_periodic.pt"

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    d_causal = model.d_causal

    print(f"\n{'='*65}")
    print(f"  Variant: {name}  |  d_causal={d_causal}  |  Params: {n_params/1e6:.3f}M")
    print(f"  Graph supervision: {use_graph_supervision}")
    print(f"{'='*65}")

    gs0 = model.causal_graph.get_graph_stats()
    print(f"  Initial graph: hard_edges={gs0['hard_edges']}  "
          f"expected={gs0['expected_edges']}  max_prob={gs0['max_edge_prob']:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=WARMUP)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP, eta_min=LR * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP])

    start_epoch, best_val_loss = 0, float("inf")
    if os.path.exists(periodic_path):
        ckpt = torch.load(periodic_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resuming from epoch {start_epoch}")

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        progress = epoch / max(EPOCHS - 1, 1)
        model.causal_graph.set_temperature(1.0 + (0.05 - 1.0) * progress)
        criterion.lambda_acyclic = min(1.0, epoch / (EPOCHS * 0.3)) * 10.0

        model.train()
        train_loss, n_batches = 0.0, 0
        for h, action, target in train_loader:
            h, action, target = h.to(device), action.to(device), target.to(device)
            delta = model(h, action)
            reg = model.get_regularization_losses()
            if use_graph_supervision:
                losses = criterion(delta, target, reg, causal_graph=model.causal_graph)
            else:
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

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            gs = model.causal_graph.get_graph_stats()
            elapsed = time.time() - t0
            eta = elapsed / max(epoch - start_epoch + 1, 1) * (EPOCHS - epoch - 1)
            print(f"Epoch {epoch:3d}/{EPOCHS}  val={val_loss:.4f}  "
                  f"corr={val_results['overall_corr']:.4f}  "
                  f"edges={gs['hard_edges']}(hard) {gs['expected_edges']}(exp)  "
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
    gs_final = model.causal_graph.get_graph_stats()
    top_edges = model.causal_graph.get_top_edges(15)

    # Compute edge recovery (how many of the 10 true edges did we find?)
    true_adj_binary = get_true_adjacency_binary().to(device)
    with torch.no_grad():
        learned_probs = model.causal_graph.adjacency_probs
        # For d_causal=8, compare directly; for larger, only compare 8x8 submatrix
        d = min(model.d_causal, 8)
        tp = ((learned_probs[:d, :d] > 0.5) & (true_adj_binary > 0.5)).sum().item()
        fp = ((learned_probs[:d, :d] > 0.5) & (true_adj_binary < 0.5)).sum().item()
        fn = ((learned_probs[:d, :d] < 0.5) & (true_adj_binary > 0.5)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\n  RESULT [{name}]")
    print(f"  Corr: {final['overall_corr']:.4f}  MAE: {final['overall_mae']:.4f}")
    print(f"  Graph: {gs_final['hard_edges']} hard edges, density={gs_final['density']:.4f}")
    print(f"  Edge recovery (vs 10 true): TP={tp}  FP={fp}  FN={fn}  "
          f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")
    print(f"  Top edges: {top_edges[:10]}")

    if os.path.exists(periodic_path):
        os.remove(periodic_path)

    return {
        "name": name,
        "n_params": n_params,
        "d_causal": model.d_causal,
        "graph_supervised": use_graph_supervision,
        "best_epoch": ckpt["epoch"],
        "best_val_loss": float(ckpt["val_loss"]),
        "overall_corr": float(final["overall_corr"]),
        "overall_mae": float(final["overall_mae"]),
        "correlations": {k: float(v) for k, v in final["correlations"].items()},
        "directional_accuracy": {k: float(v) for k, v in final["directional_accuracy"].items()},
        "graph_stats": gs_final,
        "edge_recovery": {"tp": tp, "fp": fp, "fn": fn,
                          "precision": float(precision), "recall": float(recall), "f1": float(f1)},
    }


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*65}")
    print(f"  Path A: Graph Supervision Experiment")
    print(f"  Device: {device}")
    print(f"{'='*65}\n")

    true_adj = get_true_adjacency_binary()
    print(f"Ground truth: {int(true_adj.sum())} edges in 8x8 adjacency")
    print(f"True edges: {true_adj.nonzero().tolist()}\n")

    # Load dataset
    from environments.text_scm import TextSCMDataset
    cache_path = f"cache_deployment_{MODEL}_{NUM_SAMPLES}_v2.pt"
    dataset = TextSCMDataset(model_name=MODEL, num_samples=NUM_SAMPLES,
                              device=str(device), cache_path=cache_path)
    d_model = dataset.d_model

    g = torch.Generator().manual_seed(SEED)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    print(f"Dataset: {n_train} train, {n_val} val, d_model={d_model}\n")

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    def save():
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # ---- Variant A: d_causal=8, no supervision ----
    name = "graphsup_d8_unsupervised"
    if name not in results:
        model = Causeway(d_model=d_model, d_causal=8, d_action=d_model)
        criterion = CausewayLoss(lambda_sparse=0.0, lambda_edge_count=0.0)
        r = run_variant(name, model, criterion, train_loader, val_loader, device,
                        use_graph_supervision=False)
        results[name] = r
        save()
    else:
        print(f"Skipping {name} (already done)")

    # ---- Variant B: d_causal=8, with BCE supervision ----
    name = "graphsup_d8_supervised"
    if name not in results:
        model = Causeway(d_model=d_model, d_causal=8, d_action=d_model)
        criterion = SupervisedCausewayLoss(
            true_adjacency=true_adj,
            lambda_graph_sup=2.0,   # strong supervision signal
            lambda_sparse=0.0,
            lambda_edge_count=0.0,
        )
        r = run_variant(name, model, criterion, train_loader, val_loader, device,
                        use_graph_supervision=True)
        results[name] = r
        save()
    else:
        print(f"Skipping {name} (already done)")

    # ---- Variant C: d_causal=48, no supervision (reference) ----
    name = "graphsup_d48_unsupervised"
    if name not in results:
        model = Causeway(d_model=d_model, d_causal=48, d_action=d_model)
        criterion = CausewayLoss(lambda_sparse=0.0, lambda_edge_count=0.0)
        r = run_variant(name, model, criterion, train_loader, val_loader, device,
                        use_graph_supervision=False)
        results[name] = r
        save()
    else:
        print(f"Skipping {name} (already done)")

    # ---- Summary ----
    print(f"\n{'='*65}")
    print(f"  GRAPH SUPERVISION SUMMARY")
    print(f"{'='*65}")
    ref_mlp = results.get("errbar_summary", {}).get("mlp", {}).get("mean")
    ref_noeng = results.get("errbar_summary", {}).get("no_engine", {}).get("mean")
    if ref_mlp:
        print(f"  MLP baseline (ref):         {ref_mlp:.4f}")
    if ref_noeng:
        print(f"  No-engine Causeway (ref):   {ref_noeng:.4f}")
    print()
    for n in ["graphsup_d8_unsupervised", "graphsup_d8_supervised", "graphsup_d48_unsupervised"]:
        if n in results:
            r = results[n]
            er = r.get("edge_recovery", {})
            print(f"  {n:<35} corr={r['overall_corr']:.4f}  "
                  f"d_causal={r['d_causal']}  "
                  f"edge_F1={er.get('f1', 0):.3f}  "
                  f"supervised={r['graph_supervised']}")
    print(f"\n  KEY QUESTION:")
    if "graphsup_d8_supervised" in results and "graphsup_d8_unsupervised" in results:
        sup = results["graphsup_d8_supervised"]["overall_corr"]
        unsup = results["graphsup_d8_unsupervised"]["overall_corr"]
        gap = sup - unsup
        print(f"  Supervised - Unsupervised (d_causal=8): {gap:+.4f}")
        if gap > 0.005:
            print(f"  --> Graph topology MATTERS when supervised. Residual bypass is learnable.")
        elif gap > -0.005:
            print(f"  --> Graph topology does NOT matter even with oracle supervision.")
            print(f"  --> Residual bypass is architectural, not a training issue.")
        else:
            print(f"  --> Supervision HURTS (BCE loss trades off against prediction quality).")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
