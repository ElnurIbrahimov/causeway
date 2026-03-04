"""
Causeway Ablation Suite.

Runs 6 experiments on GPT-2 to determine whether the causal machinery
contributes beyond a flat MLP baseline.

Experiments (run in order, skipped on restart if already completed):
    1. ablation_mlp_deployment     - Parameter-matched MLP, no causal structure
    2. ablation_no_graph           - Causeway without CausalGraphLayer (identity pass-through)
    3. ablation_no_engine          - Causeway without InterventionEngine (concat MLP instead)
    4. ablation_no_sparsity        - Causeway with lambda_sparse=lambda_edge_count=0
    5. ablation_confounded         - Full Causeway on confounded domain (GPT-2)
    6. ablation_mlp_confounded     - MLP baseline on confounded domain

Results saved to ablation_results.json after each experiment.
Completed experiments are skipped on restart (resume-safe).

Usage:
    python -u train_ablations.py
    python -u train_ablations.py --epochs 200 --num_samples 50000
    nohup python -u train_ablations.py --epochs 200 --num_samples 50000 > ablation.log 2>&1 &
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causeway.causeway_module import Causeway
from causeway.losses import CausewayLoss
from causeway.delta_predictor import DeltaVector, DEFAULT_DELTA_DIMS
from causeway.state_encoder import StateEncoder


# ---------------------------------------------------------------------------
# Shared result container (DeltaVector-compatible for MLP variants)
# ---------------------------------------------------------------------------

@dataclass
class SimpleDelta:
    """Minimal DeltaVector-compatible output for non-Causeway models."""
    values: torch.Tensor
    confidence: torch.Tensor
    dim_names: list = None

    def __post_init__(self):
        if self.dim_names is None:
            self.dim_names = DEFAULT_DELTA_DIMS


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

class MLPBaseline(nn.Module):
    """
    Parameter-matched flat MLP. No causal structure.
    Input: concat(h, action) -> 5-dim structured delta.
    Hidden dim chosen to approximately match Causeway's 0.794M params on GPT-2.
    """

    def __init__(self, d_model: int, d_action: int, hidden_dim: int = 512):
        super().__init__()
        in_dim = d_model + d_action

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.value_head = nn.Linear(hidden_dim // 2, 5)
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 5),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> SimpleDelta:
        x = self.backbone(torch.cat([h, action], dim=-1))
        return SimpleDelta(
            values=self.value_head(x),
            confidence=self.conf_head(x),
        )

    def get_regularization_losses(self) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        z = torch.tensor(0.0, device=device)
        return {"acyclicity": z, "sparsity": z, "edge_count": z, "orthogonality": z}


class CausewayNoGraph(Causeway):
    """
    Causeway with CausalGraphLayer replaced by identity.
    Tests whether the learned causal graph contributes anything.
    z_refined = z (raw state encoder output, no graph message-passing).
    Adjacency is all-zeros so the intervention engine gets no graph signal.
    """

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> DeltaVector:
        z = self.state_encoder(h)
        # Skip graph: use raw z as z_refined
        z_refined = z
        # Use zero adjacency: no causal propagation signal
        adjacency = torch.zeros(
            self.d_causal, self.d_causal, device=h.device
        )
        z_counterfactual, _, _ = self.intervention_engine(
            z_refined, action, adjacency,
            num_propagation_steps=self.propagation_steps,
        )
        return self.delta_predictor(z_refined, z_counterfactual)

    def get_regularization_losses(self) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        z = torch.tensor(0.0, device=device)
        # Still apply orthogonality to StateEncoder
        return {
            "acyclicity": z,
            "sparsity": z,
            "edge_count": z,
            "orthogonality": self.state_encoder.orthogonality_loss(),
        }


class CausewayNoEngine(nn.Module):
    """
    Causeway with InterventionEngine replaced by a concat MLP.
    Tests whether Pearl's do-operator adds anything over simple action fusion.
    z_post = MLP(concat(z, action_proj)) instead of abduction-action-prediction.
    """

    def __init__(self, d_model: int, d_causal: int, d_action: int, dropout: float = 0.1):
        super().__init__()
        self.d_causal = d_causal
        self.d_action = d_action

        self.state_encoder = StateEncoder(
            d_model=d_model,
            d_causal=d_causal,
            dropout=dropout,
        )
        # Simple concat fusion instead of do-operator
        hidden = max(4 * d_causal, d_action // 2)
        self.action_proj = nn.Linear(d_action, d_causal)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_causal, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_causal),
        )
        from causeway.delta_predictor import DeltaPredictor
        self.delta_predictor = DeltaPredictor(d_causal=d_causal, dropout=dropout)

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> DeltaVector:
        z = self.state_encoder(h)
        a = self.action_proj(action)
        z_post = self.fusion(torch.cat([z, a], dim=-1))
        return self.delta_predictor(z, z_post)

    def get_regularization_losses(self) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        z = torch.tensor(0.0, device=device)
        return {
            "acyclicity": z,
            "sparsity": z,
            "edge_count": z,
            "orthogonality": self.state_encoder.orthogonality_loss(),
        }


# ---------------------------------------------------------------------------
# Training + evaluation (shared)
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

    dim_names = DEFAULT_DELTA_DIMS
    corrs, dir_accs, maes = {}, {}, {}
    for i, name in enumerate(dim_names):
        p, t = preds[:, i], targets[:, i]
        if t.std() > 1e-6 and p.std() > 1e-6:
            corrs[name] = torch.corrcoef(torch.stack([p, t]))[0, 1].item()
        else:
            corrs[name] = 0.0
        dir_accs[name] = ((p.sign() == t.sign()).float().mean()).item()
        maes[name] = (p - t).abs().mean().item()

    overall_corr = torch.corrcoef(
        torch.stack([preds.flatten(), targets.flatten()])
    )[0, 1].item()

    return {
        "losses": avg,
        "correlations": corrs,
        "directional_accuracy": dir_accs,
        "mae": maes,
        "overall_corr": overall_corr,
        "overall_mae": (preds - targets).abs().mean().item(),
    }


def run_experiment(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
    lambda_sparse: float = 0.05,
    lambda_edge_count: float = 0.05,
    checkpoint_path: Optional[str] = None,
) -> Dict:
    """
    Run a single ablation experiment. Saves best checkpoint and periodic
    checkpoints every SAVE_EVERY epochs for crash recovery.
    """
    SAVE_EVERY = 25  # Save periodic checkpoint every N epochs

    if checkpoint_path is None:
        checkpoint_path = f"ablation_{name}_best.pt"
    periodic_path = f"ablation_{name}_periodic.pt"

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print(f"  Experiment: {name}")
    print(f"  Params: {n_params / 1e6:.3f}M")
    print(f"  Best checkpoint: {checkpoint_path}")
    print(f"  Periodic checkpoint: {periodic_path} (every {SAVE_EVERY} epochs)")
    print(f"{'='*70}")

    criterion = CausewayLoss(
        lambda_sparse=lambda_sparse,
        lambda_edge_count=lambda_edge_count,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    # Resume from periodic checkpoint if it exists (crash recovery)
    start_epoch = 0
    best_val_loss = float("inf")
    if os.path.exists(periodic_path):
        print(f"  [RESUME] Found periodic checkpoint {periodic_path}, resuming...")
        ckpt = torch.load(periodic_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  [RESUME] Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Temperature anneal (only for Causeway variants with causal_graph)
        if hasattr(model, "causal_graph"):
            progress = epoch / max(args.epochs - 1, 1)
            temp = args.temp_start + (args.temp_end - args.temp_start) * progress
            model.causal_graph.set_temperature(temp)
            ramp = min(1.0, epoch / (args.epochs * 0.3))
            criterion.lambda_acyclic = ramp * 10.0
        else:
            criterion.lambda_acyclic = 0.0

        model.train()
        train_loss = 0.0
        n_batches = 0

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
        train_loss /= n_batches

        val_results = evaluate(model, criterion, val_loader, device)
        val_loss = val_results["losses"]["total"]

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            eta = elapsed / max(epoch - start_epoch + 1, 1) * (args.epochs - epoch - 1)
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"corr={val_results['overall_corr']:.4f}  "
                  f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_results": val_results,
                "best_val_loss": best_val_loss,
            }, checkpoint_path)

        # Save periodic checkpoint for crash recovery
        if epoch % SAVE_EVERY == 0 or epoch == args.epochs - 1:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "val_results": val_results,
            }, periodic_path)

    # Load best and return final metrics
    best_ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final = evaluate(model, criterion, val_loader, device)

    print(f"\n  RESULT [{name}]")
    print(f"  Best val loss: {best_ckpt['val_loss']:.4f} (epoch {best_ckpt['epoch']})")
    print(f"  Overall correlation: {final['overall_corr']:.4f}")
    print(f"  Overall MAE: {final['overall_mae']:.4f}")
    print(f"  Params: {n_params / 1e6:.3f}M")
    print(f"  Per-dim correlations:")
    for dim, corr in final["correlations"].items():
        da = final["directional_accuracy"][dim]
        print(f"    {dim:<25} corr={corr:.4f}  dir_acc={da:.4f}")

    result = {
        "name": name,
        "n_params": n_params,
        "best_epoch": best_ckpt["epoch"],
        "best_val_loss": best_ckpt["val_loss"],
        "overall_corr": final["overall_corr"],
        "overall_mae": final["overall_mae"],
        "correlations": final["correlations"],
        "directional_accuracy": final["directional_accuracy"],
        "mae": final["mae"],
    }

    # Clean up periodic checkpoint after successful completion
    if os.path.exists(periodic_path):
        os.remove(periodic_path)
        print(f"  Cleaned up periodic checkpoint.")

    return result


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(domain: str, model_name: str, args, device):
    """Load (or encode+cache) a dataset. Returns (train_loader, val_loader, d_model)."""
    model_short = model_name.split("/")[-1]
    cache_path = f"cache_{domain}_{model_short}_{args.num_samples}_v2.pt"

    if domain == "clinical":
        from environments.text_clinical import TextClinicalDataset as DatasetClass
    elif domain == "confounded":
        from environments.text_confounded import TextConfoundedDataset as DatasetClass
    else:
        from environments.text_scm import TextSCMDataset as DatasetClass

    batch_encode_size = 8 if any(x in model_short.lower() for x in ["7b", "8b", "13b"]) else 32

    dataset = DatasetClass(
        model_name=model_name,
        num_samples=args.num_samples,
        batch_encode_size=batch_encode_size,
        device=str(device),
        cache_path=cache_path,
    )

    d_model = dataset.d_model
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    print(f"Dataset [{domain}]: {n_train} train, {n_val} val, d_model={d_model}")
    return train_loader, val_loader, d_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Causeway Ablation Suite")
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--d_causal", type=int, default=48)
    p.add_argument("--temp_start", type=float, default=1.0)
    p.add_argument("--temp_end", type=float, default=0.05)
    p.add_argument("--results_file", type=str, default="ablation_results.json",
                   help="JSON file to save results (completed experiments skipped on restart)")
    p.add_argument("--only", type=str, default=None,
                   help="Run only this experiment (e.g. 'ablation_mlp_deployment')")
    return p.parse_args()


def load_results(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(path: str, results: Dict):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {path}")


def print_summary(results: Dict):
    if not results:
        return
    print(f"\n{'='*70}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':<35} {'Corr':>8} {'MAE':>8} {'Params':>10}")
    print(f"  {'-'*65}")
    for name, r in results.items():
        print(f"  {name:<35} {r['overall_corr']:>8.4f} {r['overall_mae']:>8.4f} "
              f"{r['n_params']/1e6:>9.3f}M")
    print(f"{'='*70}")

    # constraint_violation directional accuracy comparison
    print(f"\n  constraint_violation directional accuracy (should be > 0.5):")
    for name, r in results.items():
        cv = r["directional_accuracy"].get("constraint_violation", "N/A")
        print(f"    {name:<35} {cv:.4f}" if isinstance(cv, float) else f"    {name:<35} {cv}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    print(f"{'='*70}")
    print(f"  Causeway Ablation Suite")
    print(f"  Model: {args.model}  |  Epochs: {args.epochs}  |  Samples: {args.num_samples}")
    print(f"  Device: {device}")
    print(f"  Results file: {args.results_file}")
    print(f"{'='*70}\n")

    results = load_results(args.results_file)
    if results:
        print(f"Loaded {len(results)} completed experiment(s) from {args.results_file}")
        print_summary(results)

    # ---- Load deployment dataset (shared by experiments 1-4) ----
    print("\n[DATASET] Loading deployment domain (GPT-2)...")
    train_dep, val_dep, d_model = load_dataset("deployment", args.model, args, device)
    d_action = d_model  # Full Transformer embedding as action

    # ---- Experiment 1: MLP baseline (deployment) ----
    exp_name = "ablation_mlp_deployment"
    if (args.only is None or args.only == exp_name) and exp_name not in results:
        print(f"\n[1/6] MLP Baseline — deployment domain")
        model = MLPBaseline(d_model=d_model, d_action=d_action, hidden_dim=512)
        n = sum(p.numel() for p in model.parameters())
        print(f"  MLP params: {n/1e6:.3f}M (Causeway baseline: 0.794M)")
        result = run_experiment(
            exp_name, model, train_dep, val_dep, args, device,
            lambda_sparse=0.0, lambda_edge_count=0.0,
        )
        results[exp_name] = result
        save_results(args.results_file, results)
    else:
        print(f"\n[1/6] Skipping {exp_name} (already completed)")

    # ---- Experiment 2: Causeway without graph ----
    exp_name = "ablation_no_graph"
    if (args.only is None or args.only == exp_name) and exp_name not in results:
        print(f"\n[2/6] Causeway — no graph (identity pass-through)")
        model = CausewayNoGraph(d_model=d_model, d_causal=args.d_causal, d_action=d_action)
        result = run_experiment(
            exp_name, model, train_dep, val_dep, args, device,
            lambda_sparse=0.0, lambda_edge_count=0.0,
        )
        results[exp_name] = result
        save_results(args.results_file, results)
    else:
        print(f"\n[2/6] Skipping {exp_name} (already completed)")

    # ---- Experiment 3: Causeway without intervention engine ----
    exp_name = "ablation_no_engine"
    if (args.only is None or args.only == exp_name) and exp_name not in results:
        print(f"\n[3/6] Causeway — no intervention engine (concat MLP)")
        model = CausewayNoEngine(d_model=d_model, d_causal=args.d_causal, d_action=d_action)
        result = run_experiment(
            exp_name, model, train_dep, val_dep, args, device,
            lambda_sparse=0.0, lambda_edge_count=0.0,
        )
        results[exp_name] = result
        save_results(args.results_file, results)
    else:
        print(f"\n[3/6] Skipping {exp_name} (already completed)")

    # ---- Experiment 4: Causeway no sparsity ----
    exp_name = "ablation_no_sparsity"
    if (args.only is None or args.only == exp_name) and exp_name not in results:
        print(f"\n[4/6] Causeway — no sparsity penalties (lambda_sparse=lambda_edge_count=0)")
        model = Causeway(d_model=d_model, d_causal=args.d_causal, d_action=d_action)
        result = run_experiment(
            exp_name, model, train_dep, val_dep, args, device,
            lambda_sparse=0.0, lambda_edge_count=0.0,
        )
        results[exp_name] = result
        save_results(args.results_file, results)
    else:
        print(f"\n[4/6] Skipping {exp_name} (already completed)")

    # ---- Load confounded dataset (shared by experiments 5-6) ----
    print("\n[DATASET] Loading confounded domain (GPT-2)...")
    train_conf, val_conf, d_model_conf = load_dataset("confounded", args.model, args, device)

    # ---- Experiment 5: Full Causeway on confounded domain ----
    exp_name = "ablation_confounded"
    if (args.only is None or args.only == exp_name) and exp_name not in results:
        print(f"\n[5/6] Full Causeway — confounded domain")
        model = Causeway(d_model=d_model_conf, d_causal=args.d_causal, d_action=d_model_conf)
        result = run_experiment(
            exp_name, model, train_conf, val_conf, args, device,
            checkpoint_path="ablation_confounded_best.pt",
        )
        results[exp_name] = result
        save_results(args.results_file, results)
    else:
        print(f"\n[5/6] Skipping {exp_name} (already completed)")

    # ---- Experiment 6: MLP baseline on confounded domain ----
    exp_name = "ablation_mlp_confounded"
    if (args.only is None or args.only == exp_name) and exp_name not in results:
        print(f"\n[6/6] MLP Baseline — confounded domain")
        model = MLPBaseline(d_model=d_model_conf, d_action=d_model_conf, hidden_dim=512)
        result = run_experiment(
            exp_name, model, train_conf, val_conf, args, device,
            lambda_sparse=0.0, lambda_edge_count=0.0,
            checkpoint_path="ablation_mlp_confounded_best.pt",
        )
        results[exp_name] = result
        save_results(args.results_file, results)
    else:
        print(f"\n[6/6] Skipping {exp_name} (already completed)")

    # ---- Final summary ----
    print_summary(results)
    print(f"\nAll results saved to {args.results_file}")


if __name__ == "__main__":
    main()
