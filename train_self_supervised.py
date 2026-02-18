"""
Train Causeway Self-Supervised: Reconstruction through Causal Bottleneck.

The Transformer's own representation shifts serve as the training signal.
No external ground truth needed.

Architecture:
    Causeway pipeline:
        h_base → StateEncoder → z → CausalGraph → z_refined
        action → InterventionEngine(z_refined, action, adj) → z_cf
        (z_refined, z_cf) → DeltaPredictor → delta (5-dim structured)
        (delta.values, delta.confidence, z_refined) → DeltaDecoder → h_delta_pred

    Baseline (control):
        (h_base, action) → flat MLP → h_delta_pred

    Training target: h_delta = h_intervened - h_base (from frozen Transformer)

If Causeway + DeltaDecoder outperforms the parameter-matched BaselineMLP,
the causal bottleneck is extracting meaningful structure from the
Transformer's representation shifts.

Usage:
    python train_self_supervised.py --model gpt2
    python train_self_supervised.py --model gpt2 --d_causal 48 --epochs 200
"""

import argparse
import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causeway.causeway_module import Causeway
from environments.self_supervised import SelfSupervisedDataset, DOMAINS


# ── New Modules ──────────────────────────────────────────────────────

class DeltaDecoder(nn.Module):
    """
    Reconstructs h_delta from Causeway's causal bottleneck output.

    Input: delta.values (5) + delta.confidence (5) + z_refined (d_causal)
    Output: h_delta_pred (d_model)

    This is the reconstruction head that proves the causal bottleneck
    preserves the essential structure of the Transformer's representation shift.
    The severe dimensionality expansion (58 → 768 for GPT-2) is intentional:
    if Causeway's 48-dim causal space captures the right factors, the decoder
    can reconstruct the high-dim shift; if not, it can't fake it.
    """

    def __init__(self, d_causal: int, d_model: int, hidden_dim: int = 512):
        super().__init__()
        input_dim = 5 + 5 + d_causal

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, delta_values, delta_confidence, z_refined):
        x = torch.cat([delta_values, delta_confidence, z_refined], dim=-1)
        return self.net(x)


class BaselineMLP(nn.Module):
    """
    Flat MLP baseline: predicts h_delta from (h_base, action) directly.

    No causal structure, no bottleneck. Gets the full (d_model + d_model)
    input that Causeway compresses through a d_causal bottleneck.
    Parameter count is matched to Causeway + DeltaDecoder for fair comparison.
    """

    def __init__(self, d_model: int, hidden_dim: int = 512):
        super().__init__()
        input_dim = 2 * d_model

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, h_base, action):
        x = torch.cat([h_base, action], dim=-1)
        return self.net(x)

    @staticmethod
    def compute_hidden_dim(target_params: int, d_model: int) -> int:
        """Find hidden_dim that yields approximately target_params total."""
        # Approximate param count for 3-layer MLP with LayerNorm:
        #   (2*d_model)*h + h + h*h + h + h*d_model + d_model + layernorm overhead
        # ≈ h^2 + (3*d_model)*h   (dominant terms)
        # Solve: h^2 + 3*d_model*h - target_params = 0
        a = 1
        b = 3 * d_model
        c = -target_params
        h = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
        return max(64, int(h))


# ── Args ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Causeway Self-Supervised")
    p.add_argument("--model", type=str, default="gpt2",
                   help="HuggingFace model name")
    p.add_argument("--d_causal", type=int, default=48)
    p.add_argument("--d_action", type=int, default=None,
                   help="Action dim. Defaults to d_model.")
    p.add_argument("--graph_layers", type=int, default=2)
    p.add_argument("--propagation_steps", type=int, default=3)
    p.add_argument("--decoder_hidden", type=int, default=512,
                   help="Hidden dim for DeltaDecoder.")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_encode_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--lambda_cos", type=float, default=0.5,
                   help="Weight for cosine direction loss.")
    p.add_argument("--lambda_sparse", type=float, default=0.05)
    p.add_argument("--lambda_edge_count", type=float, default=0.05)
    p.add_argument("--temp_start", type=float, default=1.0)
    p.add_argument("--temp_end", type=float, default=0.05)
    p.add_argument("--save_path", type=str, default=None)
    return p.parse_args()


# ── Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_causeway(causeway, decoder, loader, device):
    """Evaluate Causeway + DeltaDecoder reconstruction quality."""
    causeway.eval()
    decoder.eval()

    all_preds, all_targets, all_domains = [], [], []

    for h_base, action, h_delta, d_idx in loader:
        h_base = h_base.to(device)
        action = action.to(device)
        h_delta = h_delta.to(device)

        # Step through Causeway components to access z_refined
        z = causeway.state_encoder(h_base)
        z_refined = causeway.causal_graph(z)
        adj = causeway.causal_graph.adjacency
        z_cf, mask, eps = causeway.intervention_engine(
            z_refined, action, adj, causeway.propagation_steps
        )
        delta = causeway.delta_predictor(z_refined, z_cf)
        h_delta_pred = decoder(delta.values, delta.confidence, z_refined)

        all_preds.append(h_delta_pred.cpu())
        all_targets.append(h_delta.cpu())
        all_domains.append(d_idx)

    return _compute_metrics(all_preds, all_targets, all_domains)


@torch.no_grad()
def evaluate_baseline(baseline, loader, device):
    """Evaluate BaselineMLP reconstruction quality."""
    baseline.eval()

    all_preds, all_targets, all_domains = [], [], []

    for h_base, action, h_delta, d_idx in loader:
        h_base = h_base.to(device)
        action = action.to(device)
        h_delta = h_delta.to(device)

        h_delta_pred = baseline(h_base, action)

        all_preds.append(h_delta_pred.cpu())
        all_targets.append(h_delta.cpu())
        all_domains.append(d_idx)

    return _compute_metrics(all_preds, all_targets, all_domains)


def _compute_metrics(all_preds, all_targets, all_domains):
    """Shared metric computation for both models."""
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    domains = torch.cat(all_domains)

    # Global
    mse = F.mse_loss(preds, targets).item()
    cosine = F.cosine_similarity(preds, targets, dim=-1).mean().item()

    pred_norms = preds.norm(dim=-1)
    target_norms = targets.norm(dim=-1)
    rel_mag = ((pred_norms - target_norms).abs() / (target_norms + 1e-8)).mean().item()

    # Per-domain
    per_domain = {}
    for d_idx, domain in enumerate(DOMAINS):
        mask = domains == d_idx
        if mask.sum() == 0:
            continue
        d_p = preds[mask]
        d_t = targets[mask]
        d_pn = d_p.norm(dim=-1)
        d_tn = d_t.norm(dim=-1)
        per_domain[domain['name']] = {
            'mse': F.mse_loss(d_p, d_t).item(),
            'cosine': F.cosine_similarity(d_p, d_t, dim=-1).mean().item(),
            'rel_mag': ((d_pn - d_tn).abs() / (d_tn + 1e-8)).mean().item(),
            'n': mask.sum().item(),
        }

    return {'mse': mse, 'cosine': cosine, 'rel_mag': rel_mag, 'per_domain': per_domain}


# ── Training ─────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    model_short = args.model.split("/")[-1]
    if args.save_path is None:
        args.save_path = f"causeway_ss_{model_short}.pt"

    print(f"{'='*70}")
    print(f"  Self-Supervised Causeway Training")
    print(f"  Backbone: {args.model}")
    print(f"{'='*70}")
    print(f"Device: {device}")

    # ── Dataset ────────────────────────────────────────────
    cache_path = f"cache_ss_{model_short}_{args.num_samples}.pt"

    if args.batch_encode_size is None:
        args.batch_encode_size = 8 if "7b" in model_short.lower() or "8b" in model_short.lower() else 32
        print(f"batch_encode_size auto-set to {args.batch_encode_size}")

    dataset = SelfSupervisedDataset(
        model_name=args.model,
        num_samples=args.num_samples,
        batch_encode_size=args.batch_encode_size,
        device=str(device),
        cache_path=cache_path,
    )

    d_model = dataset.d_model
    if args.d_action is None:
        args.d_action = d_model
        print(f"d_action defaulting to d_model={d_model}")

    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    print(f"\nDataset: {n_train} train, {n_val} val, d_model={d_model}")

    # ══════════════════════════════════════════════════════════════════
    #  Phase 1: Causeway + DeltaDecoder
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  Phase 1: Causeway + DeltaDecoder")
    print(f"{'='*70}")

    causeway = Causeway(
        d_model=d_model,
        d_causal=args.d_causal,
        d_action=args.d_action,
        graph_layers=args.graph_layers,
        propagation_steps=args.propagation_steps,
    ).to(device)

    decoder = DeltaDecoder(
        d_causal=args.d_causal,
        d_model=d_model,
        hidden_dim=args.decoder_hidden,
    ).to(device)

    causeway_params = sum(p.numel() for p in causeway.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    total_params = causeway_params + decoder_params
    print(f"Causeway params: {causeway_params / 1e6:.3f}M")
    print(f"Decoder params:  {decoder_params / 1e6:.3f}M")
    print(f"Total:           {total_params / 1e6:.3f}M")

    all_cw_params = list(causeway.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(all_cw_params, lr=args.lr, weight_decay=1e-5)

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[args.warmup_epochs],
    )

    best_val_mse = float("inf")
    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        # Temperature anneal for Gumbel-sigmoid
        progress = epoch / max(args.epochs - 1, 1)
        temp = args.temp_start + (args.temp_end - args.temp_start) * progress
        causeway.causal_graph.set_temperature(temp)

        # Ramp up acyclicity constraint
        ramp = min(1.0, epoch / (args.epochs * 0.3))
        lambda_acyclic = ramp * 10.0

        causeway.train()
        decoder.train()
        train_mse_sum = 0.0
        train_cos_sum = 0.0
        n_batches = 0

        for h_base, action, h_delta, _d_idx in train_loader:
            h_base = h_base.to(device)
            action = action.to(device)
            h_delta = h_delta.to(device)

            # Forward through Causeway components individually
            # (need z_refined for the decoder — can't use causeway.forward())
            z = causeway.state_encoder(h_base)
            z_refined = causeway.causal_graph(z)
            adj = causeway.causal_graph.adjacency
            z_cf, mask, eps = causeway.intervention_engine(
                z_refined, action, adj, causeway.propagation_steps
            )
            delta = causeway.delta_predictor(z_refined, z_cf)

            # Reconstruct h_delta through causal bottleneck
            h_delta_pred = decoder(delta.values, delta.confidence, z_refined)

            # Reconstruction losses
            mse_loss = F.mse_loss(h_delta_pred, h_delta)
            cos_sim = F.cosine_similarity(h_delta_pred, h_delta, dim=-1).mean()
            cos_loss = 1.0 - cos_sim

            # Causeway structural regularization
            reg = causeway.get_regularization_losses()
            reg_loss = (
                lambda_acyclic * reg['acyclicity']
                + args.lambda_sparse * reg['sparsity']
                + args.lambda_edge_count * reg['edge_count']
                + 0.1 * reg['orthogonality']
            )

            total_loss = mse_loss + args.lambda_cos * cos_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(all_cw_params, 1.0)
            optimizer.step()

            train_mse_sum += mse_loss.item()
            train_cos_sum += cos_sim.item()
            n_batches += 1

        scheduler.step()
        train_mse = train_mse_sum / n_batches
        train_cos = train_cos_sum / n_batches

        # Evaluate every epoch for checkpointing
        val = evaluate_causeway(causeway, decoder, val_loader, device)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            gs = causeway.causal_graph.get_graph_stats()
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train_mse={train_mse:.6f}  cos={train_cos:.4f}  "
                  f"val_mse={val['mse']:.6f}  val_cos={val['cosine']:.4f}  "
                  f"edges={gs['expected_edges']}  temp={gs['temperature']:.3f}")

        if val['mse'] < best_val_mse:
            best_val_mse = val['mse']
            torch.save({
                'epoch': epoch,
                'causeway_state': causeway.state_dict(),
                'decoder_state': decoder.state_dict(),
                'val_mse': val['mse'],
                'val_cosine': val['cosine'],
                'val_results': val,
                'args': vars(args),
                'backbone': args.model,
                'd_model': d_model,
            }, args.save_path)

    # Load best Causeway checkpoint
    ckpt = torch.load(args.save_path, weights_only=False)
    causeway.load_state_dict(ckpt['causeway_state'])
    decoder.load_state_dict(ckpt['decoder_state'])
    causeway_final = evaluate_causeway(causeway, decoder, val_loader, device)

    print(f"\nPhase 1 best: epoch {ckpt['epoch']}, "
          f"MSE={causeway_final['mse']:.6f}, cos={causeway_final['cosine']:.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  Phase 2: Baseline MLP (parameter-matched control)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  Phase 2: Baseline MLP (parameter-matched control)")
    print(f"{'='*70}")

    baseline_hidden = BaselineMLP.compute_hidden_dim(total_params, d_model)
    baseline = BaselineMLP(d_model=d_model, hidden_dim=baseline_hidden).to(device)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"Baseline hidden dim: {baseline_hidden}")
    print(f"Baseline params:     {baseline_params / 1e6:.3f}M  "
          f"(target: {total_params / 1e6:.3f}M)")

    bl_optimizer = torch.optim.AdamW(
        baseline.parameters(), lr=args.lr, weight_decay=1e-5
    )
    bl_warmup = torch.optim.lr_scheduler.LinearLR(
        bl_optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    bl_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        bl_optimizer, T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01,
    )
    bl_scheduler = torch.optim.lr_scheduler.SequentialLR(
        bl_optimizer,
        schedulers=[bl_warmup, bl_cosine],
        milestones=[args.warmup_epochs],
    )

    best_bl_mse = float("inf")
    bl_save_path = args.save_path.replace('.pt', '_baseline.pt')
    print(f"\nTraining baseline for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        baseline.train()
        train_mse_sum = 0.0
        n_batches = 0

        for h_base, action, h_delta, _d_idx in train_loader:
            h_base = h_base.to(device)
            action = action.to(device)
            h_delta = h_delta.to(device)

            h_delta_pred = baseline(h_base, action)

            mse_loss = F.mse_loss(h_delta_pred, h_delta)
            cos_loss = 1.0 - F.cosine_similarity(h_delta_pred, h_delta, dim=-1).mean()
            total_loss = mse_loss + args.lambda_cos * cos_loss

            bl_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
            bl_optimizer.step()

            train_mse_sum += mse_loss.item()
            n_batches += 1

        bl_scheduler.step()
        train_mse = train_mse_sum / n_batches

        # Evaluate every epoch for checkpointing
        bl_val = evaluate_baseline(baseline, val_loader, device)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train_mse={train_mse:.6f}  "
                  f"val_mse={bl_val['mse']:.6f}  val_cos={bl_val['cosine']:.4f}")

        if bl_val['mse'] < best_bl_mse:
            best_bl_mse = bl_val['mse']
            torch.save({
                'epoch': epoch,
                'model_state': baseline.state_dict(),
                'val_mse': bl_val['mse'],
                'val_cosine': bl_val['cosine'],
                'val_results': bl_val,
            }, bl_save_path)

    # Load best baseline checkpoint
    bl_ckpt = torch.load(bl_save_path, weights_only=False)
    baseline.load_state_dict(bl_ckpt['model_state'])
    baseline_final = evaluate_baseline(baseline, val_loader, device)

    print(f"\nPhase 2 best: epoch {bl_ckpt['epoch']}, "
          f"MSE={baseline_final['mse']:.6f}, cos={baseline_final['cosine']:.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  Final Comparison
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  RESULTS: Causeway vs Baseline MLP")
    print(f"  Backbone: {args.model} (d_model={d_model})")
    print(f"{'='*70}")

    print(f"\n{'Metric':<25} {'Causeway':>12} {'Baseline':>12} {'Winner':>10}")
    print("-" * 62)

    metrics = [
        ('MSE (lower=better)',       causeway_final['mse'],     baseline_final['mse'],     'lower'),
        ('Cosine (higher=better)',    causeway_final['cosine'],  baseline_final['cosine'],  'higher'),
        ('RelMag (lower=better)',     causeway_final['rel_mag'], baseline_final['rel_mag'], 'lower'),
    ]

    cw_wins = 0
    for name, cw_val, bl_val, direction in metrics:
        if direction == 'lower':
            winner = "Causeway" if cw_val < bl_val else "Baseline"
            if cw_val < bl_val:
                cw_wins += 1
        else:
            winner = "Causeway" if cw_val > bl_val else "Baseline"
            if cw_val > bl_val:
                cw_wins += 1
        print(f"{name:<25} {cw_val:>12.6f} {bl_val:>12.6f} {winner:>10}")

    # Per-domain breakdown
    print(f"\n{'Domain':<18} {'CW MSE':>10} {'BL MSE':>10} "
          f"{'CW Cos':>10} {'BL Cos':>10} {'Winner':>10}")
    print("-" * 72)

    domain_cw_wins = 0
    for dname in causeway_final['per_domain']:
        cw_d = causeway_final['per_domain'][dname]
        bl_d = baseline_final['per_domain'][dname]
        winner = "Causeway" if cw_d['mse'] < bl_d['mse'] else "Baseline"
        if cw_d['mse'] < bl_d['mse']:
            domain_cw_wins += 1
        print(f"{dname:<18} {cw_d['mse']:>10.6f} {bl_d['mse']:>10.6f} "
              f"{cw_d['cosine']:>10.4f} {bl_d['cosine']:>10.4f} {winner:>10}")

    # Summary
    gs = causeway.causal_graph.get_graph_stats()
    print(f"\nCauseway graph: {gs}")
    print(f"Causeway + Decoder: {total_params / 1e6:.3f}M params")
    print(f"Baseline MLP:       {baseline_params / 1e6:.3f}M params")
    print(f"Bottleneck:         {args.d_causal} causal dims -> {d_model} reconstruction dims")
    print(f"Domains won:        Causeway {domain_cw_wins}/{len(DOMAINS)}")
    print(f"\nSaved: {args.save_path}, {bl_save_path}")

    print(f"\n{'='*70}")
    if cw_wins >= 2:
        print("  VERDICT: Causal bottleneck OUTPERFORMS flat MLP.")
        print("  The causal structure extracts meaningful decomposition")
        print("  from the Transformer's representation shifts.")
    elif cw_wins == 0:
        print("  VERDICT: Flat MLP outperforms causal bottleneck.")
        print("  The causal structure may need more capacity or")
        print("  architectural changes to capture representation shifts.")
    else:
        print("  VERDICT: Mixed results — neither clearly dominates.")
        print("  The causal bottleneck shows promise but needs refinement.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
