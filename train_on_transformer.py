"""
Train Causeway on Real Transformer Hidden States.

Encodes SCM scenarios as text, runs them through a frozen Transformer,
and trains Causeway to predict counterfactual deltas from the real
hidden state representations.

Supports any HuggingFace model: GPT-2, TinyLlama, LLaMA-3.2, etc.

Usage:
    python train_on_transformer.py --model gpt2
    python train_on_transformer.py --model gpt2 --d_causal 48 --epochs 200 --num_samples 50000
    python train_on_transformer.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python train_on_transformer.py --model meta-llama/Llama-3.2-1B

V2 changes:
    - d_action defaults to d_model (768 for GPT-2) to match full action embeddings
    - Increased samples: 20K → 50K for better coverage of the representation space
    - Increased epochs: 120 → 200 for convergence with larger capacity
    - 10-epoch linear LR warmup before cosine annealing prevents early instability
    - New cache path (*_v2.pt) to avoid loading stale v1 data
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causeway.causeway_module import Causeway
from causeway.losses import CausewayLoss


def parse_args():
    p = argparse.ArgumentParser(description="Train Causeway on Transformer")
    p.add_argument("--domain", type=str, default="deployment",
                   choices=["deployment", "clinical", "confounded"],
                   help="SCM domain: deployment (software), clinical (treatment), or confounded (neutral causal)")
    p.add_argument("--model", type=str, default="gpt2",
                   help="HuggingFace model name (gpt2, TinyLlama/TinyLlama-1.1B-Chat-v1.0, etc)")
    p.add_argument("--d_causal", type=int, default=48)
    p.add_argument("--d_action", type=int, default=None,
                   help="Action dim. Defaults to d_model (768 for GPT-2) to preserve full action embeddings.")
    p.add_argument("--graph_layers", type=int, default=2)
    p.add_argument("--propagation_steps", type=int, default=3)
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_encode_size", type=int, default=None,
                   help="Batch size for Transformer encoding. Defaults to 32 for small models, 8 for 7B+.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_epochs", type=int, default=10,
                   help="Number of linear LR warmup epochs before cosine annealing.")
    p.add_argument("--lambda_edge_count", type=float, default=0.05)
    p.add_argument("--lambda_sparse", type=float, default=0.05)
    p.add_argument("--temp_start", type=float, default=1.0)
    p.add_argument("--temp_end", type=float, default=0.05)
    p.add_argument("--save_path", type=str, default=None,
                   help="Checkpoint save path (auto-named if not set)")
    return p.parse_args()


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

    dim_names = ["risk_shift", "goal_progress", "constraint_viol",
                 "resource_cost", "success_prob"]
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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    model_short = args.model.split("/")[-1]
    if args.save_path is None:
        if args.domain == "deployment":
            args.save_path = f"causeway_{model_short}.pt"
        else:
            args.save_path = f"causeway_{args.domain}_{model_short}.pt"

    print(f"{'='*70}")
    print(f"  Training Causeway on: {args.model} ({args.domain} domain)")
    print(f"{'='*70}")
    print(f"Device: {device}")

    # ---- Dataset ----
    # Domain-aware imports and cache paths
    if args.domain == "clinical":
        from environments.text_clinical import TextClinicalDataset as DatasetClass
    elif args.domain == "confounded":
        from environments.text_confounded import TextConfoundedDataset as DatasetClass
    else:
        from environments.text_scm import TextSCMDataset as DatasetClass

    cache_path = f"cache_{args.domain}_{model_short}_{args.num_samples}_v2.pt"

    # Auto-scale batch_encode_size: smaller for large models to fit in VRAM
    if args.batch_encode_size is None:
        args.batch_encode_size = 8 if "7b" in model_short.lower() or "8b" in model_short.lower() else 32
        print(f"batch_encode_size auto-set to {args.batch_encode_size}")

    dataset = DatasetClass(
        model_name=args.model,
        num_samples=args.num_samples,
        batch_encode_size=args.batch_encode_size,
        device=str(device),
        cache_path=cache_path,
    )

    d_model = dataset.d_model
    # Default d_action to d_model (full Transformer embedding dim) if not specified
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

    print(f"\nDataset: {n_train} train, {n_val} val")
    print(f"Backbone d_model: {d_model}")
    print(f"h dtype: {dataset.h.dtype}, action dtype: {dataset.actions.dtype}")

    # ---- Model ----
    model = Causeway(
        d_model=d_model,
        d_causal=args.d_causal,
        d_action=args.d_action,
        graph_layers=args.graph_layers,
        propagation_steps=args.propagation_steps,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Causeway params: {n_params / 1e6:.3f}M")

    # ---- Loss + Optimizer ----
    criterion = CausewayLoss(
        lambda_sparse=args.lambda_sparse,
        lambda_edge_count=args.lambda_edge_count,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # LR schedule: linear warmup for warmup_epochs, then cosine annealing.
    # Warmup prevents early gradient instability from the large 768-dim inputs.
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

    # ---- Training ----
    best_val_loss = float("inf")
    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        # Temperature anneal
        progress = epoch / max(args.epochs - 1, 1)
        temp = args.temp_start + (args.temp_end - args.temp_start) * progress
        model.causal_graph.set_temperature(temp)

        # Acyclicity ramp
        ramp = min(1.0, epoch / (args.epochs * 0.3))
        criterion.lambda_acyclic = ramp * 10.0

        model.train()
        train_loss = 0
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

        # Evaluate
        val_results = evaluate(model, criterion, val_loader, device)
        val_loss = val_results["losses"]["total"]

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            gs = model.causal_graph.get_graph_stats()
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"edges={gs['expected_edges']}  temp={gs['temperature']:.3f}")
            print(f"  Corr: ", end="")
            for name, c in val_results["correlations"].items():
                print(f"{name[:8]}={c:.3f} ", end="")
            print(f"\n  DirAcc:", end="")
            for name, a in val_results["directional_accuracy"].items():
                print(f"{name[:8]}={a:.3f} ", end="")
            print()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_results": val_results,
                "args": vars(args),
                "backbone": args.model,
                "d_model": d_model,
            }, args.save_path)

    # ---- Final ----
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS: Causeway on {args.model}")
    print(f"{'='*70}")

    ckpt = torch.load(args.save_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final = evaluate(model, criterion, val_loader, device)

    print(f"\nBest val loss: {ckpt['val_loss']:.4f} (epoch {ckpt['epoch']})")
    print(f"Overall correlation: {final['overall_corr']:.4f}")
    print(f"Overall MAE: {final['overall_mae']:.4f}")

    print(f"\n{'Dimension':<25} {'Correlation':>12} {'Dir Acc':>10} {'MAE':>10}")
    print("-" * 60)
    for name in final["correlations"]:
        c = final["correlations"][name]
        a = final["directional_accuracy"][name]
        m = final["mae"][name]
        print(f"{name:<25} {c:>12.4f} {a:>10.4f} {m:>10.4f}")

    gs = model.causal_graph.get_graph_stats()
    print(f"\nGraph: {gs}")
    print(f"Params: {n_params / 1e6:.3f}M")
    print(f"Saved: {args.save_path}")


if __name__ == "__main__":
    main()
