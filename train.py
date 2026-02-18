"""
Causeway Training Script.

Trains the Causeway module on the synthetic SCM environment
with ground-truth counterfactual supervision.

Usage:
    python train.py
    python train.py --d_causal 64 --epochs 200 --lr 1e-3
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causeway.causeway_module import Causeway
from causeway.losses import CausewayLoss
from environments.synthetic_scm import SyntheticSCM, SCMDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Causeway")
    parser.add_argument("--d_model", type=int, default=64, help="Simulated Transformer hidden dim")
    parser.add_argument("--d_causal", type=int, default=32, help="Number of causal variables")
    parser.add_argument("--d_action", type=int, default=32, help="Action representation dim")
    parser.add_argument("--graph_layers", type=int, default=2, help="Causal graph message-passing layers")
    parser.add_argument("--propagation_steps", type=int, default=3, help="Intervention propagation depth")
    parser.add_argument("--num_samples", type=int, default=50000, help="Training dataset size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--lambda_acyclic", type=float, default=1.0, help="Acyclicity constraint weight")
    parser.add_argument("--lambda_sparse", type=float, default=0.01, help="Sparsity penalty weight")
    parser.add_argument("--lambda_edge_count", type=float, default=0.1, help="Edge count (L0) penalty weight")
    parser.add_argument("--lambda_ortho", type=float, default=0.1, help="Orthogonality penalty weight")
    parser.add_argument("--acyclic_ramp", action="store_true", default=True,
                        help="Gradually increase acyclicity penalty")
    parser.add_argument("--temp_start", type=float, default=1.0, help="Gumbel-sigmoid start temperature")
    parser.add_argument("--temp_end", type=float, default=0.1, help="Gumbel-sigmoid end temperature")
    parser.add_argument("--noise_scale", type=float, default=0.05, help="SCM exogenous noise scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_path", type=str, default="causeway_checkpoint.pt", help="Model save path")
    return parser.parse_args()


def train_epoch(
    model: Causeway,
    criterion: CausewayLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    acyclic_ramp: bool,
) -> dict:
    """Train for one epoch. Returns average losses."""
    model.train()
    total_losses = {}
    n_batches = 0

    # Optionally ramp up acyclicity penalty over training
    if acyclic_ramp:
        ramp = min(1.0, epoch / (total_epochs * 0.3))
        criterion.lambda_acyclic = ramp * 10.0  # ramp from 0 to 10

    for h, action, target in dataloader:
        h, action, target = h.to(device), action.to(device), target.to(device)

        # Forward pass
        delta = model(h, action)
        reg_losses = model.get_regularization_losses()

        # Compute loss
        losses = criterion(delta, target, reg_losses)

        # Backward pass
        optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(
    model: Causeway,
    criterion: CausewayLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate on validation set. Returns average losses + metrics."""
    model.eval()
    total_losses = {}
    all_preds = []
    all_targets = []
    n_batches = 0

    for h, action, target in dataloader:
        h, action, target = h.to(device), action.to(device), target.to(device)

        delta = model(h, action)
        reg_losses = model.get_regularization_losses()
        losses = criterion(delta, target, reg_losses)

        all_preds.append(delta.values)
        all_targets.append(target)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    avg_losses = {k: v / n_batches for k, v in total_losses.items()}

    # Compute additional metrics
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Per-dimension correlation (how well we track the direction of change)
    dim_names = ["risk_shift", "goal_progress", "constraint_violation",
                 "resource_cost", "success_probability"]
    correlations = {}
    for i, name in enumerate(dim_names):
        p, t = preds[:, i], targets[:, i]
        if t.std() > 1e-6 and p.std() > 1e-6:
            corr = torch.corrcoef(torch.stack([p, t]))[0, 1].item()
        else:
            corr = 0.0
        correlations[name] = corr

    # Directional accuracy: did we get the sign of the delta right?
    sign_accuracy = ((preds.sign() == targets.sign()).float().mean(dim=0))
    dir_acc = {name: sign_accuracy[i].item() for i, name in enumerate(dim_names)}

    # MAE per dimension
    mae = (preds - targets).abs().mean(dim=0)
    maes = {name: mae[i].item() for i, name in enumerate(dim_names)}

    return {
        "losses": avg_losses,
        "correlations": correlations,
        "directional_accuracy": dir_acc,
        "mae": maes,
    }


def main():
    args = parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create environment
    print("Generating synthetic SCM dataset...")
    scm = SyntheticSCM(noise_scale=args.noise_scale, seed=args.seed)
    dataset = SCMDataset(
        scm=scm,
        num_samples=args.num_samples,
        d_model=args.d_model,
        d_action=args.d_action,
        seed=args.seed,
    )

    # Train/val split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    print(f"Train: {n_train}, Val: {n_val}")

    # Create model
    model = Causeway(
        d_model=args.d_model,
        d_causal=args.d_causal,
        d_action=args.d_action,
        graph_layers=args.graph_layers,
        propagation_steps=args.propagation_steps,
    ).to(device)

    diagnostics = model.get_diagnostics()
    print(f"Causeway parameters: {diagnostics['total_parameters_human']}")

    # Loss and optimizer
    criterion = CausewayLoss(
        lambda_acyclic=args.lambda_acyclic,
        lambda_sparse=args.lambda_sparse,
        lambda_edge_count=args.lambda_edge_count,
        lambda_ortho=args.lambda_ortho,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Training loop
    best_val_loss = float("inf")
    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        # Anneal Gumbel-sigmoid temperature: start warm, end cold
        progress = epoch / max(args.epochs - 1, 1)
        temp = args.temp_start + (args.temp_end - args.temp_start) * progress
        model.causal_graph.set_temperature(temp)

        # Train
        train_losses = train_epoch(
            model, criterion, train_loader, optimizer, device,
            epoch, args.epochs, args.acyclic_ramp,
        )

        # Evaluate
        val_results = evaluate(model, criterion, val_loader, device)
        val_loss = val_results["losses"]["total"]

        # Step scheduler
        scheduler.step()

        # Print progress
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            graph_stats = model.causal_graph.get_graph_stats()
            print(f"Epoch {epoch:3d}/{args.epochs}")
            print(f"  Train loss: {train_losses['total']:.4f} "
                  f"(delta={train_losses['delta']:.4f}, "
                  f"acyclic={train_losses['acyclicity']:.4f}, "
                  f"edges={train_losses.get('edge_count', 0):.4f})")
            print(f"  Val loss:   {val_loss:.4f}")
            print(f"  Graph: {graph_stats['hard_edges']} hard edges "
                  f"(expected={graph_stats['expected_edges']}), "
                  f"temp={graph_stats['temperature']:.3f}")
            print(f"  Correlations: ", end="")
            for name, corr in val_results["correlations"].items():
                print(f"{name[:8]}={corr:.3f} ", end="")
            print()
            print(f"  Dir accuracy: ", end="")
            for name, acc in val_results["directional_accuracy"].items():
                print(f"{name[:8]}={acc:.3f} ", end="")
            print("\n")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_results": val_results,
                "args": vars(args),
                "diagnostics": diagnostics,
            }, args.save_path)

    # Final evaluation
    print("=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(args.save_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_results = evaluate(model, criterion, val_loader, device)

    print(f"\nBest validation loss: {checkpoint['val_loss']:.4f} (epoch {checkpoint['epoch']})")
    print(f"\nPer-dimension results:")
    print(f"{'Dimension':<25} {'Correlation':>12} {'Dir Acc':>10} {'MAE':>10}")
    print("-" * 60)
    for name in final_results["correlations"]:
        corr = final_results["correlations"][name]
        acc = final_results["directional_accuracy"][name]
        mae = final_results["mae"][name]
        print(f"{name:<25} {corr:>12.4f} {acc:>10.4f} {mae:>10.4f}")

    print(f"\nGraph diagnostics:")
    stats = model.causal_graph.get_graph_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\nModel saved to: {args.save_path}")
    print(f"Parameters: {diagnostics['total_parameters_human']}")


if __name__ == "__main__":
    main()
