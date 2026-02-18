"""
Causeway Evaluation Script.

Loads a trained Causeway checkpoint and runs detailed evaluation,
including counterfactual accuracy, graph recovery, and example predictions.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint causeway_checkpoint.pt
"""

import argparse
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causeway.causeway_module import Causeway
from causeway.losses import CausewayLoss
from environments.synthetic_scm import SyntheticSCM, SCMDataset, NUM_VARIABLES


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Causeway")
    parser.add_argument("--checkpoint", type=str, default="causeway_checkpoint.pt")
    parser.add_argument("--num_examples", type=int, default=5, help="Detailed examples to show")
    parser.add_argument("--seed", type=int, default=123, help="Eval seed (different from training)")
    return parser.parse_args()


def graph_recovery_analysis(model: Causeway, scm: SyntheticSCM):
    """Compare learned causal graph to ground truth."""
    print("\n" + "=" * 60)
    print("GRAPH RECOVERY ANALYSIS")
    print("=" * 60)

    # Ground truth adjacency (only over causal variables space,
    # which is larger than the SCM's 8 variables)
    gt_adj = scm.get_adjacency_tensor()

    # Learned adjacency
    learned_adj = model.causal_graph.adjacency_probs.detach().cpu()

    print(f"\nGround truth graph: {NUM_VARIABLES}x{NUM_VARIABLES}, "
          f"{(gt_adj.abs() > 0).sum().item()} edges")
    print(f"Learned graph: {learned_adj.shape[0]}x{learned_adj.shape[1]}, "
          f"{(learned_adj > 0.5).sum().item()} edges (threshold=0.5)")

    # Graph statistics
    stats = model.causal_graph.get_graph_stats()
    print(f"\nLearned graph stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Show top learned edges
    d = learned_adj.shape[0]
    edges = []
    for i in range(d):
        for j in range(d):
            if i != j and learned_adj[i, j] > 0.3:
                edges.append((i, j, learned_adj[i, j].item()))
    edges.sort(key=lambda x: -x[2])

    print(f"\nTop learned edges (prob > 0.3):")
    for src, dst, prob in edges[:20]:
        print(f"  {src:2d} -> {dst:2d}  (prob={prob:.3f})")


def counterfactual_examples(
    model: Causeway,
    dataset: SCMDataset,
    device: torch.device,
    n_examples: int = 5,
):
    """Show detailed counterfactual predictions vs ground truth."""
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL EXAMPLES")
    print("=" * 60)

    dim_names = ["risk_shift", "goal_progress", "constraint_viol",
                 "resource_cost", "success_prob"]

    model.eval()
    with torch.no_grad():
        for i in range(min(n_examples, len(dataset))):
            h, action, target = dataset[i]
            h = h.unsqueeze(0).to(device)
            action = action.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)

            delta = model(h, action)
            pred = delta.values[0].cpu()
            conf = delta.confidence[0].cpu()
            tgt = target[0].cpu()

            print(f"\n--- Example {i + 1} ---")
            print(f"  Intervention mask (raw): "
                  f"{dataset.intervention_masks[i][:4].numpy()}")
            print(f"  {'Dimension':<22} {'Predicted':>10} {'Actual':>10} "
                  f"{'Error':>10} {'Conf':>8} {'Sign OK':>8}")
            print(f"  {'-'*70}")

            for j, name in enumerate(dim_names):
                p, t = pred[j].item(), tgt[j].item()
                err = abs(p - t)
                c = conf[j].item()
                sign_ok = "Y" if (p >= 0) == (t >= 0) else "N"
                print(f"  {name:<22} {p:>10.4f} {t:>10.4f} "
                      f"{err:>10.4f} {c:>8.3f} {sign_ok:>8}")


def stress_test(
    model: Causeway,
    scm: SyntheticSCM,
    device: torch.device,
    training_seed: int = 42,
    n_samples: int = 1000,
):
    """Test on fresh data not seen during training."""
    print("\n" + "=" * 60)
    print("STRESS TEST (fresh data, same projection space)")
    print("=" * 60)

    # Use SAME projections as training (same seed) but fresh SCM samples.
    # This simulates the real scenario: same frozen Transformer, new inputs.
    fresh_dataset = SCMDataset(
        scm=SyntheticSCM(noise_scale=scm.noise_scale, seed=99999),
        num_samples=n_samples,
        d_model=model.d_model,
        d_action=model.d_action,
        seed=training_seed,  # same projections as training
    )

    loader = torch.utils.data.DataLoader(fresh_dataset, batch_size=256)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for h, action, target in loader:
            h, action = h.to(device), action.to(device)
            delta = model(h, action)
            all_preds.append(delta.values.cpu())
            all_targets.append(target)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    dim_names = ["risk_shift", "goal_progress", "constraint_viol",
                 "resource_cost", "success_prob"]

    print(f"\n  {'Dimension':<22} {'Correlation':>12} {'Dir Acc':>10} "
          f"{'MAE':>10} {'RMSE':>10}")
    print(f"  {'-'*66}")

    for i, name in enumerate(dim_names):
        p, t = preds[:, i], targets[:, i]
        if t.std() > 1e-6 and p.std() > 1e-6:
            corr = torch.corrcoef(torch.stack([p, t]))[0, 1].item()
        else:
            corr = 0.0
        dir_acc = ((p.sign() == t.sign()).float().mean()).item()
        mae = (p - t).abs().mean().item()
        rmse = ((p - t) ** 2).mean().sqrt().item()
        print(f"  {name:<22} {corr:>12.4f} {dir_acc:>10.4f} "
              f"{mae:>10.4f} {rmse:>10.4f}")

    # Overall
    overall_corr = torch.corrcoef(
        torch.stack([preds.flatten(), targets.flatten()])
    )[0, 1].item()
    overall_mae = (preds - targets).abs().mean().item()
    print(f"\n  Overall correlation: {overall_corr:.4f}")
    print(f"  Overall MAE: {overall_mae:.4f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = checkpoint["args"]

    # Recreate model
    model = Causeway(
        d_model=model_args["d_model"],
        d_causal=model_args["d_causal"],
        d_action=model_args["d_action"],
        graph_layers=model_args["graph_layers"],
        propagation_steps=model_args["propagation_steps"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model loaded (epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.4f})")
    print(f"Parameters: {checkpoint['diagnostics']['total_parameters_human']}")

    # Create environment
    scm = SyntheticSCM(
        noise_scale=model_args["noise_scale"],
        seed=args.seed,
    )

    # Evaluation dataset (same projection as training for fair comparison)
    eval_dataset = SCMDataset(
        scm=scm,
        num_samples=2000,
        d_model=model_args["d_model"],
        d_action=model_args["d_action"],
        seed=model_args["seed"],  # same projections
    )

    # Run evaluations
    graph_recovery_analysis(model, scm)
    counterfactual_examples(model, eval_dataset, device, args.num_examples)
    stress_test(model, scm, device, training_seed=model_args["seed"])


if __name__ == "__main__":
    main()
