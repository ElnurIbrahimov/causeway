"""
Compare Causeway results across backbone models.

Loads checkpoint files from Falcon-H1R-7B and Mistral-7B-v0.3 runs
and prints a side-by-side comparison of causal alignment metrics.

Usage:
    python compare_results.py --falcon_dir /workspace/results/falcon --mistral_dir .
    python compare_results.py  # uses defaults
"""

import argparse
import os
import torch


DOMAINS = ["deployment", "clinical", "confounded"]
DIMS = ["risk_shift", "goal_progress", "constraint_viol", "resource_cost", "success_prob"]


def load_ckpt(path):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def print_model_summary(name, ckpts):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    for domain in DOMAINS:
        ckpt = ckpts.get(domain)
        if ckpt is None:
            print(f"\n  [{domain}] -- NOT FOUND")
            continue

        results = ckpt.get("val_results", {})
        corrs = results.get("correlations", {})
        dir_accs = results.get("directional_accuracy", {})
        overall_corr = results.get("overall_corr", float("nan"))
        overall_mae = results.get("overall_mae", float("nan"))
        epoch = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", float("nan"))
        backbone = ckpt.get("backbone", "unknown")

        print(f"\n  [{domain.upper()}] — epoch {epoch}, val_loss={val_loss:.4f}")
        print(f"  Backbone: {backbone}")
        print(f"  Overall correlation: {overall_corr:.4f} | MAE: {overall_mae:.4f}")
        print(f"  {'Dimension':<22} {'Corr':>8} {'DirAcc':>8}")
        print(f"  {'-'*40}")
        for dim in DIMS:
            c = corrs.get(dim, float("nan"))
            a = dir_accs.get(dim, float("nan"))
            print(f"  {dim:<22} {c:>8.4f} {a:>8.4f}")


def print_head_to_head(falcon_ckpts, mistral_ckpts):
    print(f"\n{'='*60}")
    print(f"  HEAD-TO-HEAD: Falcon-H1R-7B vs Mistral-7B-v0.3")
    print(f"{'='*60}")
    print(f"\n  {'Domain':<14} {'Metric':<22} {'Falcon':>10} {'Mistral':>10} {'Winner':>10}")
    print(f"  {'-'*60}")

    for domain in DOMAINS:
        f_ckpt = falcon_ckpts.get(domain)
        m_ckpt = mistral_ckpts.get(domain)

        if f_ckpt is None and m_ckpt is None:
            print(f"  {domain:<14} -- both missing")
            continue

        f_res = f_ckpt.get("val_results", {}) if f_ckpt else {}
        m_res = m_ckpt.get("val_results", {}) if m_ckpt else {}

        f_corr = f_res.get("overall_corr", float("nan"))
        m_corr = m_res.get("overall_corr", float("nan"))
        f_mae = f_res.get("overall_mae", float("nan"))
        m_mae = m_res.get("overall_mae", float("nan"))

        def winner_corr(f, m):
            if f != f: return "N/A"
            if m != m: return "N/A"
            return "Falcon" if f > m else ("Mistral" if m > f else "Tie")

        def winner_mae(f, m):
            if f != f: return "N/A"
            if m != m: return "N/A"
            return "Falcon" if f < m else ("Mistral" if m < f else "Tie")

        print(f"  {domain:<14} {'overall_corr':<22} {f_corr:>10.4f} {m_corr:>10.4f} {winner_corr(f_corr, m_corr):>10}")
        print(f"  {domain:<14} {'overall_mae':<22} {f_mae:>10.4f} {m_mae:>10.4f} {winner_mae(f_mae, m_mae):>10}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--falcon_dir", type=str, default="/workspace/results/falcon",
                   help="Directory containing Falcon checkpoint .pt files")
    p.add_argument("--mistral_dir", type=str, default=".",
                   help="Directory containing Mistral checkpoint .pt files")
    args = p.parse_args()

    # Load Falcon checkpoints
    falcon_ckpts = {}
    for domain in DOMAINS:
        path = os.path.join(args.falcon_dir, f"causeway_falcon_{domain}.pt")
        ckpt = load_ckpt(path)
        if ckpt:
            falcon_ckpts[domain] = ckpt

    # Load Mistral checkpoints (existing naming convention)
    mistral_ckpts = {}
    mistral_names = {
        "deployment": "causeway_Mistral-7B-v0.3.pt",
        "clinical":   "causeway_clinical_Mistral-7B-v0.3.pt",
        "confounded": "causeway_confounded_Mistral-7B-v0.3.pt",
    }
    for domain, fname in mistral_names.items():
        path = os.path.join(args.mistral_dir, fname)
        ckpt = load_ckpt(path)
        if ckpt:
            mistral_ckpts[domain] = ckpt

    if not falcon_ckpts and not mistral_ckpts:
        print("No checkpoints found. Run experiments first.")
        return

    print_model_summary("Falcon-H1R-7B (Hybrid SSM+Attention)", falcon_ckpts)
    print_model_summary("Mistral-7B-v0.3 (Pure Transformer)", mistral_ckpts)
    print_head_to_head(falcon_ckpts, mistral_ckpts)

    print(f"\n\nKey question: Does Falcon's SSM-hybrid architecture encode")
    print(f"causal structure differently than a pure transformer?")
    print(f"Higher correlation = better causal alignment in hidden states.")


if __name__ == "__main__":
    main()
