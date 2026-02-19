"""
Train Surgical CausewayLayer: Distillation + End-to-End Pairwise Ranking.

Phase 2 (Distillation):
    Adapts the oracle Causeway to extract structured deltas from mid-layer
    residual-stream features instead of final-layer pooled hidden states.
    Only layers 0..insertion_layer are needed (memory-efficient).

    Loss: L_delta_match + lambda_reg * (L_acyclicity + L_sparsity + L_edge_count + L_ortho)
          + lambda_gate * L_gate_open

Phase 3 (End-to-End):
    Fine-tunes CausewayLayer so its residual-stream modification improves
    the LM's pairwise decisions. Full model needed but Transformer is frozen.

    Loss: L_ranking + lambda_distill * L_delta_consistency + lambda_reg * L_causeway_reg

Usage:
    # Phase 2 only (distillation)
    python train_surgical.py --phase 2 --model gpt2 --domain confounded

    # Phase 3 only (end-to-end, requires Phase 2 checkpoint)
    python train_surgical.py --phase 3 --model gpt2 --domain confounded \\
        --surgical_checkpoint surgical_distill_gpt2.pt

    # Both phases
    python train_surgical.py --phase both --model gpt2 --domain confounded
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causeway.causeway_module import Causeway
from integration.causeway_layer import CausewayLayer
from integration.surgical_insertion import (
    insert_causeway_layer,
    freeze_transformer_weights,
    get_causeway_layer,
    detect_architecture,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Surgical CausewayLayer")
    p.add_argument("--phase", type=str, default="both",
                   choices=["2", "3", "both"])
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--domain", type=str, default="confounded",
                   choices=["deployment", "clinical", "confounded"])
    p.add_argument("--causeway_checkpoint", type=str, default=None,
                   help="Oracle Causeway checkpoint (Phase 1 output). Auto-named if not set.")
    p.add_argument("--surgical_checkpoint", type=str, default=None,
                   help="Phase 2 surgical checkpoint (for Phase 3 warm-start). Auto-named if not set.")
    p.add_argument("--cache", type=str, default=None,
                   help="Cached Transformer hidden states. Auto-named if not set.")
    p.add_argument("--layer_idx", type=int, default=None,
                   help="Insertion layer index. Defaults to midpoint.")
    p.add_argument("--d_pool", type=int, default=None,
                   help="Attention pooler bottleneck dim. Defaults to d_causal*4.")
    p.add_argument("--bottleneck_dim", type=int, default=256)
    p.add_argument("--gate_init", type=float, default=-5.0)

    # Phase 2 args
    p.add_argument("--p2_epochs", type=int, default=100)
    p.add_argument("--p2_freeze_epochs", type=int, default=30,
                   help="Epochs with Causeway core frozen, train only adapters.")
    p.add_argument("--p2_lr_frozen", type=float, default=1e-3)
    p.add_argument("--p2_lr_unfrozen", type=float, default=3e-4)
    p.add_argument("--p2_lambda_reg", type=float, default=0.1)
    p.add_argument("--p2_lambda_gate", type=float, default=0.1)

    # Phase 3 args
    p.add_argument("--p3_epochs", type=int, default=40)
    p.add_argument("--p3_lr", type=float, default=1e-4)
    p.add_argument("--p3_warmup_epochs", type=int, default=5)
    p.add_argument("--p3_lambda_distill", type=float, default=1.0,
                   help="Starting weight for delta consistency loss (decays to 0.1).")
    p.add_argument("--p3_lambda_reg", type=float, default=0.01)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default=".")

    return p.parse_args()


def quality_score(delta):
    """Scalar quality from structured delta."""
    return -delta[0] + delta[1] - delta[2] - delta[3] + delta[4]


# =====================================================================
# Phase 2: Distillation
# =====================================================================

def run_phase2(args, device):
    """Phase 2: Adapt oracle Causeway to mid-layer residual stream features."""
    print(f"\n{'='*70}")
    print("  PHASE 2: Distillation Adaptation")
    print(f"{'='*70}")

    model_short = args.model.split("/")[-1]

    # Auto-name paths
    if args.causeway_checkpoint is None:
        if args.domain == "deployment":
            args.causeway_checkpoint = f"causeway_{model_short}.pt"
        else:
            args.causeway_checkpoint = f"causeway_{args.domain}_{model_short}.pt"
    if args.cache is None:
        args.cache = f"cache_{args.domain}_{model_short}_{args.num_samples}_v2.pt"

    distill_save = os.path.join(
        args.save_dir, f"surgical_distill_{args.domain}_{model_short}.pt")

    # === Load oracle Causeway ===
    print(f"Loading oracle Causeway from {args.causeway_checkpoint}...")
    ckpt = torch.load(args.causeway_checkpoint, map_location=device, weights_only=False)
    cw_args = ckpt["args"]
    d_model = ckpt["d_model"]
    d_causal = cw_args["d_causal"]
    d_action = cw_args.get("d_action", d_model)

    # Oracle Causeway (frozen, for generating targets)
    oracle = Causeway(
        d_model=d_model, d_causal=d_causal, d_action=d_action,
        graph_layers=cw_args.get("graph_layers", 2),
        propagation_steps=cw_args.get("propagation_steps", 3),
    ).to(device)
    oracle.load_state_dict(ckpt["model_state_dict"])
    oracle.eval()
    for p in oracle.parameters():
        p.requires_grad = False

    # Surgical Causeway (warm-started from oracle)
    surgical_cw = Causeway(
        d_model=d_model, d_causal=d_causal, d_action=d_action,
        graph_layers=cw_args.get("graph_layers", 2),
        propagation_steps=cw_args.get("propagation_steps", 3),
    ).to(device)
    surgical_cw.load_state_dict(ckpt["model_state_dict"])

    # === Load Transformer (only need layers 0..insertion_layer) ===
    print(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm_model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True).to(device)

    architecture = detect_architecture(lm_model)

    # Determine insertion point
    if architecture == "gpt2":
        n_layers = len(lm_model.transformer.h)
    else:
        n_layers = len(lm_model.model.layers)
    layer_idx = args.layer_idx if args.layer_idx is not None else n_layers // 2
    print(f"Insertion point: layer {layer_idx} of {n_layers}")

    # Insert CausewayLayer
    cl = insert_causeway_layer(
        lm_model, surgical_cw, layer_idx=layer_idx,
        architecture=architecture, d_pool=args.d_pool,
        bottleneck_dim=args.bottleneck_dim,
        gate_init=args.gate_init, device=device,
    )

    # Freeze everything except CausewayLayer
    freeze_transformer_weights(lm_model, architecture)

    # === Load cached dataset ===
    print(f"Loading dataset from {args.cache}...")
    data = torch.load(args.cache, weights_only=False, map_location="cpu")
    h_all = data["h"].float()      # (N, d_model) -- final-layer hidden states
    a_all = data["actions"].float()  # (N, d_model)
    targets_all = data["targets"]   # (N, 5)

    # Generate oracle delta targets
    print("Computing oracle delta targets...")
    oracle_deltas = []
    oracle_z_refined = []
    oracle_z_cf = []
    with torch.no_grad():
        for i in range(0, len(h_all), 256):
            h_batch = h_all[i:i+256].to(device)
            a_batch = a_all[i:i+256].to(device)
            internals = oracle.forward_with_internals(h_batch, a_batch)
            oracle_deltas.append(internals['delta'].values.cpu())
            oracle_z_refined.append(internals['z_refined'].cpu())
            oracle_z_cf.append(internals['z_counterfactual'].cpu())
    oracle_deltas = torch.cat(oracle_deltas)
    oracle_z_refined_all = torch.cat(oracle_z_refined)
    oracle_z_cf_all = torch.cat(oracle_z_cf)
    print(f"Oracle deltas shape: {oracle_deltas.shape}")

    # We need to generate mid-layer hidden states by running text through
    # layers 0..layer_idx. Use cached h as state embedding + a as action embedding
    # to form prompts and extract mid-layer features.
    #
    # For distillation, we directly use the cached embeddings as input to the
    # CausewayLayer's pooler (simulating mid-layer features). This is approximate
    # but avoids re-encoding all text through the Transformer.
    #
    # The pooler will learn to extract state/action from these features, and
    # the write-back will learn to produce deltas matching the oracle.

    # Create dataset: h_all acts as mid-layer features (seq_len=1 broadened)
    # We create a pseudo-sequence by concatenating h (state) and a (action)
    # as two "tokens" in the sequence dimension
    h_seq = torch.stack([h_all, a_all], dim=1)  # (N, 2, d_model)

    dataset = TensorDataset(h_seq, a_all, oracle_deltas, oracle_z_refined_all, oracle_z_cf_all)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    print(f"Dataset: {n_train} train, {n_val} val")

    # === Training ===
    # Phase 2a: freeze Causeway core, train only adapters
    # Phase 2b: unfreeze all CausewayLayer

    def set_causeway_core_frozen(cl, frozen):
        """Freeze/unfreeze the Causeway core inside the CausewayLayer."""
        for p in cl.causeway.parameters():
            p.requires_grad = not frozen
        # Always keep pooler, write_back, gate trainable
        for p in cl.pooler.parameters():
            p.requires_grad = True
        for p in cl.write_back.parameters():
            p.requires_grad = True
        cl.gate_logit.requires_grad = True

    # Start with frozen core
    set_causeway_core_frozen(cl, frozen=True)

    trainable_params = [p for p in cl.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.p2_lr_frozen, weight_decay=1e-5)

    best_val_loss = float("inf")
    print(f"\nPhase 2: Training for {args.p2_epochs} epochs "
          f"(frozen core: 0-{args.p2_freeze_epochs}, unfrozen: {args.p2_freeze_epochs}-{args.p2_epochs})\n")

    for epoch in range(args.p2_epochs):
        # Transition: unfreeze Causeway core at p2_freeze_epochs
        if epoch == args.p2_freeze_epochs:
            print(f"\n--- Unfreezing Causeway core at epoch {epoch} ---")
            set_causeway_core_frozen(cl, frozen=False)
            trainable_params = [p for p in cl.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params, lr=args.p2_lr_unfrozen, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.p2_epochs - args.p2_freeze_epochs,
                eta_min=args.p2_lr_unfrozen * 0.01)

        cl.train()
        train_loss_total = 0
        n_batches = 0

        for h_seq_batch, a_batch, delta_target, z_ref_target, z_cf_target in train_loader:
            h_seq_batch = h_seq_batch.to(device)
            a_batch = a_batch.to(device)
            delta_target = delta_target.to(device)
            z_ref_target = z_ref_target.to(device)
            z_cf_target = z_cf_target.to(device)

            # Run CausewayLayer with internals
            modified, internals = cl._apply_causeway_with_internals(h_seq_batch)

            # Delta match loss: MSE between inline delta and oracle targets
            # We use the write-back output projected through delta_predictor
            # Since we have z_refined and z_counterfactual from internals,
            # run delta_predictor to get inline deltas
            inline_delta = cl.causeway.delta_predictor(
                internals['z_refined'], internals['z_counterfactual'])
            l_delta_match = F.mse_loss(inline_delta.values, delta_target)

            # Causal state match
            l_z_match = (F.mse_loss(internals['z_refined'], z_ref_target)
                         + F.mse_loss(internals['z_counterfactual'], z_cf_target))

            # Regularization
            reg = cl.get_regularization_losses()
            l_reg = (reg['acyclicity'] + reg['sparsity']
                     + reg['edge_count'] + reg['orthogonality'])

            # Gate opening loss: encourage gate to open
            l_gate = -torch.log(cl.gate + 1e-8)

            loss = (l_delta_match
                    + 0.5 * l_z_match
                    + args.p2_lambda_reg * l_reg
                    + args.p2_lambda_gate * l_gate)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            train_loss_total += loss.item()
            n_batches += 1

        if epoch >= args.p2_freeze_epochs:
            scheduler.step()

        # Validation
        cl.eval()
        val_loss = 0
        val_delta_corr = []
        n_val_batches = 0

        with torch.no_grad():
            for h_seq_batch, a_batch, delta_target, z_ref_target, z_cf_target in val_loader:
                h_seq_batch = h_seq_batch.to(device)
                delta_target = delta_target.to(device)
                z_ref_target = z_ref_target.to(device)
                z_cf_target = z_cf_target.to(device)

                modified, internals = cl._apply_causeway_with_internals(h_seq_batch)
                inline_delta = cl.causeway.delta_predictor(
                    internals['z_refined'], internals['z_counterfactual'])

                l_delta = F.mse_loss(inline_delta.values, delta_target)
                val_loss += l_delta.item()
                n_val_batches += 1

                # Correlation
                pred = inline_delta.values.cpu()
                tgt = delta_target.cpu()
                for dim in range(pred.shape[1]):
                    if tgt[:, dim].std() > 1e-6 and pred[:, dim].std() > 1e-6:
                        c = torch.corrcoef(torch.stack([pred[:, dim], tgt[:, dim]]))[0, 1].item()
                        val_delta_corr.append(c)

        val_loss /= max(n_val_batches, 1)
        avg_corr = np.mean(val_delta_corr) if val_delta_corr else 0.0

        if epoch % 10 == 0 or epoch == args.p2_epochs - 1:
            print(f"Epoch {epoch:3d}/{args.p2_epochs}  "
                  f"train_loss={train_loss_total / max(n_batches, 1):.4f}  "
                  f"val_delta_mse={val_loss:.4f}  "
                  f"val_corr={avg_corr:.4f}  "
                  f"gate={cl.gate.item():.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'causeway_layer_state_dict': cl.state_dict(),
                'oracle_args': cw_args,
                'd_model': d_model,
                'd_causal': d_causal,
                'd_action': d_action,
                'layer_idx': layer_idx,
                'architecture': architecture,
                'bottleneck_dim': args.bottleneck_dim,
                'd_pool': args.d_pool,
                'gate_init': args.gate_init,
                'val_loss': val_loss,
                'val_corr': avg_corr,
                'epoch': epoch,
                'domain': args.domain,
                'model': args.model,
            }, distill_save)

    print(f"\nPhase 2 complete. Best val MSE: {best_val_loss:.4f}")
    print(f"Saved: {distill_save}")
    return distill_save


# =====================================================================
# Phase 3: End-to-End Pairwise Ranking
# =====================================================================

def run_phase3(args, device, distill_checkpoint=None):
    """Phase 3: Fine-tune CausewayLayer for pairwise ranking."""
    print(f"\n{'='*70}")
    print("  PHASE 3: End-to-End Pairwise Ranking")
    print(f"{'='*70}")

    model_short = args.model.split("/")[-1]

    # Auto-name paths
    if distill_checkpoint is None:
        if args.surgical_checkpoint:
            distill_checkpoint = args.surgical_checkpoint
        else:
            distill_checkpoint = os.path.join(
                args.save_dir, f"surgical_distill_{args.domain}_{model_short}.pt")

    e2e_save = os.path.join(
        args.save_dir, f"surgical_e2e_{args.domain}_{model_short}.pt")

    if args.cache is None:
        args.cache = f"cache_{args.domain}_{model_short}_{args.num_samples}_v2.pt"

    # === Load Phase 2 checkpoint ===
    print(f"Loading Phase 2 checkpoint from {distill_checkpoint}...")
    ckpt = torch.load(distill_checkpoint, map_location=device, weights_only=False)
    d_model = ckpt['d_model']
    d_causal = ckpt['d_causal']
    d_action = ckpt['d_action']
    layer_idx = ckpt['layer_idx']
    cw_args = ckpt['oracle_args']

    # === Load full Transformer ===
    print(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm_model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True).to(device)

    architecture = detect_architecture(lm_model)

    # Create Causeway and CausewayLayer
    surgical_cw = Causeway(
        d_model=d_model, d_causal=d_causal, d_action=d_action,
        graph_layers=cw_args.get("graph_layers", 2),
        propagation_steps=cw_args.get("propagation_steps", 3),
    ).to(device)

    cl = insert_causeway_layer(
        lm_model, surgical_cw, layer_idx=layer_idx,
        architecture=architecture,
        d_pool=ckpt.get('d_pool'),
        bottleneck_dim=ckpt.get('bottleneck_dim', 256),
        gate_init=ckpt.get('gate_init', -5.0),
        device=device,
    )

    # Load Phase 2 weights
    cl.load_state_dict(ckpt['causeway_layer_state_dict'])
    print(f"Loaded Phase 2 weights (val_corr={ckpt.get('val_corr', 'N/A')})")

    # Cache Phase 2 delta for consistency loss
    phase2_gate = cl.gate.item()
    print(f"Phase 2 gate value: {phase2_gate:.4f}")

    # Freeze Transformer, train only CausewayLayer
    freeze_transformer_weights(lm_model, architecture)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(lm_model, 'gradient_checkpointing_enable'):
        lm_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    # === Load dataset ===
    print(f"Loading dataset from {args.cache}...")
    data = torch.load(args.cache, weights_only=False, map_location="cpu")
    h_all = data["h"].float()
    a_all = data["actions"].float()
    targets_all = data["targets"]
    q_all = torch.tensor([quality_score(targets_all[i].numpy())
                          for i in range(len(targets_all))])

    # Ensure even for pairing
    n_samples = len(h_all) - (len(h_all) % 2)
    h_all = h_all[:n_samples]
    a_all = a_all[:n_samples]
    targets_all = targets_all[:n_samples]
    q_all = q_all[:n_samples]

    dataset = TensorDataset(h_all, a_all, targets_all, q_all)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    if n_train % 2 != 0:
        n_train -= 1
        n_val += 1
    if n_val % 2 != 0:
        n_val -= 1
        n_train += 1
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, drop_last=True)

    print(f"Dataset: {n_train} train, {n_val} val ({n_train // 2} pairs)")

    # === Setup probe tokens ===
    token_A = tokenizer.encode(" A")[0]
    token_B = tokenizer.encode(" B")[0]

    if args.domain == "confounded":
        prompt_text = "The better option is Option"
    elif args.domain == "clinical":
        prompt_text = "The safer treatment option is Option"
    else:
        prompt_text = "The safer deployment option is Option"

    prompt_ids = tokenizer.encode(prompt_text)
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    # Get embeddings layer
    if architecture == "gpt2":
        embed_fn = lm_model.transformer.wte
    else:
        embed_fn = lm_model.model.embed_tokens

    prompt_embeds = embed_fn(prompt_tensor)  # (1, seq, d_model)
    print(f"Prompt: '{prompt_text}' ({len(prompt_ids)} tokens)")
    print(f"Probe: ' A' (id={token_A}) vs ' B' (id={token_B})")

    # === Training ===
    trainable_params = [p for p in cl.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.p3_lr, weight_decay=1e-5)

    # Warmup + cosine
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=args.p3_warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.p3_epochs - args.p3_warmup_epochs,
        eta_min=args.p3_lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.p3_warmup_epochs])

    rng = np.random.RandomState(42)
    best_val_acc = 0
    print(f"\nPhase 3: Training for {args.p3_epochs} epochs\n")

    for epoch in range(args.p3_epochs):
        # Lambda_distill decay: 1.0 -> 0.1 over training
        progress = epoch / max(args.p3_epochs - 1, 1)
        lambda_distill = args.p3_lambda_distill * (1.0 - 0.9 * progress)

        cl.train()
        # Ensure other layers stay in eval
        lm_model.eval()
        # Re-enable training on CausewayLayer
        cl.train()

        train_loss_total = 0
        train_correct = 0
        train_total = 0

        for h, a, targets, q in train_loader:
            h, a, targets, q = (
                h.to(device), a.to(device), targets.to(device), q.to(device))
            bs = h.shape[0]
            n_pairs = bs // 2

            if n_pairs == 0:
                continue

            # Form pairs
            h1 = h[0::2][:n_pairs]
            a1 = a[0::2][:n_pairs]
            q1 = q[0::2][:n_pairs]
            h2 = h[1::2][:n_pairs]
            a2 = a[1::2][:n_pairs]
            q2 = q[1::2][:n_pairs]

            # Random A/B assignment
            swap = torch.tensor(rng.random(n_pairs) > 0.5, device=device)
            better_is_A_no_swap = q1 > q2
            better_is_A_swap = q2 > q1
            better_is_A = torch.where(swap, better_is_A_swap, better_is_A_no_swap)
            target_labels = (~better_is_A).long()

            # Build input embedding:
            # We concatenate h1 (state) + better_action as a 2-token "context"
            # then append the prompt
            a_opt_a = torch.where(swap.unsqueeze(-1), a2, a1)
            a_opt_b = torch.where(swap.unsqueeze(-1), a1, a2)

            # Create pseudo input: embed the prompt text with state context
            # The CausewayLayer in the middle of the model will process these
            prompt_batch = prompt_embeds.expand(n_pairs, -1, -1)

            # Create state+action context as embeddings
            # Use h1 as state embedding, a_opt_a and a_opt_b as action embeddings
            # Format: [state_embed, action_a_embed, action_b_embed, prompt_embeds]
            context_embeds = torch.stack([h1, a_opt_a, a_opt_b], dim=1)  # (n_pairs, 3, d_model)
            input_embeds = torch.cat([context_embeds, prompt_batch], dim=1)

            # Forward through the full model (CausewayLayer is in the middle)
            logits = lm_model(inputs_embeds=input_embeds).logits[:, -1, :]

            # Ranking loss: CE on A/B tokens
            probe_logits = logits[:, [token_A, token_B]]
            l_ranking = F.cross_entropy(probe_logits, target_labels)

            # Regularization
            reg = cl.get_regularization_losses()
            l_reg = (reg['acyclicity'] + reg['sparsity']
                     + reg['edge_count'] + reg['orthogonality'])

            loss = l_ranking + args.p3_lambda_reg * l_reg

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            train_loss_total += loss.item() * n_pairs
            train_correct += (probe_logits.argmax(1) == target_labels).sum().item()
            train_total += n_pairs

        scheduler.step()

        # Validation
        cl.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for h, a, targets, q in val_loader:
                h, a, targets, q = (
                    h.to(device), a.to(device), targets.to(device), q.to(device))
                bs = h.shape[0]
                n_pairs = bs // 2
                if n_pairs == 0:
                    continue

                h1 = h[0::2][:n_pairs]
                a1 = a[0::2][:n_pairs]
                q1 = q[0::2][:n_pairs]
                h2 = h[1::2][:n_pairs]
                a2 = a[1::2][:n_pairs]
                q2 = q[1::2][:n_pairs]

                swap = torch.tensor(rng.random(n_pairs) > 0.5, device=device)
                better_is_A_no_swap = q1 > q2
                better_is_A_swap = q2 > q1
                better_is_A = torch.where(swap, better_is_A_swap, better_is_A_no_swap)
                target_labels = (~better_is_A).long()

                a_opt_a = torch.where(swap.unsqueeze(-1), a2, a1)
                a_opt_b = torch.where(swap.unsqueeze(-1), a1, a2)

                prompt_batch = prompt_embeds.expand(n_pairs, -1, -1)
                context_embeds = torch.stack([h1, a_opt_a, a_opt_b], dim=1)
                input_embeds = torch.cat([context_embeds, prompt_batch], dim=1)

                logits = lm_model(inputs_embeds=input_embeds).logits[:, -1, :]
                probe_logits = logits[:, [token_A, token_B]]

                val_correct += (probe_logits.argmax(1) == target_labels).sum().item()
                val_total += n_pairs

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if epoch % 5 == 0 or epoch == args.p3_epochs - 1:
            print(f"Epoch {epoch:3d}/{args.p3_epochs}  "
                  f"train_loss={train_loss_total / max(train_total, 1):.4f}  "
                  f"train_acc={train_acc:.3f}  "
                  f"val_acc={val_acc:.3f}  "
                  f"gate={cl.gate.item():.4f}  "
                  f"lambda_d={lambda_distill:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'causeway_layer_state_dict': cl.state_dict(),
                'oracle_args': cw_args,
                'd_model': d_model,
                'd_causal': d_causal,
                'd_action': d_action,
                'layer_idx': layer_idx,
                'architecture': architecture,
                'bottleneck_dim': ckpt.get('bottleneck_dim', 256),
                'd_pool': ckpt.get('d_pool'),
                'gate_init': ckpt.get('gate_init', -5.0),
                'val_acc': val_acc,
                'epoch': epoch,
                'domain': args.domain,
                'model': args.model,
                'phase': 3,
            }, e2e_save)

    print(f"\nPhase 3 complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Saved: {e2e_save}")
    return e2e_save


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"{'='*70}")
    print(f"  SURGICAL CAUSEWAY TRAINING")
    print(f"  Model: {args.model}  Domain: {args.domain}  Phase: {args.phase}")
    print(f"{'='*70}")
    print(f"Device: {device}")

    distill_ckpt = None

    if args.phase in ("2", "both"):
        distill_ckpt = run_phase2(args, device)

    if args.phase in ("3", "both"):
        run_phase3(args, device, distill_checkpoint=distill_ckpt)


if __name__ == "__main__":
    main()
