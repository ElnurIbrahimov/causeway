"""
Train TransformerBridge: Close the Loop with Pairwise Ranking.

Trains the bridge to convert Causeway's causal state into prefix tokens
that steer a frozen LM toward correct pairwise action rankings.

Supports:
    - Multiple domains: deployment (software), clinical (treatment), confounded (neutral causal)
    - Multiple bridge versions: v1 (TransformerBridge), v2 (TransformerBridgeV2)
    - Any HuggingFace causal LM (GPT-2, TinyLlama, LLaMA-3.2, etc.)

Frozen: LM backbone, Causeway
Trainable: Bridge projection (~2-4M params depending on bridge version)

Training objective: pairwise ranking
    Given a shared state and two candidate actions, the bridge must steer
    the LM to predict which action leads to a better outcome (Option A vs B).
    A/B ordering is randomized per pair to prevent position bias.

After training, runs closed-loop pairwise evaluation comparing:
    1. Baseline LM (random guess)
    2. Causeway standalone ranking
    3. LM + trained bridge

Usage:
    python train_bridge.py
    python train_bridge.py --model gpt2 --domain deployment --bridge_version v2
    python train_bridge.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --domain clinical
    python train_bridge.py --bridge_version v1 --epochs 50
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--domain", type=str, default="deployment",
                   choices=["deployment", "clinical", "confounded"])
    p.add_argument("--bridge_version", type=str, default="v2",
                   choices=["v1", "v2"])
    p.add_argument("--causeway_checkpoint", default=None,
                   help="Auto-named if not set")
    p.add_argument("--cache", default=None,
                   help="Auto-named if not set")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_prefix_tokens", type=int, default=4)
    p.add_argument("--save_path", default=None)
    return p.parse_args()


def quality_score(delta):
    return -delta[0] + delta[1] - delta[2] - delta[3] + delta[4]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # --- Auto-naming ---
    model_short = args.model.split("/")[-1]
    if args.causeway_checkpoint is None:
        if args.domain == "deployment":
            args.causeway_checkpoint = f"causeway_{model_short}.pt"
        else:
            args.causeway_checkpoint = f"causeway_{args.domain}_{model_short}.pt"
    if args.cache is None:
        args.cache = f"cache_{args.domain}_{model_short}_50000_v2.pt"
    if args.save_path is None:
        args.save_path = f"bridge_{args.bridge_version}_{args.domain}_{model_short}.pt"

    print("=" * 60)
    print(f"  BRIDGE TRAINING: Pairwise Ranking ({args.bridge_version})")
    print(f"  Model: {args.model}  Domain: {args.domain}")
    print("=" * 60)
    print(f"Device: {device}")

    # === Load LM (frozen) ===
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm_model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    for p in lm_model.parameters():
        p.requires_grad = False

    d_model_lm = lm_model.config.hidden_size

    # === Load Causeway (frozen) ===
    print(f"Loading Causeway from {args.causeway_checkpoint}...")
    ckpt = torch.load(args.causeway_checkpoint, map_location=device, weights_only=False)
    cw_args = ckpt["args"]
    d_causal = cw_args["d_causal"]
    d_model = ckpt["d_model"]

    causeway = Causeway(
        d_model=d_model,
        d_causal=d_causal,
        d_action=cw_args.get("d_action", d_model),
        graph_layers=cw_args.get("graph_layers", 2),
        propagation_steps=cw_args.get("propagation_steps", 3),
    ).to(device)
    causeway.load_state_dict(ckpt["model_state_dict"])
    causeway.eval()
    for p in causeway.parameters():
        p.requires_grad = False

    # === Create Bridge (trainable projection only) ===
    if args.bridge_version == "v2":
        from integration.transformer_bridge import TransformerBridgeV2
        bridge = TransformerBridgeV2(
            causeway=causeway,
            d_model=d_model,
            d_causal=d_causal,
            n_prefix_tokens=args.n_prefix_tokens,
        ).to(device)
    else:
        from integration.transformer_bridge import TransformerBridge
        bridge = TransformerBridge(
            causeway=causeway,
            d_model=d_model,
            n_prefix_tokens=args.n_prefix_tokens,
        ).to(device)

    # Freeze Causeway inside bridge, train only projection layers
    bridge.causeway.requires_grad_(False)
    if args.bridge_version == "v2":
        trainable_params = (
            list(bridge.prefix_generator.parameters())
            + [bridge.prefix_positions]
            + list(bridge.norm.parameters())
        )
    else:
        trainable_params = (
            list(bridge.delta_to_prefix.parameters())
            + [bridge.prefix_positions]
            + list(bridge.norm.parameters())
        )
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Bridge version: {args.bridge_version}")
    print(f"Bridge trainable params: {n_trainable / 1e6:.3f}M")

    # === Load cached dataset ===
    print(f"Loading dataset from {args.cache}...")
    data = torch.load(args.cache, weights_only=False, map_location="cpu")
    h_all = data["h"].float()
    a_all = data["actions"].float()
    targets_all = data["targets"]

    # Compute quality scores for all samples
    q_all = torch.tensor([quality_score(targets_all[i]) for i in range(len(targets_all))])

    # Ensure even number of samples for pairing
    n_samples = len(h_all) - (len(h_all) % 2)
    h_all = h_all[:n_samples]
    a_all = a_all[:n_samples]
    targets_all = targets_all[:n_samples]
    q_all = q_all[:n_samples]

    print(f"Samples: {n_samples} (forming {n_samples // 2} pairs)")

    # Split
    dataset = TensorDataset(h_all, a_all, targets_all, q_all)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    # Ensure even splits
    if n_train % 2 != 0:
        n_train -= 1
        n_val += 1
    if n_val % 2 != 0:
        n_val -= 1
        n_train += 1
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, drop_last=True)

    # === Probe setup ===
    token_A = tokenizer.encode(" A")[0]
    token_B = tokenizer.encode(" B")[0]

    prompt_text = "Which option leads to a better outcome? Option"
    prompt_ids = tokenizer.encode(prompt_text)
    prompt_tensor = torch.tensor([prompt_ids], device=device)
    prompt_embeds = lm_model.get_input_embeddings()(prompt_tensor)  # (1, seq, d_model)

    print(f"Prompt: '{prompt_text}' ({len(prompt_ids)} tokens)")
    print(f"Probe: ' A' (id={token_A}) vs ' B' (id={token_B})")

    # === Training ===
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_acc = 0
    rng = np.random.RandomState(42)
    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        bridge.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for h, a, targets, q in train_loader:
            h, a, targets, q = (
                h.to(device), a.to(device), targets.to(device), q.to(device)
            )
            bs = h.shape[0]

            # Form pairs from consecutive samples in the batch
            n_pairs = bs // 2
            if n_pairs == 0:
                continue

            h1 = h[0::2][:n_pairs]
            a1 = a[0::2][:n_pairs]
            q1 = q[0::2][:n_pairs]

            h2 = h[1::2][:n_pairs]
            a2 = a[1::2][:n_pairs]
            q2 = q[1::2][:n_pairs]

            # Use h1 as shared state for both actions
            # Randomly assign actions to Option A/B to prevent position bias
            swap = torch.tensor(rng.random(n_pairs) > 0.5, device=device)

            # Determine which option is better based on ground-truth quality
            # When not swapped: a1=OptionA, a2=OptionB => better_is_A if q1>q2
            # When swapped:     a2=OptionA, a1=OptionB => better_is_A if q2>q1
            better_is_A_no_swap = q1 > q2   # (n_pairs,)
            better_is_A_swap = q2 > q1       # (n_pairs,)
            better_is_A = torch.where(swap, better_is_A_swap, better_is_A_no_swap)

            # Target: 0 = "A" is better, 1 = "B" is better
            target_labels = (~better_is_A).long()

            # Select the "better" action for prefix generation
            # The bridge generates prefix from the better action's causal state
            # to steer the LM toward the correct answer
            a_opt_a = torch.where(swap.unsqueeze(-1), a2, a1)
            a_opt_b = torch.where(swap.unsqueeze(-1), a1, a2)

            # Generate prefix from the better action
            better_a = torch.where(better_is_A.unsqueeze(-1), a_opt_a, a_opt_b)
            prefix = bridge.get_causal_prefix(h1, better_a)  # (n_pairs, n_prefix, d_model)

            # Prompt
            prompt_batch = prompt_embeds.expand(n_pairs, -1, -1)
            augmented = torch.cat([prefix, prompt_batch], dim=1)

            # LM forward (frozen weights, gradient flows through prefix)
            logits = lm_model(inputs_embeds=augmented).logits[:, -1, :]

            # CE on [A, B] tokens
            probe_logits = logits[:, [token_A, token_B]]  # (n_pairs, 2)
            loss = F.cross_entropy(probe_logits, target_labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            train_loss += loss.item() * n_pairs
            train_correct += (probe_logits.argmax(1) == target_labels).sum().item()
            train_total += n_pairs

        scheduler.step()

        # Validation
        bridge.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for h, a, targets, q in val_loader:
                h, a, targets, q = (
                    h.to(device), a.to(device), targets.to(device), q.to(device)
                )
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
                better_a = torch.where(better_is_A.unsqueeze(-1), a_opt_a, a_opt_b)
                prefix = bridge.get_causal_prefix(h1, better_a)

                prompt_batch = prompt_embeds.expand(n_pairs, -1, -1)
                augmented = torch.cat([prefix, prompt_batch], dim=1)

                logits = lm_model(inputs_embeds=augmented).logits[:, -1, :]
                probe_logits = logits[:, [token_A, token_B]]

                val_loss += F.cross_entropy(probe_logits, target_labels).item() * n_pairs
                val_correct += (probe_logits.argmax(1) == target_labels).sum().item()
                val_total += n_pairs

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"train_loss={train_loss / max(train_total, 1):.4f}  "
                f"train_acc={train_acc:.3f}  "
                f"val_acc={val_acc:.3f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dict = {
                "bridge_version": args.bridge_version,
                "val_acc": val_acc,
                "epoch": epoch,
                "domain": args.domain,
                "model": args.model,
                "d_causal": d_causal,
            }
            if args.bridge_version == "v2":
                save_dict["prefix_generator"] = bridge.prefix_generator.state_dict()
            else:
                save_dict["delta_to_prefix"] = bridge.delta_to_prefix.state_dict()
            save_dict["prefix_positions"] = bridge.prefix_positions.data
            save_dict["norm"] = bridge.norm.state_dict()
            torch.save(save_dict, args.save_path)

    # === Load best and evaluate ===
    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    bridge_ckpt = torch.load(args.save_path, weights_only=False, map_location=device)
    if args.bridge_version == "v2":
        bridge.prefix_generator.load_state_dict(bridge_ckpt["prefix_generator"])
    else:
        bridge.delta_to_prefix.load_state_dict(bridge_ckpt["delta_to_prefix"])
    bridge.prefix_positions.data = bridge_ckpt["prefix_positions"]
    bridge.norm.load_state_dict(bridge_ckpt["norm"])
    bridge.eval()

    # ================================================================
    # EVAL: Pairwise Action Selection (closed loop)
    # ================================================================
    print("\n" + "=" * 60)
    print("  EVAL: Pairwise Action Selection (closed loop)")
    print(f"  Domain: {args.domain}  Bridge: {args.bridge_version}")
    print("=" * 60)

    from transformers import AutoModel

    lm_enc = AutoModel.from_pretrained(args.model).to(device).eval()
    for p in lm_enc.parameters():
        p.requires_grad = False

    def encode_text(text):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            out = lm_enc(**inputs)
            seq_len = inputs.attention_mask.sum(dim=1) - 1
            h = out.last_hidden_state[0, seq_len[0]]
        return h.unsqueeze(0)

    # Domain-aware imports for eval
    if args.domain == "clinical":
        from environments.clinical_scm import ClinicalSCM as SCMClass
        from environments.text_clinical import state_to_text, action_to_text
    elif args.domain == "confounded":
        from environments.confounded_scm import ConfoundedSCM as SCMClass
        from environments.text_confounded import state_to_text, action_to_text
    else:
        from environments.synthetic_scm import SyntheticSCM as SCMClass
        from environments.text_scm import state_to_text, action_to_text

    # Import domain-specific variable indices
    if args.domain == "clinical":
        from environments.clinical_scm import (
            NUM_VARIABLES, NUM_CONTROLLABLE,
        )
    elif args.domain == "confounded":
        from environments.confounded_scm import (
            NUM_VARIABLES, NUM_CONTROLLABLE,
        )
    else:
        from environments.synthetic_scm import (
            NUM_VARIABLES, NUM_CONTROLLABLE,
        )

    # Generate 200 scenario pairs
    scm = SCMClass(noise_scale=0.05, seed=42)
    eval_rng = np.random.RandomState(999)
    n_test = 200

    baseline_correct = 0
    bridge_correct = 0
    causeway_correct = 0
    total = 0

    for i in tqdm(range(n_test), desc="Pairwise eval"):
        state = scm.sample_state(1)

        # Action A: increase a random controllable variable
        ctrl_idx_a = eval_rng.randint(0, NUM_CONTROLLABLE)
        mask_a = np.zeros((1, NUM_VARIABLES))
        vals_a = np.zeros((1, NUM_VARIABLES))
        mask_a[0, ctrl_idx_a] = 1.0
        vals_a[0, ctrl_idx_a] = min(
            state[0, ctrl_idx_a] + eval_rng.uniform(0.15, 0.4), 1.0
        )

        # Action B: decrease a random controllable variable
        ctrl_idx_b = eval_rng.randint(0, NUM_CONTROLLABLE)
        mask_b = np.zeros((1, NUM_VARIABLES))
        vals_b = np.zeros((1, NUM_VARIABLES))
        mask_b[0, ctrl_idx_b] = 1.0
        vals_b[0, ctrl_idx_b] = max(
            state[0, ctrl_idx_b] - eval_rng.uniform(0.15, 0.4), 0.0
        )

        # Ground truth
        _, delta_raw_a = scm.intervene(state, mask_a, vals_a)
        _, delta_raw_b = scm.intervene(state, mask_b, vals_b)
        delta_a = scm.compute_structured_delta(delta_raw_a)[0]
        delta_b = scm.compute_structured_delta(delta_raw_b)[0]

        q_a = quality_score(delta_a)
        q_b = quality_score(delta_b)

        if abs(q_a - q_b) < 0.02:
            continue

        ground_truth_better = "A" if q_a > q_b else "B"

        # Encode texts
        state_text = state_to_text(state[0], i % 4)
        text_a = action_to_text(mask_a[0], vals_a[0], i % 4)
        text_b = action_to_text(mask_b[0], vals_b[0], i % 4)

        h = encode_text(state_text)
        a_enc_a = encode_text(text_a)
        a_enc_b = encode_text(text_b)

        with torch.no_grad():
            # Causeway standalone ranking
            cw_delta_a = causeway(h, a_enc_a).values[0].cpu().numpy()
            cw_delta_b = causeway(h, a_enc_b).values[0].cpu().numpy()
            cw_better = "A" if quality_score(cw_delta_a) > quality_score(cw_delta_b) else "B"
            if cw_better == ground_truth_better:
                causeway_correct += 1

            # Bridge: generate prefix from each action, compare P(A) vs P(B)
            prefix_a = bridge.get_causal_prefix(h, a_enc_a)
            prefix_b = bridge.get_causal_prefix(h, a_enc_b)

            prompt_1 = prompt_embeds.expand(1, -1, -1)

            # Action A assessment
            aug_a = torch.cat([prefix_a, prompt_1], dim=1)
            logits_a = lm_model(inputs_embeds=aug_a).logits[0, -1, :]
            p_A_given_a = F.softmax(logits_a[[token_A, token_B]], dim=0)[0].item()

            # Action B assessment
            aug_b = torch.cat([prefix_b, prompt_1], dim=1)
            logits_b = lm_model(inputs_embeds=aug_b).logits[0, -1, :]
            p_A_given_b = F.softmax(logits_b[[token_A, token_B]], dim=0)[0].item()

            # Bridge says A is better if P(A|prefix_a) > P(A|prefix_b)
            bridge_better = "A" if p_A_given_a > p_A_given_b else "B"
            if bridge_better == ground_truth_better:
                bridge_correct += 1

            # Baseline: no prefix, random guess
            if eval_rng.random() > 0.5:
                baseline_correct += 1

        total += 1

    print(f"\n  Scenarios tested: {total}")
    print(f"\n  LM alone (random):          {baseline_correct / max(total, 1) * 100:.1f}%")
    print(f"  Causeway ranking:           {causeway_correct / max(total, 1) * 100:.1f}%")
    print(f"  LM + trained bridge:        {bridge_correct / max(total, 1) * 100:.1f}%")

    if bridge_correct / max(total, 1) > 0.6:
        print("\n  >>> The closed loop works.")
        print("  >>> Causeway's causal state, projected through the trained bridge,")
        print(f"  >>> steers {model_short} toward ground-truth-correct rankings.")
    else:
        print("\n  >>> Bridge needs more training or a different approach.")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
