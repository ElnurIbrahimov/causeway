"""
Train TransformerBridge: Close the Loop.

Trains the bridge to convert Causeway's causal delta into prefix tokens
that steer GPT-2's risk assessment toward ground truth.

Frozen: GPT-2 (124M), Causeway (0.794M)
Trainable: Bridge projection (~2.4M params)

After training, runs closed-loop evaluation to prove the prefix
actually changes GPT-2's decisions.

Usage:
    python train_bridge.py
    python train_bridge.py --causeway_checkpoint causeway_gpt2.pt --epochs 30
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
from integration.transformer_bridge import TransformerBridge
from environments.synthetic_scm import (
    SyntheticSCM, NUM_VARIABLES, NUM_CONTROLLABLE,
    CODE_COMPLEXITY, TEST_COVERAGE, DEPLOY_LOAD, ROLLBACK_READINESS,
)
from environments.text_scm import state_to_text, action_to_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--causeway_checkpoint", default="causeway_gpt2.pt")
    p.add_argument("--cache", default="cache_gpt2_50000_v2.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_path", default="bridge_gpt2.pt")
    return p.parse_args()


def quality_score(delta):
    return -delta[0] + delta[1] - delta[2] - delta[3] + delta[4]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    print("=" * 60)
    print("  BRIDGE TRAINING: Teaching GPT-2 to hear Causeway")
    print("=" * 60)
    print(f"Device: {device}")

    # === Load GPT-2 (frozen) ===
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    gpt2_lm = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    for p in gpt2_lm.parameters():
        p.requires_grad = False

    d_model = gpt2_lm.config.n_embd  # 768

    # === Load Causeway (frozen) ===
    print(f"Loading Causeway from {args.causeway_checkpoint}...")
    ckpt = torch.load(args.causeway_checkpoint, map_location=device, weights_only=False)
    cw_args = ckpt["args"]

    causeway = Causeway(
        d_model=d_model,
        d_causal=cw_args["d_causal"],
        d_action=cw_args.get("d_action", d_model),
        graph_layers=cw_args.get("graph_layers", 2),
        propagation_steps=cw_args.get("propagation_steps", 3),
    ).to(device)
    causeway.load_state_dict(ckpt["model_state_dict"])
    causeway.eval()
    for p in causeway.parameters():
        p.requires_grad = False

    # === Create Bridge (trainable projection only) ===
    bridge = TransformerBridge(
        causeway=causeway,
        d_model=d_model,
        n_prefix_tokens=4,
    ).to(device)

    # Freeze Causeway inside bridge, train only projection layers
    bridge.causeway.requires_grad_(False)
    trainable_params = (
        list(bridge.delta_to_prefix.parameters())
        + [bridge.prefix_positions]
        + list(bridge.norm.parameters())
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Bridge trainable params: {n_trainable / 1e6:.3f}M")

    # === Load cached dataset ===
    print(f"Loading dataset from {args.cache}...")
    data = torch.load(args.cache, weights_only=False, map_location="cpu")
    h_all = data["h"].float()
    a_all = data["actions"].float()
    targets_all = data["targets"]

    # Filter out ambiguous samples (near-zero risk shift)
    risk_shifts = targets_all[:, 0]
    mask = risk_shifts.abs() > 0.02
    h_all = h_all[mask]
    a_all = a_all[mask]
    targets_all = targets_all[mask]
    labels_all = (targets_all[:, 0] > 0).long()  # 1=risky, 0=safe

    n_risky = labels_all.sum().item()
    n_safe = len(labels_all) - n_risky
    print(f"Samples: {len(h_all)} ({n_risky} risky, {n_safe} safe)")

    # Split
    dataset = TensorDataset(h_all, a_all, labels_all)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # === Probe setup ===
    high_token = tokenizer.encode(" high")[0]
    low_token = tokenizer.encode(" low")[0]

    prompt_text = "Deployment risk level:"
    prompt_ids = tokenizer.encode(prompt_text)
    prompt_tensor = torch.tensor([prompt_ids], device=device)
    prompt_embeds = gpt2_lm.transformer.wte(prompt_tensor)  # (1, seq, 768)

    print(f"Prompt: '{prompt_text}' ({len(prompt_ids)} tokens)")
    print(f"Probe: ' high' (id={high_token}) vs ' low' (id={low_token})")

    # === Training ===
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_acc = 0
    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        bridge.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for h, a, label in train_loader:
            h, a, label = h.to(device), a.to(device), label.to(device)
            bs = h.shape[0]

            # Bridge produces prefix from Causeway's delta
            prefix = bridge.get_causal_prefix(h, a)  # (B, 4, 768)
            prompt_batch = prompt_embeds.expand(bs, -1, -1)
            augmented = torch.cat([prefix, prompt_batch], dim=1)

            # GPT-2 forward (frozen weights, gradient flows through prefix)
            logits = gpt2_lm(inputs_embeds=augmented).logits[:, -1, :]

            # CE on [high, low] tokens
            # risky (label=1) -> should predict "high" (index 0)
            # safe  (label=0) -> should predict "low"  (index 1)
            probe_logits = logits[:, [high_token, low_token]]  # (B, 2)
            target = 1 - label  # risky->0 (high), safe->1 (low)

            loss = F.cross_entropy(probe_logits, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            train_loss += loss.item() * bs
            train_correct += (probe_logits.argmax(1) == target).sum().item()
            train_total += bs

        scheduler.step()

        # Validation
        bridge.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for h, a, label in val_loader:
                h, a, label = h.to(device), a.to(device), label.to(device)
                bs = h.shape[0]

                prefix = bridge.get_causal_prefix(h, a)
                prompt_batch = prompt_embeds.expand(bs, -1, -1)
                augmented = torch.cat([prefix, prompt_batch], dim=1)

                logits = gpt2_lm(inputs_embeds=augmented).logits[:, -1, :]
                probe_logits = logits[:, [high_token, low_token]]
                target = 1 - label

                val_loss += F.cross_entropy(probe_logits, target).item() * bs
                val_correct += (probe_logits.argmax(1) == target).sum().item()
                val_total += bs

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"train_loss={train_loss / train_total:.4f}  "
                f"train_acc={train_acc:.3f}  "
                f"val_acc={val_acc:.3f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "delta_to_prefix": bridge.delta_to_prefix.state_dict(),
                    "prefix_positions": bridge.prefix_positions.data,
                    "norm": bridge.norm.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                args.save_path,
            )

    # === Load best and evaluate ===
    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    bridge_ckpt = torch.load(args.save_path, weights_only=False, map_location=device)
    bridge.delta_to_prefix.load_state_dict(bridge_ckpt["delta_to_prefix"])
    bridge.prefix_positions.data = bridge_ckpt["prefix_positions"]
    bridge.norm.load_state_dict(bridge_ckpt["norm"])
    bridge.eval()

    # ================================================================
    # EVAL 1: Risk classification — baseline vs trained bridge
    # ================================================================
    print("\n" + "=" * 60)
    print("  EVAL 1: Risk Classification")
    print("=" * 60)

    baseline_correct = 0
    bridge_correct = 0
    total = 0

    with torch.no_grad():
        for h, a, label in val_loader:
            h, a, label = h.to(device), a.to(device), label.to(device)
            bs = h.shape[0]
            target = 1 - label

            # Baseline: prompt only, no prefix
            prompt_batch = prompt_embeds.expand(bs, -1, -1)
            logits_base = gpt2_lm(inputs_embeds=prompt_batch).logits[:, -1, :]
            probe_base = logits_base[:, [high_token, low_token]]
            baseline_correct += (probe_base.argmax(1) == target).sum().item()

            # Bridge: prefix + prompt
            prefix = bridge.get_causal_prefix(h, a)
            augmented = torch.cat([prefix, prompt_batch], dim=1)
            logits_aug = gpt2_lm(inputs_embeds=augmented).logits[:, -1, :]
            probe_aug = logits_aug[:, [high_token, low_token]]
            bridge_correct += (probe_aug.argmax(1) == target).sum().item()

            total += bs

    base_acc = baseline_correct / total
    bridge_acc = bridge_correct / total

    print(f"\n  GPT-2 alone (no prefix):    {base_acc * 100:.1f}%")
    print(f"  GPT-2 + trained bridge:     {bridge_acc * 100:.1f}%")
    print(f"  Lift:                       {(bridge_acc - base_acc) * 100:+.1f}%")

    # ================================================================
    # EVAL 2: Pairwise action selection — the real closed-loop test
    # ================================================================
    print("\n" + "=" * 60)
    print("  EVAL 2: Pairwise Action Selection (closed loop)")
    print("=" * 60)

    from transformers import GPT2Model

    gpt2_enc = GPT2Model.from_pretrained("gpt2").to(device).eval()
    for p in gpt2_enc.parameters():
        p.requires_grad = False

    def encode_text(text):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            out = gpt2_enc(**inputs)
            seq_len = inputs.attention_mask.sum(dim=1) - 1
            h = out.last_hidden_state[0, seq_len[0]]
        return h.unsqueeze(0)

    # Generate 200 scenario pairs
    scm = SyntheticSCM(noise_scale=0.05, seed=42)
    rng = np.random.RandomState(999)
    n_test = 200

    baseline_correct = 0
    bridge_correct = 0
    causeway_correct = 0
    total = 0

    for i in tqdm(range(n_test), desc="Pairwise eval"):
        state = scm.sample_state(1)

        # Safe action: increase test coverage
        safe_mask = np.zeros((1, NUM_VARIABLES))
        safe_vals = np.zeros((1, NUM_VARIABLES))
        safe_mask[0, TEST_COVERAGE] = 1.0
        safe_vals[0, TEST_COVERAGE] = min(
            state[0, TEST_COVERAGE] + rng.uniform(0.15, 0.4), 1.0
        )

        # Risky action: decrease test coverage
        risky_mask = np.zeros((1, NUM_VARIABLES))
        risky_vals = np.zeros((1, NUM_VARIABLES))
        risky_mask[0, TEST_COVERAGE] = 1.0
        risky_vals[0, TEST_COVERAGE] = max(
            state[0, TEST_COVERAGE] - rng.uniform(0.15, 0.4), 0.0
        )

        # Ground truth
        _, safe_raw = scm.intervene(state, safe_mask, safe_vals)
        _, risky_raw = scm.intervene(state, risky_mask, risky_vals)
        safe_delta = scm.compute_structured_delta(safe_raw)[0]
        risky_delta = scm.compute_structured_delta(risky_raw)[0]

        if quality_score(safe_delta) <= quality_score(risky_delta) + 0.02:
            continue

        # Encode texts
        state_text = state_to_text(state[0], i % 4)
        safe_text = action_to_text(safe_mask[0], safe_vals[0], i % 4)
        risky_text = action_to_text(risky_mask[0], risky_vals[0], i % 4)

        h = encode_text(state_text)
        a_safe = encode_text(safe_text)
        a_risky = encode_text(risky_text)

        with torch.no_grad():
            # Causeway standalone ranking
            delta_safe = causeway(h, a_safe).values[0].cpu().numpy()
            delta_risky = causeway(h, a_risky).values[0].cpu().numpy()
            if quality_score(delta_safe) > quality_score(delta_risky):
                causeway_correct += 1

            # Bridge: get P(low risk) for each action
            prefix_safe = bridge.get_causal_prefix(h, a_safe)
            prefix_risky = bridge.get_causal_prefix(h, a_risky)

            prompt_1 = prompt_embeds.expand(1, -1, -1)

            # Safe action assessment
            aug_safe = torch.cat([prefix_safe, prompt_1], dim=1)
            logits_safe = gpt2_lm(inputs_embeds=aug_safe).logits[0, -1, :]
            p_low_safe = F.softmax(logits_safe[[high_token, low_token]], dim=0)[1].item()

            # Risky action assessment
            aug_risky = torch.cat([prefix_risky, prompt_1], dim=1)
            logits_risky = gpt2_lm(inputs_embeds=aug_risky).logits[0, -1, :]
            p_low_risky = F.softmax(logits_risky[[high_token, low_token]], dim=0)[1].item()

            # Bridge says safe action is safer if P(low|safe) > P(low|risky)
            if p_low_safe > p_low_risky:
                bridge_correct += 1

            # Baseline: no prefix, same prompt for both (will always be 50%)
            logits_none = gpt2_lm(inputs_embeds=prompt_1).logits[0, -1, :]
            # Baseline can't distinguish — assign random
            if rng.random() > 0.5:
                baseline_correct += 1

        total += 1

    print(f"\n  Scenarios tested: {total}")
    print(f"\n  GPT-2 alone (random):       {baseline_correct / total * 100:.1f}%")
    print(f"  Causeway ranking:           {causeway_correct / total * 100:.1f}%")
    print(f"  GPT-2 + trained bridge:     {bridge_correct / total * 100:.1f}%")

    if bridge_correct / total > 0.6:
        print("\n  >>> The closed loop works.")
        print("  >>> Causeway's delta, projected through the trained bridge,")
        print("  >>> steers GPT-2 toward ground-truth-correct risk assessments.")
    else:
        print("\n  >>> Bridge needs more training or a different approach.")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
