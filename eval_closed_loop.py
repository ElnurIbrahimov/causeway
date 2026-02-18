"""
Closed-Loop Evaluation: Does Causeway Make GPT-2 Decide Better?

The concept says: "The Transformer then conditions its next step on that delta."
This script tests that claim against SCM ground truth.

Tests:
    1. Delta Accuracy — Causeway predictions vs SCM ground truth (sanity check)
    2. Action Ranking — Can Causeway identify safer actions? (standalone, no generation)
    3. Risk Probe — Does injecting Causeway's analysis shift GPT-2's risk assessment?
    4. Pairwise Selection — Does GPT-2 + Causeway pick safer actions more often?

All scored against the SCM's ground-truth causal effects. No LLM judges.

Usage:
    python eval_closed_loop.py
    python eval_closed_loop.py --checkpoint causeway_gpt2.pt --n_scenarios 200
"""

import argparse
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causeway.causeway_module import Causeway
from integration.transformer_bridge import TransformerBridge
from environments.synthetic_scm import (
    SyntheticSCM, NUM_VARIABLES, NUM_CONTROLLABLE,
    CODE_COMPLEXITY, TEST_COVERAGE, DEPLOY_LOAD, ROLLBACK_READINESS,
)
from environments.text_scm import state_to_text, action_to_text

DIM_NAMES = ["risk_shift", "goal_progress", "constraint_viol", "resource_cost", "success_prob"]


def parse_args():
    p = argparse.ArgumentParser(description="Closed-loop evaluation: Causeway + GPT-2")
    p.add_argument("--checkpoint", type=str, default="causeway_gpt2.pt",
                   help="Path to trained Causeway checkpoint")
    p.add_argument("--n_scenarios", type=int, default=200,
                   help="Number of test scenarios to generate")
    p.add_argument("--seed", type=int, default=777)
    return p.parse_args()


def section(title):
    print(f"\n{'-'*70}")
    print(f"  {title}")
    print(f"{'-'*70}")


def quality_score(delta):
    """Scalar quality from structured delta. Higher = better action.

    delta: [risk_shift, goal_progress, constraint_violation, resource_cost, success_probability]
    Good actions: low risk, high progress, low violations, low cost, high success.
    """
    return -delta[0] + delta[1] - delta[2] - delta[3] + delta[4]


def delta_to_text(delta_values):
    """Convert Causeway's predicted delta to natural language for text injection."""
    names = ["risk", "goal progress", "constraint violation",
             "resource cost", "success probability"]
    parts = []
    for name, val in zip(names, delta_values):
        if abs(val) < 0.01:
            continue
        if val > 0.15:
            parts.append(f"{name} increases significantly")
        elif val > 0.01:
            parts.append(f"{name} increases slightly")
        elif val < -0.15:
            parts.append(f"{name} decreases significantly")
        else:
            parts.append(f"{name} decreases slightly")
    if not parts:
        return "Causal analysis: negligible impact expected. "
    return "Causal analysis: " + ", ".join(parts) + ". "


# =====================================================================
# Model Loading
# =====================================================================

def load_models(checkpoint_path, device):
    """Load frozen Transformer and trained Causeway."""
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    # Load checkpoint first to determine which backbone to use
    print(f"Loading Causeway from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    d_model = ckpt["d_model"]
    backbone = ckpt.get("backbone", "gpt2")

    print(f"Loading {backbone}...")
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load encoder (hidden states) and LM head (logits for probing)
    encoder = AutoModel.from_pretrained(backbone, trust_remote_code=True).to(device).eval()
    lm_model = AutoModelForCausalLM.from_pretrained(backbone, trust_remote_code=True).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in lm_model.parameters():
        p.requires_grad = False

    print(f"Backbone: {backbone}, d_model={d_model}")

    causeway = Causeway(
        d_model=d_model,
        d_causal=args["d_causal"],
        d_action=args.get("d_action", d_model),
        graph_layers=args.get("graph_layers", 2),
        propagation_steps=args.get("propagation_steps", 3),
    ).to(device)
    causeway.load_state_dict(ckpt["model_state_dict"])
    causeway.eval()

    bridge = TransformerBridge(
        causeway=causeway,
        d_model=d_model,
        n_prefix_tokens=4,
    ).to(device)
    bridge.eval()

    n_params = sum(p.numel() for p in causeway.parameters())
    val_corr = ckpt["val_results"]["overall_corr"]
    print(f"Causeway: {n_params / 1e6:.3f}M params, val corr={val_corr:.4f}")

    return tokenizer, encoder, lm_model, causeway, bridge, d_model


def encode_text(text, encoder, tokenizer, device):
    """Last-token pooling of hidden states (matches training pipeline)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = encoder(**inputs)
        seq_len = inputs.attention_mask.sum(dim=1) - 1
        h = out.last_hidden_state[0, seq_len[0]]
    return h.unsqueeze(0).float()  # (1, d_model), ensure fp32


# =====================================================================
# Scenario Generation
# =====================================================================

def generate_scenarios(scm, n, seed):
    """Generate deployment scenarios with paired safe/risky actions.

    Each pair is verified against SCM ground truth: the safe action
    must have a clearly higher quality score than the risky action.
    """
    rng = np.random.RandomState(seed)
    scenarios = []
    attempts = 0

    while len(scenarios) < n and attempts < n * 10:
        attempts += 1
        state = scm.sample_state(1)

        action_type = rng.choice(["coverage", "load", "rollback", "multi"])
        safe_mask = np.zeros((1, NUM_VARIABLES))
        safe_vals = np.zeros((1, NUM_VARIABLES))
        risky_mask = np.zeros((1, NUM_VARIABLES))
        risky_vals = np.zeros((1, NUM_VARIABLES))

        if action_type == "coverage":
            # Safe: increase test coverage. Risky: decrease it.
            safe_mask[0, TEST_COVERAGE] = 1.0
            safe_vals[0, TEST_COVERAGE] = min(
                state[0, TEST_COVERAGE] + rng.uniform(0.15, 0.4), 1.0)
            risky_mask[0, TEST_COVERAGE] = 1.0
            risky_vals[0, TEST_COVERAGE] = max(
                state[0, TEST_COVERAGE] - rng.uniform(0.15, 0.4), 0.0)

        elif action_type == "load":
            # Safe: decrease deploy load. Risky: increase it.
            safe_mask[0, DEPLOY_LOAD] = 1.0
            safe_vals[0, DEPLOY_LOAD] = max(
                state[0, DEPLOY_LOAD] - rng.uniform(0.15, 0.4), 0.0)
            risky_mask[0, DEPLOY_LOAD] = 1.0
            risky_vals[0, DEPLOY_LOAD] = min(
                state[0, DEPLOY_LOAD] + rng.uniform(0.15, 0.4), 1.0)

        elif action_type == "rollback":
            # Safe: increase rollback readiness. Risky: decrease it.
            safe_mask[0, ROLLBACK_READINESS] = 1.0
            safe_vals[0, ROLLBACK_READINESS] = min(
                state[0, ROLLBACK_READINESS] + rng.uniform(0.15, 0.4), 1.0)
            risky_mask[0, ROLLBACK_READINESS] = 1.0
            risky_vals[0, ROLLBACK_READINESS] = max(
                state[0, ROLLBACK_READINESS] - rng.uniform(0.15, 0.4), 0.0)

        else:  # multi: coverage + load combined
            safe_mask[0, TEST_COVERAGE] = 1.0
            safe_mask[0, DEPLOY_LOAD] = 1.0
            safe_vals[0, TEST_COVERAGE] = min(
                state[0, TEST_COVERAGE] + rng.uniform(0.1, 0.3), 1.0)
            safe_vals[0, DEPLOY_LOAD] = max(
                state[0, DEPLOY_LOAD] - rng.uniform(0.1, 0.3), 0.0)
            risky_mask[0, TEST_COVERAGE] = 1.0
            risky_mask[0, DEPLOY_LOAD] = 1.0
            risky_vals[0, TEST_COVERAGE] = max(
                state[0, TEST_COVERAGE] - rng.uniform(0.1, 0.3), 0.0)
            risky_vals[0, DEPLOY_LOAD] = min(
                state[0, DEPLOY_LOAD] + rng.uniform(0.1, 0.3), 1.0)

        # Compute SCM ground-truth deltas
        _, safe_raw = scm.intervene(state, safe_mask, safe_vals)
        _, risky_raw = scm.intervene(state, risky_mask, risky_vals)
        safe_delta = scm.compute_structured_delta(safe_raw)[0]
        risky_delta = scm.compute_structured_delta(risky_raw)[0]

        # Require clear separation — safe must be genuinely safer
        if quality_score(safe_delta) <= quality_score(risky_delta) + 0.05:
            continue

        idx = len(scenarios)
        scenarios.append({
            "state": state,
            "state_text": state_to_text(state[0], idx % 4),
            "safe_text": action_to_text(safe_mask[0], safe_vals[0], idx % 4),
            "risky_text": action_to_text(risky_mask[0], risky_vals[0], idx % 4),
            "safe_delta_gt": safe_delta,
            "risky_delta_gt": risky_delta,
            "safe_quality_gt": quality_score(safe_delta),
            "risky_quality_gt": quality_score(risky_delta),
        })

    print(f"Generated {len(scenarios)} scenarios ({attempts} attempts)")
    return scenarios


# =====================================================================
# TEST 1: Delta Accuracy
# =====================================================================

def test_delta_accuracy(scenarios, encoder, tokenizer, causeway, device):
    """Sanity check: Causeway's predicted deltas vs SCM ground truth."""
    section("TEST 1: Delta Accuracy (sanity check)")

    all_pred = []
    all_gt = []

    for sc in tqdm(scenarios, desc="Delta accuracy"):
        h = encode_text(sc["state_text"], encoder, tokenizer, device)
        for action_type in ["safe", "risky"]:
            a = encode_text(sc[f"{action_type}_text"], encoder, tokenizer, device)
            gt = sc[f"{action_type}_delta_gt"]
            with torch.no_grad():
                pred = causeway(h, a).values[0].cpu().numpy()
            all_pred.append(pred)
            all_gt.append(gt)

    all_pred = np.array(all_pred)
    all_gt = np.array(all_gt)

    print(f"\n  {'Dimension':<22} {'Correlation':>12} {'MAE':>10}")
    print(f"  {'-'*46}")
    for i, name in enumerate(DIM_NAMES):
        corr = np.corrcoef(all_pred[:, i], all_gt[:, i])[0, 1]
        mae = np.abs(all_pred[:, i] - all_gt[:, i]).mean()
        print(f"  {name:<22} {corr:>12.4f} {mae:>10.4f}")

    overall_corr = np.corrcoef(all_pred.flatten(), all_gt.flatten())[0, 1]
    overall_mae = np.abs(all_pred - all_gt).mean()
    print(f"  {'-'*46}")
    print(f"  {'Overall':<22} {overall_corr:>12.4f} {overall_mae:>10.4f}")

    return all_pred, all_gt, overall_corr


# =====================================================================
# TEST 2: Action Ranking
# =====================================================================

def test_action_ranking(scenarios, encoder, tokenizer, causeway, device):
    """Can Causeway correctly rank safe actions above risky ones?"""
    section("TEST 2: Action Ranking (Causeway standalone)")

    correct = 0
    quality_gaps_gt = []
    quality_gaps_pred = []

    for sc in tqdm(scenarios, desc="Action ranking"):
        h = encode_text(sc["state_text"], encoder, tokenizer, device)
        a_safe = encode_text(sc["safe_text"], encoder, tokenizer, device)
        a_risky = encode_text(sc["risky_text"], encoder, tokenizer, device)

        with torch.no_grad():
            delta_safe = causeway(h, a_safe).values[0].cpu().numpy()
            delta_risky = causeway(h, a_risky).values[0].cpu().numpy()

        q_safe = quality_score(delta_safe)
        q_risky = quality_score(delta_risky)

        if q_safe > q_risky:
            correct += 1

        quality_gaps_gt.append(sc["safe_quality_gt"] - sc["risky_quality_gt"])
        quality_gaps_pred.append(q_safe - q_risky)

    acc = correct / len(scenarios)
    gap_corr = np.corrcoef(quality_gaps_gt, quality_gaps_pred)[0, 1]

    print(f"\n  Causeway ranking accuracy: {correct}/{len(scenarios)} ({acc*100:.1f}%)")
    print(f"  Quality gap correlation:  {gap_corr:.4f}")
    print(f"  Oracle gap:    mean={np.mean(quality_gaps_gt):.3f}, std={np.std(quality_gaps_gt):.3f}")
    print(f"  Predicted gap: mean={np.mean(quality_gaps_pred):.3f}, std={np.std(quality_gaps_pred):.3f}")

    return acc, gap_corr


# =====================================================================
# TEST 3: Risk Probe
# =====================================================================

def test_risk_probe(scenarios, encoder, lm_model, tokenizer, causeway, device):
    """Does injecting Causeway's delta as text shift GPT-2's risk assessment?"""
    section("TEST 3: Risk Probe (text injection)")

    # Find single-token probe words
    risk_ids = []
    safe_ids = []
    for w in [" high", " High", " risky", " dangerous", " critical"]:
        ids = tokenizer.encode(w)
        if len(ids) == 1:
            risk_ids.append(ids[0])
    for w in [" low", " Low", " safe", " minimal", " good"]:
        ids = tokenizer.encode(w)
        if len(ids) == 1:
            safe_ids.append(ids[0])

    print(f"  Risk tokens: {[tokenizer.decode([t]) for t in risk_ids]}")
    print(f"  Safe tokens: {[tokenizer.decode([t]) for t in safe_ids]}")

    baseline_scores = []
    augmented_scores = []
    gt_risks = []

    for sc in tqdm(scenarios, desc="Risk probe"):
        h = encode_text(sc["state_text"], encoder, tokenizer, device)

        for action_type in ["safe", "risky"]:
            action_text = sc[f"{action_type}_text"]
            gt_delta = sc[f"{action_type}_delta_gt"]
            gt_risk = gt_delta[0]  # risk_shift dimension

            # Get Causeway's predicted delta
            a = encode_text(action_text, encoder, tokenizer, device)
            with torch.no_grad():
                pred_delta = causeway(h, a).values[0].cpu().numpy()

            # Baseline prompt
            base_prompt = (
                f"{sc['state_text']} {action_text} Risk level:"
            )
            # Augmented prompt (Causeway analysis prepended)
            causal_text = delta_to_text(pred_delta)
            aug_prompt = (
                f"{causal_text}{sc['state_text']} {action_text} Risk level:"
            )

            # Baseline logits
            inputs_base = tokenizer(
                base_prompt, return_tensors="pt", truncation=True, max_length=200
            ).to(device)
            with torch.no_grad():
                logits_base = lm_model(**inputs_base).logits[0, -1, :]
            probs_base = F.softmax(logits_base, dim=-1)
            p_risk = probs_base[risk_ids].sum().item()
            p_safe = probs_base[safe_ids].sum().item()
            baseline_scores.append(p_risk / (p_risk + p_safe + 1e-8))

            # Augmented logits (text injection)
            inputs_aug = tokenizer(
                aug_prompt, return_tensors="pt", truncation=True, max_length=300
            ).to(device)
            with torch.no_grad():
                logits_aug = lm_model(**inputs_aug).logits[0, -1, :]
            probs_aug = F.softmax(logits_aug, dim=-1)
            p_risk = probs_aug[risk_ids].sum().item()
            p_safe = probs_aug[safe_ids].sum().item()
            augmented_scores.append(p_risk / (p_risk + p_safe + 1e-8))

            gt_risks.append(gt_risk)

    baseline_scores = np.array(baseline_scores)
    augmented_scores = np.array(augmented_scores)
    gt_risks = np.array(gt_risks)
    gt_binary = (gt_risks > 0).astype(float)

    # Correlations with ground-truth risk
    base_corr = np.corrcoef(baseline_scores, gt_risks)[0, 1]
    aug_corr = np.corrcoef(augmented_scores, gt_risks)[0, 1]

    # Binary accuracy: does risk_score > 0.5 match gt_risk > 0?
    base_acc = ((baseline_scores > 0.5) == gt_binary).mean()
    aug_acc = ((augmented_scores > 0.5) == gt_binary).mean()

    print(f"\n  {'Method':<25} {'Risk Corr':>12} {'Risk Acc':>10}")
    print(f"  {'-'*49}")
    print(f"  {'Baseline (GPT-2)':<25} {base_corr:>12.4f} {base_acc*100:>9.1f}%")
    print(f"  {'+ Causeway text':<25} {aug_corr:>12.4f} {aug_acc*100:>9.1f}%")
    print(f"  {'-'*49}")
    print(f"  {'Lift':<25} {aug_corr - base_corr:>+12.4f} {(aug_acc - base_acc)*100:>+9.1f}%")

    return base_corr, aug_corr, base_acc, aug_acc


# =====================================================================
# TEST 4: Pairwise Selection
# =====================================================================

def test_pairwise_selection(scenarios, encoder, lm_model, tokenizer, causeway, bridge, device):
    """Given two actions, does GPT-2 pick the safer one with Causeway's help?"""
    section("TEST 4: Pairwise Selection")

    token_A = tokenizer.encode(" A")[0]
    token_B = tokenizer.encode(" B")[0]

    rng = np.random.RandomState(42)
    baseline_correct = 0
    text_correct = 0
    prefix_correct = 0
    total = 0

    for sc in tqdm(scenarios, desc="Pairwise selection"):
        # Randomly assign safe/risky to A/B to avoid position bias
        if rng.random() > 0.5:
            opt_a_text, opt_b_text = sc["safe_text"], sc["risky_text"]
            correct_token = token_A
        else:
            opt_a_text, opt_b_text = sc["risky_text"], sc["safe_text"]
            correct_token = token_B

        # Encode for Causeway
        h = encode_text(sc["state_text"], encoder, tokenizer, device)
        a_a = encode_text(opt_a_text, encoder, tokenizer, device)
        a_b = encode_text(opt_b_text, encoder, tokenizer, device)

        with torch.no_grad():
            delta_a = causeway(h, a_a).values[0].cpu().numpy()
            delta_b = causeway(h, a_b).values[0].cpu().numpy()

        # Base prompt (no Causeway info)
        base_prompt = (
            f"{sc['state_text']} "
            f"Option A: {opt_a_text} "
            f"Option B: {opt_b_text} "
            f"The safer deployment option is Option"
        )

        # Augmented prompt (Causeway analysis for both options)
        analysis_a = delta_to_text(delta_a)
        analysis_b = delta_to_text(delta_b)
        aug_prompt = (
            f"Option A impact: {analysis_a}"
            f"Option B impact: {analysis_b}"
            f"{sc['state_text']} "
            f"Option A: {opt_a_text} "
            f"Option B: {opt_b_text} "
            f"The safer deployment option is Option"
        )

        # === Baseline: GPT-2 alone ===
        inputs = tokenizer(
            base_prompt, return_tensors="pt", truncation=True, max_length=300
        ).to(device)
        with torch.no_grad():
            logits = lm_model(**inputs).logits[0, -1, :]
        choice = token_A if logits[token_A] > logits[token_B] else token_B
        if choice == correct_token:
            baseline_correct += 1

        # === Text injection: GPT-2 + Causeway analysis as text ===
        inputs_aug = tokenizer(
            aug_prompt, return_tensors="pt", truncation=True, max_length=400
        ).to(device)
        with torch.no_grad():
            logits_aug = lm_model(**inputs_aug).logits[0, -1, :]
        choice = token_A if logits_aug[token_A] > logits_aug[token_B] else token_B
        if choice == correct_token:
            text_correct += 1

        # === Prefix injection: GPT-2 + TransformerBridge (untrained) ===
        # Inject prefix from the Causeway-preferred action
        q_a = quality_score(delta_a)
        q_b = quality_score(delta_b)
        best_action_repr = a_a if q_a > q_b else a_b
        prefix = bridge.get_causal_prefix(h, best_action_repr)

        # Get input embeddings (works for both GPT-2 and Mistral/Llama)
        if hasattr(lm_model, 'transformer'):
            input_embeds = lm_model.transformer.wte(inputs.input_ids)
        else:
            input_embeds = lm_model.model.embed_tokens(inputs.input_ids)
        augmented_embeds = torch.cat([prefix, input_embeds], dim=1)
        aug_mask = torch.ones(1, augmented_embeds.shape[1], device=device)
        pos_ids = torch.arange(augmented_embeds.shape[1], device=device).unsqueeze(0)

        with torch.no_grad():
            logits_prefix = lm_model(
                inputs_embeds=augmented_embeds,
                attention_mask=aug_mask,
                position_ids=pos_ids,
            ).logits[0, -1, :]
        choice = token_A if logits_prefix[token_A] > logits_prefix[token_B] else token_B
        if choice == correct_token:
            prefix_correct += 1

        total += 1

    base_acc = baseline_correct / total
    text_acc = text_correct / total
    prefix_acc = prefix_correct / total

    print(f"\n  {'Method':<32} {'Selection Accuracy':>20}")
    print(f"  {'-'*54}")
    print(f"  {'Baseline (GPT-2)':<32} {base_acc*100:>19.1f}%")
    print(f"  {'+ Causeway text injection':<32} {text_acc*100:>19.1f}%")
    print(f"  {'+ Bridge prefix (untrained)':<32} {prefix_acc*100:>19.1f}%")
    print(f"  {'-'*54}")
    print(f"  {'Text lift vs baseline':<32} {(text_acc - base_acc)*100:>+19.1f}%")
    print(f"  {'Prefix lift vs baseline':<32} {(prefix_acc - base_acc)*100:>+19.1f}%")

    return base_acc, text_acc, prefix_acc


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"{'='*70}")
    print(f"  CLOSED-LOOP EVALUATION: Causeway + GPT-2")
    print(f"{'='*70}")
    print(f"Device: {device}")

    # Load models
    tokenizer, encoder, lm_model, causeway, bridge, d_model = load_models(
        args.checkpoint, device
    )

    # Generate test scenarios
    print(f"\nGenerating {args.n_scenarios} test scenarios...")
    scm = SyntheticSCM(noise_scale=0.05, seed=42)
    scenarios = generate_scenarios(scm, args.n_scenarios, args.seed)

    # Run tests
    all_pred, all_gt, delta_corr = test_delta_accuracy(
        scenarios, encoder, tokenizer, causeway, device
    )

    rank_acc, gap_corr = test_action_ranking(
        scenarios, encoder, tokenizer, causeway, device
    )

    base_risk_corr, aug_risk_corr, base_risk_acc, aug_risk_acc = test_risk_probe(
        scenarios, encoder, lm_model, tokenizer, causeway, device
    )

    base_sel, text_sel, prefix_sel = test_pairwise_selection(
        scenarios, encoder, lm_model, tokenizer, causeway, bridge, device
    )

    # Summary
    section("SUMMARY")
    print(f"""
  Causeway delta accuracy:     {delta_corr:.4f} correlation with ground truth
  Causeway action ranking:     {rank_acc*100:.1f}% correct (safe > risky)
  Quality gap correlation:     {gap_corr:.4f}

  Risk probe:
    GPT-2 baseline:            corr={base_risk_corr:.4f}, acc={base_risk_acc*100:.1f}%
    + Causeway text:           corr={aug_risk_corr:.4f}, acc={aug_risk_acc*100:.1f}%
    Lift:                      {(aug_risk_corr - base_risk_corr):+.4f} corr, {(aug_risk_acc - base_risk_acc)*100:+.1f}% acc

  Pairwise selection:
    GPT-2 baseline:            {base_sel*100:.1f}%
    + Causeway text:           {text_sel*100:.1f}% ({(text_sel - base_sel)*100:+.1f}%)
    + Bridge prefix:           {prefix_sel*100:.1f}% ({(prefix_sel - base_sel)*100:+.1f}%)
""")

    # Interpretation
    if text_sel > base_sel + 0.05:
        print("  >>> The closed loop works. Causeway's causal deltas, injected as text,")
        print("  >>> improve GPT-2's deployment decisions against SCM ground truth.")
    elif text_sel > base_sel:
        print("  >>> Marginal improvement. The signal helps but the effect is small.")
    else:
        print("  >>> No improvement from text injection. The injection mechanism needs work.")

    if prefix_sel < text_sel - 0.05:
        print("  >>> Bridge prefix underperforms text injection — bridge needs training.")
        print("  >>> Next step: train the bridge on labeled decision data.")
    elif prefix_sel > base_sel + 0.05:
        print("  >>> Bridge prefix works even untrained — the delta signal is strong.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
