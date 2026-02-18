"""
Causeway + GPT-2 Integration Demo.

Demonstrates Causeway bolted onto a real frozen GPT-2 model.

Scenario: Software deployment decision-making via natural language.
    1. GPT-2 encodes a deployment scenario description → hidden states
    2. An "action" is encoded as a modification to the scenario
    3. Causeway predicts structured causal deltas
    4. The delta is injected back as a causal prefix
    5. GPT-2 generates a response conditioned on the causal information

This demonstrates the full pipeline: frozen Transformer + Causeway adapter
producing counterfactual-aware outputs.

Usage:
    python demo_gpt2.py
"""

import sys
import os
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causeway.causeway_module import Causeway
from causeway.losses import CausewayLoss
from causeway.delta_predictor import DeltaVector
from integration.transformer_bridge import TransformerBridge
from environments.synthetic_scm import SyntheticSCM, SCMDataset


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ================================================================
    # PHASE 1: Load frozen GPT-2
    # ================================================================
    print_section("PHASE 1: Loading Frozen GPT-2")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
    gpt2_lm = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Freeze everything
    for param in gpt2.parameters():
        param.requires_grad = False
    for param in gpt2_lm.parameters():
        param.requires_grad = False

    d_model = gpt2.config.n_embd  # 768 for gpt2
    print(f"GPT-2 loaded: {sum(p.numel() for p in gpt2.parameters()) / 1e6:.1f}M params (frozen)")
    print(f"Hidden dim: {d_model}")

    # ================================================================
    # PHASE 2: Create Causeway for GPT-2's representation space
    # ================================================================
    print_section("PHASE 2: Creating Causeway Module")

    d_causal = 48   # More causal variables for richer representations
    d_action = 128   # Larger action space for text-derived actions

    causeway = Causeway(
        d_model=d_model,   # 768 — matches GPT-2
        d_causal=d_causal,
        d_action=d_action,
        graph_layers=2,
        propagation_steps=3,
    ).to(device)

    bridge = TransformerBridge(
        causeway=causeway,
        d_model=d_model,
        n_prefix_tokens=4,
    ).to(device)

    causeway_params = sum(p.numel() for p in causeway.parameters())
    bridge_params = sum(p.numel() for p in bridge.parameters()) - causeway_params
    total_new = causeway_params + bridge_params
    gpt2_params = sum(p.numel() for p in gpt2.parameters())

    print(f"Causeway params:  {causeway_params / 1e6:.3f}M")
    print(f"Bridge params:    {bridge_params / 1e6:.3f}M")
    print(f"Total new params: {total_new / 1e6:.3f}M")
    print(f"Overhead vs GPT-2: {total_new / gpt2_params * 100:.2f}%")

    # ================================================================
    # PHASE 3: Extract hidden states from deployment scenarios
    # ================================================================
    print_section("PHASE 3: Encoding Deployment Scenarios")

    scenarios = [
        "Deploy the new authentication service with minimal testing to production servers under heavy load.",
        "Deploy the new authentication service with full test coverage to production servers under low load.",
        "Deploy a minor CSS fix with full test coverage to staging servers under no load.",
        "Deploy a major database migration with no rollback plan to production during peak hours.",
    ]

    actions = [
        "Increase test coverage before deploying.",
        "Reduce deployment load by scheduling off-peak.",
        "Add a rollback plan and monitoring.",
        "Cancel the deployment entirely.",
    ]

    # Encode scenarios and actions with GPT-2
    tokenizer.pad_token = tokenizer.eos_token
    scenario_hidden = []
    action_hidden = []

    for i, (scenario, action) in enumerate(zip(scenarios, actions)):
        # Encode scenario → get last hidden state (summary vector)
        inputs = tokenizer(scenario, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = gpt2(**inputs)
            # Use mean pooling of hidden states as summary
            h = outputs.last_hidden_state.mean(dim=1)  # (1, 768)
            scenario_hidden.append(h)

        # Encode action → get hidden state
        act_inputs = tokenizer(action, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            act_outputs = gpt2(**act_inputs)
            a = act_outputs.last_hidden_state.mean(dim=1)  # (1, 768)
            action_hidden.append(a)

        print(f"\nScenario {i+1}: {scenario[:60]}...")
        print(f"  Action: {action}")
        print(f"  Hidden state norm: {h.norm().item():.2f}")

    # ================================================================
    # PHASE 4: Run Causeway counterfactual inference
    # ================================================================
    print_section("PHASE 4: Causeway Counterfactual Inference")

    dim_names = ["risk_shift", "goal_progress", "constraint_viol",
                 "resource_cost", "success_prob"]

    # Project action hidden states to Causeway's d_action
    # In a trained system, this projection would be learned
    # For demo, we use a random but fixed projection
    torch.manual_seed(42)
    action_proj = nn.Linear(d_model, d_action).to(device)
    for p in action_proj.parameters():
        p.requires_grad = False

    all_deltas = []
    causeway.eval()
    with torch.no_grad():
        for i, (h, a_raw) in enumerate(zip(scenario_hidden, action_hidden)):
            a = action_proj(a_raw)  # (1, d_action)
            delta = causeway(h, a)
            all_deltas.append(delta)

            print(f"\nScenario {i+1}: {scenarios[i][:50]}...")
            print(f"  Action: {actions[i]}")
            print(f"  {'Dimension':<25} {'Delta':>10} {'Confidence':>12}")
            print(f"  {'-'*49}")
            for j, name in enumerate(dim_names):
                val = delta.values[0, j].item()
                conf = delta.confidence[0, j].item()
                direction = "+" if val > 0 else ""
                print(f"  {name:<25} {direction}{val:>9.4f} {conf:>12.3f}")

    # ================================================================
    # PHASE 5: Inject causal prefix and generate
    # ================================================================
    print_section("PHASE 5: Causal Prefix Injection + Generation")

    for i, (h, a_raw) in enumerate(zip(scenario_hidden, action_hidden)):
        a = action_proj(a_raw)

        # Generate causal prefix tokens
        prefix = bridge.get_causal_prefix(h, a)  # (1, 4, 768)

        # Create scenario input embeddings
        inputs = tokenizer(
            f"Deployment risk assessment: {scenarios[i]} Recommendation:",
            return_tensors="pt",
        ).to(device)

        # Get GPT-2 input embeddings
        input_embeds = gpt2_lm.transformer.wte(inputs.input_ids)  # (1, seq, 768)

        # Prepend causal prefix
        augmented_embeds = torch.cat([prefix, input_embeds], dim=1)

        # Generate with augmented embeddings
        # Create attention mask that includes prefix tokens
        prefix_mask = torch.ones(1, prefix.shape[1], device=device)
        orig_mask = inputs.attention_mask
        augmented_mask = torch.cat([prefix_mask, orig_mask], dim=1)

        # Generate without prefix (baseline)
        with torch.no_grad():
            baseline_out = gpt2_lm.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            baseline_text = tokenizer.decode(
                baseline_out[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

        # Generate with causal prefix
        with torch.no_grad():
            # Build position_ids to account for prefix
            total_len = augmented_embeds.shape[1]
            position_ids = torch.arange(total_len, device=device).unsqueeze(0)

            # Forward pass with embeddings to get logits for generation
            causal_out = gpt2_lm(
                inputs_embeds=augmented_embeds,
                attention_mask=augmented_mask,
                position_ids=position_ids,
            )

            # Greedy decode a few tokens from the augmented output
            generated_ids = []
            next_embeds = augmented_embeds
            next_mask = augmented_mask
            for step in range(40):
                pos_ids = torch.arange(next_embeds.shape[1], device=device).unsqueeze(0)
                out = gpt2_lm(
                    inputs_embeds=next_embeds,
                    attention_mask=next_mask,
                    position_ids=pos_ids,
                )
                next_logits = out.logits[:, -1, :] / 0.7
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated_ids.append(next_token.item())
                if next_token.item() == tokenizer.eos_token_id:
                    break
                next_embed = gpt2_lm.transformer.wte(next_token)
                next_embeds = torch.cat([next_embeds, next_embed], dim=1)
                next_mask = torch.cat([
                    next_mask,
                    torch.ones(1, 1, device=device),
                ], dim=1)

            causal_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"\n--- Scenario {i+1} ---")
        print(f"Scenario: {scenarios[i][:70]}...")
        print(f"Action:   {actions[i]}")
        print(f"\nBaseline GPT-2:        {baseline_text[:150]}")
        print(f"Causeway-augmented:    {causal_text[:150]}")

        # Show delta summary
        delta = all_deltas[i]
        risk = delta.values[0, 0].item()
        success = delta.values[0, 4].item()
        print(f"Causal signal injected: risk_shift={risk:+.3f}, success_prob={success:+.3f}")

    # ================================================================
    # PHASE 6: Architecture Summary
    # ================================================================
    print_section("ARCHITECTURE SUMMARY")

    print(f"""
Frozen GPT-2:     {gpt2_params / 1e6:.1f}M params (untouched)
Causeway module:  {causeway_params / 1e6:.3f}M params (trainable)
Bridge module:    {bridge_params / 1e6:.3f}M params (trainable)
Total overhead:   {total_new / gpt2_params * 100:.2f}% additional parameters

Pipeline:
  1. GPT-2 encodes scenario -> hidden state h (d={d_model})
  2. GPT-2 encodes action   -> action repr  a (d={d_model})
  3. Causeway(h, a)         -> delta vector (5 dims)
  4. Bridge(delta)          -> prefix tokens (4 x {d_model})
  5. GPT-2([prefix; input]) -> causally-informed generation

The Transformer is never retrained. Causeway sits beside it.

NOTE: In this demo, Causeway is UNTRAINED on GPT-2's representation space.
The deltas and generated text are baselines showing the pipeline works end-to-end.
To get meaningful outputs, train Causeway on GPT-2 hidden states with
interventional data from the target domain.
    """)

    print("Diagnostics:")
    diag = causeway.get_diagnostics()
    for k, v in diag.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
