"""
Text-Encoded SCM Dataset for Training on Real Transformer Hidden States.

Converts the synthetic SCM's states and interventions into natural language
descriptions, encodes them with a frozen Transformer, and pairs the resulting
hidden states with ground-truth counterfactual deltas.

This bridges synthetic causal supervision with real Transformer representations.

Flow:
    SCM state (8 floats) → text description → Transformer → h ∈ R^d_model
    SCM action (mask+values) → text description → Transformer → a ∈ R^d_model
    SCM ground truth → structured delta ∈ R^5

    Train Causeway: f(h, a) → delta

V2 changes (information preservation):
    - Last-token pooling replaces mean pooling. For autoregressive models like
      GPT-2, the last token has attended to the full sequence and carries the
      richest contextual summary. Mean pooling dilutes this signal.
    - Actions stored as full d_model-dim embeddings instead of being projected
      through a random matrix to 128 dims. The random projection destroyed
      most action information, creating an information bottleneck.
    - Template diversity: 4 text templates per state/action rotate across
      samples so GPT-2 produces more varied embeddings across the
      representation space, preventing template-induced clustering.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.synthetic_scm import (
    SyntheticSCM, NUM_VARIABLES, NUM_CONTROLLABLE,
    CODE_COMPLEXITY, TEST_COVERAGE, DEPLOY_LOAD, ROLLBACK_READINESS,
)


VAR_NAMES = [
    "code_complexity", "test_coverage", "deploy_load", "rollback_readiness",
    "error_rate", "latency_impact", "user_impact", "resource_usage",
]

CONTROLLABLE_NAMES = VAR_NAMES[:NUM_CONTROLLABLE]


def _describe_level(val, high="high", mid="moderate", low="low",
                     high_thresh=0.7, mid_thresh=0.4):
    if val > high_thresh:
        return high
    elif val > mid_thresh:
        return mid
    return low


_STATE_TEMPLATES = [
    # Template 0: original
    lambda cc, tc, dl, rr, cd, td, dd, rd: (
        f"Deployment scenario: {cd} code complexity ({cc:.2f}), "
        f"{td} test coverage ({tc:.2f}), "
        f"{dd} server load ({dl:.2f}), "
        f"rollback {rd} ({rr:.2f})."
    ),
    # Template 1: status report style
    lambda cc, tc, dl, rr, cd, td, dd, rd: (
        f"System status — complexity: {cd} ({cc:.2f}), "
        f"tests: {td} ({tc:.2f}), load: {dd} ({dl:.2f}), "
        f"rollback: {rd} ({rr:.2f})."
    ),
    # Template 2: narrative style
    lambda cc, tc, dl, rr, cd, td, dd, rd: (
        f"The codebase has {cd} complexity ({cc:.2f}) with {td} tests ({tc:.2f}). "
        f"Server load is {dd} ({dl:.2f}) and rollback is {rd} ({rr:.2f})."
    ),
    # Template 3: bullet-like style
    lambda cc, tc, dl, rr, cd, td, dd, rd: (
        f"State: complexity={cd} ({cc:.2f}), coverage={td} ({tc:.2f}), "
        f"load={dd} ({dl:.2f}), rollback={rd} ({rr:.2f})."
    ),
]


def state_to_text(state: np.ndarray, template_idx: int = 0) -> str:
    """Convert a single SCM state vector to a natural language description."""
    cc = state[CODE_COMPLEXITY]
    tc = state[TEST_COVERAGE]
    dl = state[DEPLOY_LOAD]
    rr = state[ROLLBACK_READINESS]

    cc_desc = _describe_level(cc)
    tc_desc = _describe_level(tc, "comprehensive", "partial", "minimal", 0.8, 0.5)
    dl_desc = _describe_level(dl, "heavy", "moderate", "light")
    rr_desc = _describe_level(rr, "ready", "partial", "unprepared")

    template = _STATE_TEMPLATES[template_idx % len(_STATE_TEMPLATES)]
    return template(cc, tc, dl, rr, cc_desc, tc_desc, dl_desc, rr_desc)


_ACTION_TEMPLATES = [
    # Template 0: original
    lambda parts: "Proposed action: " + ", ".join(parts) + ".",
    # Template 1: imperative
    lambda parts: "Action plan — " + "; ".join(parts) + ".",
    # Template 2: change-request style
    lambda parts: "Requesting changes: " + ", ".join(parts) + ".",
    # Template 3: terse
    lambda parts: "Do: " + ", ".join(parts) + ".",
]

_NO_ACTION_TEMPLATES = [
    "No changes to the deployment.",
    "No interventions planned.",
    "Keep all parameters unchanged.",
    "No action taken.",
]


def action_to_text(mask: np.ndarray, values: np.ndarray, template_idx: int = 0) -> str:
    """Convert an intervention to a natural language action description."""
    parts = []
    for i in range(NUM_CONTROLLABLE):
        if mask[i] > 0.5:
            old_label = CONTROLLABLE_NAMES[i].replace("_", " ")
            new_val = values[i]
            level = _describe_level(new_val)
            parts.append(f"set {old_label} to {level} ({new_val:.2f})")

    if not parts:
        return _NO_ACTION_TEMPLATES[template_idx % len(_NO_ACTION_TEMPLATES)]

    template = _ACTION_TEMPLATES[template_idx % len(_ACTION_TEMPLATES)]
    return template(parts)


class TextSCMDataset(Dataset):
    """
    Dataset that encodes SCM states as text through a real Transformer.

    Precomputes all hidden states to avoid re-encoding during training.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        num_samples: int = 10000,
        noise_scale: float = 0.05,
        scm_seed: int = 42,
        data_seed: int = 1042,
        batch_encode_size: int = 64,
        device: str = "cuda",
        cache_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            model_name: HuggingFace model name for the Transformer backbone.
            num_samples: Number of training samples to generate.
            noise_scale: SCM exogenous noise scale.
            scm_seed: Seed for the SCM.
            data_seed: Seed for data generation.
            batch_encode_size: Batch size for encoding text with Transformer.
            device: Device for Transformer encoding.
            cache_path: If set, cache encoded data to disk.
        """
        super().__init__()
        self.model_name = model_name

        # Check cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            data = torch.load(cache_path, weights_only=False)
            self.h = data["h"]
            self.actions = data["actions"]
            self.targets = data["targets"]
            self.d_model = data["d_model"]
            self.texts_scenario = data["texts_scenario"]
            self.texts_action = data["texts_action"]
            self.intervention_masks = data["intervention_masks"]
            print(f"Loaded {len(self.h)} samples, d_model={self.d_model}")
            return

        # Generate SCM data
        print(f"Generating {num_samples} SCM samples...")
        scm = SyntheticSCM(noise_scale=noise_scale, seed=scm_seed)
        data_rng = np.random.RandomState(data_seed)

        states = scm.sample_state(num_samples)
        intervention_mask = np.zeros((num_samples, NUM_VARIABLES))
        intervention_values = np.zeros((num_samples, NUM_VARIABLES))

        for i in range(num_samples):
            n_intervene = data_rng.randint(1, NUM_CONTROLLABLE)
            targets = data_rng.choice(NUM_CONTROLLABLE, n_intervene, replace=False)
            intervention_mask[i, targets] = 1.0
            intervention_values[i, targets] = data_rng.uniform(0.0, 1.0, n_intervene)

        _, raw_deltas = scm.intervene(states, intervention_mask, intervention_values)
        structured_deltas = scm.compute_structured_delta(raw_deltas)

        # Convert to text (with template diversity)
        print("Converting to natural language...")
        n_templates = len(_STATE_TEMPLATES)
        texts_scenario = [
            state_to_text(states[i], template_idx=i % n_templates)
            for i in range(num_samples)
        ]
        texts_action = [
            action_to_text(intervention_mask[i], intervention_values[i],
                           template_idx=i % n_templates)
            for i in range(num_samples)
        ]

        # Encode with Transformer
        print(f"Encoding with {model_name}...")
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load in float16 for large models (7B+ need ~14GB instead of ~28GB in fp32)
        # float32 for small models like GPT-2 where precision matters more than VRAM
        load_dtype = kwargs.get("dtype", None)
        if load_dtype is None:
            load_dtype = torch.float16 if "gpt2" not in model_name.lower() else torch.float32
        print(f"Loading model in {load_dtype}...")

        transformer = AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=load_dtype,
        ).to(device)
        transformer.eval()

        self.d_model = transformer.config.hidden_size
        print(f"Model hidden dim: {self.d_model}")

        # Batch encode scenarios using last-token pooling
        # GPT-2 is autoregressive — the last token has attended to everything
        all_h = []
        all_a = []

        def _last_token_pool(hidden_states, attention_mask):
            """Extract the last non-padding token's hidden state."""
            # attention_mask: (batch, seq_len), 1 for real tokens, 0 for padding
            # last real token index per sample
            seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, seq_lengths]  # (batch, d_model)

        for start in tqdm(range(0, num_samples, batch_encode_size), desc="Encoding"):
            end = min(start + batch_encode_size, num_samples)
            batch_scenarios = texts_scenario[start:end]
            batch_actions = texts_action[start:end]

            with torch.no_grad():
                # Encode scenarios — last-token pooling
                s_inputs = tokenizer(
                    batch_scenarios,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(device)
                s_out = transformer(**s_inputs)
                h = _last_token_pool(s_out.last_hidden_state, s_inputs.attention_mask)
                all_h.append(h.cpu())

                # Encode actions — last-token pooling, full d_model dims
                a_inputs = tokenizer(
                    batch_actions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64,
                ).to(device)
                a_out = transformer(**a_inputs)
                a = _last_token_pool(a_out.last_hidden_state, a_inputs.attention_mask)
                all_a.append(a.cpu())

        # Free Transformer memory
        del transformer
        torch.cuda.empty_cache()

        self.h = torch.cat(all_h, dim=0).float()       # (N, d_model) — ensure fp32
        self.actions = torch.cat(all_a, dim=0).float()  # (N, d_model) — ensure fp32

        self.targets = torch.tensor(structured_deltas, dtype=torch.float32)
        self.texts_scenario = texts_scenario
        self.texts_action = texts_action
        self.intervention_masks = torch.tensor(intervention_mask, dtype=torch.float32)

        print(f"Dataset ready: {len(self.h)} samples")
        print(f"  h shape: {self.h.shape}")
        print(f"  action shape: {self.actions.shape}")
        print(f"  target shape: {self.targets.shape}")

        # Cache to disk
        if cache_path:
            print(f"Caching to {cache_path}")
            torch.save({
                "h": self.h,
                "actions": self.actions,
                "targets": self.targets,
                "d_model": self.d_model,
                "texts_scenario": self.texts_scenario,
                "texts_action": self.texts_action,
                "intervention_masks": self.intervention_masks,
            }, cache_path)

    def __len__(self) -> int:
        return len(self.h)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.h[idx], self.actions[idx], self.targets[idx]
