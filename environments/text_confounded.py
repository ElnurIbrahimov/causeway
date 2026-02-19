"""
Text-Encoded Confounded SCM Dataset for Training on Real Transformer Hidden States.

Converts the confounded SCM's states and interventions into natural language
descriptions, encodes them with a frozen Transformer, and pairs the resulting
hidden states with ground-truth counterfactual deltas.

Same interface as text_scm.py but for the confounded domain.

CRITICAL DESIGN CHOICE: This domain uses completely neutral language with NO
qualitative descriptors. No "high", "low", "good", "bad", "risky", "safe".
Just numbers. The text gives zero directional cues about what's good or bad,
forcing the causal model to learn purely from structural relationships.

Flow:
    Confounded SCM state (8 floats) -> text description -> Transformer -> h
    Confounded SCM action (mask+values) -> text description -> Transformer -> a
    Confounded SCM ground truth -> structured delta in R^5

    Train Causeway: f(h, a) -> delta
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.confounded_scm import (
    ConfoundedSCM, NUM_VARIABLES, NUM_CONTROLLABLE,
    ALPHA, BETA, GAMMA, DELTA_VAR,
)


VAR_NAMES = [
    "alpha", "beta", "gamma", "delta_var",
    "epsilon_var", "zeta", "eta", "theta",
]

CONTROLLABLE_NAMES = VAR_NAMES[:NUM_CONTROLLABLE]


# ---------------------------------------------------------------------------
# State templates: pure numerical, no qualitative descriptors
# ---------------------------------------------------------------------------

_STATE_TEMPLATES = [
    # Template 0: system state
    lambda a, b, g, d, e, z, et, th: (
        f"System state: alpha={a:.2f}, beta={b:.2f}, gamma={g:.2f}, "
        f"delta={d:.2f}, epsilon={e:.2f}, zeta={z:.2f}, eta={et:.2f}, "
        f"theta={th:.2f}."
    ),
    # Template 1: configuration
    lambda a, b, g, d, e, z, et, th: (
        f"Configuration: alpha {a:.2f}, beta {b:.2f}, gamma {g:.2f}, "
        f"delta {d:.2f}, epsilon {e:.2f}, zeta {z:.2f}, eta {et:.2f}, "
        f"theta {th:.2f}."
    ),
    # Template 2: parameters
    lambda a, b, g, d, e, z, et, th: (
        f"Parameters -- alpha: {a:.2f}, beta: {b:.2f}, gamma: {g:.2f}, "
        f"delta: {d:.2f}, epsilon: {e:.2f}, zeta: {z:.2f}, eta: {et:.2f}, "
        f"theta: {th:.2f}."
    ),
    # Template 3: state vector
    lambda a, b, g, d, e, z, et, th: (
        f"State vector: [alpha={a:.2f}, beta={b:.2f}, gamma={g:.2f}, "
        f"delta={d:.2f}, epsilon={e:.2f}, zeta={z:.2f}, eta={et:.2f}, "
        f"theta={th:.2f}]."
    ),
]


def state_to_text(state: np.ndarray, template_idx: int = 0) -> str:
    """Convert a single confounded SCM state vector to neutral text.

    No qualitative descriptors -- just formatted float values.
    """
    a = state[ALPHA]
    b = state[BETA]
    g = state[GAMMA]
    d = state[DELTA_VAR]
    # Effect variables (indices 4-7)
    e = state[4]
    z = state[5]
    et = state[6]
    th = state[7]

    template = _STATE_TEMPLATES[template_idx % len(_STATE_TEMPLATES)]
    return template(a, b, g, d, e, z, et, th)


# ---------------------------------------------------------------------------
# Action templates: pure numerical, no qualitative descriptors
# ---------------------------------------------------------------------------

_ACTION_TEMPLATES = [
    # Template 0: intervention
    lambda parts: "Intervention: " + ", ".join(parts) + ".",
    # Template 1: arrow notation
    lambda parts: "Apply: " + ", ".join(parts) + ".",
    # Template 2: adjust
    lambda parts: "Adjust parameters: " + ", ".join(parts) + ".",
    # Template 3: change
    lambda parts: "Change: " + ", ".join(parts) + ".",
]

_NO_ACTION_TEMPLATES = [
    "No intervention.",
    "No changes applied.",
    "Parameters unchanged.",
    "Null operation.",
]

# Per-template formatters for individual variable interventions
_ACTION_PART_TEMPLATES = [
    # Template 0: "set X to Y"
    lambda name, val: f"set {name} to {val:.2f}",
    # Template 1: "X -> Y"  (arrow notation)
    lambda name, val: f"{name} -> {val:.2f}",
    # Template 2: "X=Y"
    lambda name, val: f"{name}={val:.2f}",
    # Template 3: "X to Y"
    lambda name, val: f"{name} to {val:.2f}",
]


def action_to_text(mask: np.ndarray, values: np.ndarray, template_idx: int = 0) -> str:
    """Convert an intervention to neutral text.

    No qualitative descriptors -- just variable names and float values.
    """
    tidx = template_idx % len(_ACTION_TEMPLATES)
    part_formatter = _ACTION_PART_TEMPLATES[tidx]

    parts = []
    for i in range(NUM_CONTROLLABLE):
        if mask[i] > 0.5:
            name = CONTROLLABLE_NAMES[i]
            new_val = values[i]
            parts.append(part_formatter(name, new_val))

    if not parts:
        return _NO_ACTION_TEMPLATES[tidx]

    template = _ACTION_TEMPLATES[tidx]
    return template(parts)


class TextConfoundedDataset(Dataset):
    """
    Dataset that encodes confounded SCM states as text through a real Transformer.

    Precomputes all hidden states to avoid re-encoding during training.
    Same interface as TextSCMDataset.

    Uses completely neutral language with no qualitative descriptors,
    so the Transformer embeddings carry no directional cues about
    what's good or bad.
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
            self.h = data["h"].float()
            self.actions = data["actions"].float()
            self.targets = data["targets"]
            self.d_model = data["d_model"]
            self.texts_scenario = data["texts_scenario"]
            self.texts_action = data["texts_action"]
            self.intervention_masks = data["intervention_masks"]
            print(f"Loaded {len(self.h)} samples, d_model={self.d_model}, dtype={self.h.dtype}")
            return

        # Generate SCM data
        print(f"Generating {num_samples} confounded SCM samples...")
        scm = ConfoundedSCM(noise_scale=noise_scale, seed=scm_seed)
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
        print("Converting to neutral text descriptions...")
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

        # Batch encode using last-token pooling
        all_h = []
        all_a = []

        def _last_token_pool(hidden_states, attention_mask):
            """Extract the last non-padding token's hidden state."""
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, seq_lengths]

        for start in tqdm(range(0, num_samples, batch_encode_size), desc="Encoding"):
            end = min(start + batch_encode_size, num_samples)
            batch_scenarios = texts_scenario[start:end]
            batch_actions = texts_action[start:end]

            with torch.no_grad():
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

        del transformer
        torch.cuda.empty_cache()

        self.h = torch.cat(all_h, dim=0).float()       # ensure fp32
        self.actions = torch.cat(all_a, dim=0).float()  # ensure fp32

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
