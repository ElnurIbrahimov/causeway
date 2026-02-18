"""
Text-Encoded Clinical SCM Dataset for Training on Real Transformer Hidden States.

Converts the clinical SCM's states and interventions into natural language
descriptions, encodes them with a frozen Transformer, and pairs the resulting
hidden states with ground-truth counterfactual deltas.

Same interface as text_scm.py but for the clinical treatment domain.

Flow:
    Clinical SCM state (8 floats) -> text description -> Transformer -> h
    Clinical SCM action (mask+values) -> text description -> Transformer -> a
    Clinical SCM ground truth -> structured delta in R^5

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
from environments.clinical_scm import (
    ClinicalSCM, NUM_VARIABLES, NUM_CONTROLLABLE,
    DRUG_DOSAGE, TREATMENT_INTENSITY, MONITORING_FREQUENCY, ACTIVITY_RESTRICTION,
)


VAR_NAMES = [
    "drug_dosage", "treatment_intensity", "monitoring_frequency", "activity_restriction",
    "adverse_events", "recovery_rate", "organ_stress", "care_cost",
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
    # Template 0: clinical report
    lambda dd, ti, mf, ar, dd_d, ti_d, mf_d, ar_d: (
        f"Patient assessment: {dd_d} drug dosage ({dd:.2f}), "
        f"{ti_d} treatment intensity ({ti:.2f}), "
        f"{mf_d} monitoring frequency ({mf:.2f}), "
        f"activity restriction {ar_d} ({ar:.2f})."
    ),
    # Template 1: chart note
    lambda dd, ti, mf, ar, dd_d, ti_d, mf_d, ar_d: (
        f"Clinical status — dosage: {dd_d} ({dd:.2f}), "
        f"treatment: {ti_d} ({ti:.2f}), monitoring: {mf_d} ({mf:.2f}), "
        f"restriction: {ar_d} ({ar:.2f})."
    ),
    # Template 2: narrative
    lambda dd, ti, mf, ar, dd_d, ti_d, mf_d, ar_d: (
        f"The patient receives a {dd_d} dose ({dd:.2f}) with {ti_d} treatment ({ti:.2f}). "
        f"Monitoring is {mf_d} ({mf:.2f}) and activity is {ar_d} restricted ({ar:.2f})."
    ),
    # Template 3: structured
    lambda dd, ti, mf, ar, dd_d, ti_d, mf_d, ar_d: (
        f"Vitals report: dose={dd_d} ({dd:.2f}), intensity={ti_d} ({ti:.2f}), "
        f"monitoring={mf_d} ({mf:.2f}), restriction={ar_d} ({ar:.2f})."
    ),
]


def state_to_text(state: np.ndarray, template_idx: int = 0) -> str:
    """Convert a single clinical SCM state vector to natural language."""
    dd = state[DRUG_DOSAGE]
    ti = state[TREATMENT_INTENSITY]
    mf = state[MONITORING_FREQUENCY]
    ar = state[ACTIVITY_RESTRICTION]

    dd_desc = _describe_level(dd)
    ti_desc = _describe_level(ti, "aggressive", "standard", "conservative")
    mf_desc = _describe_level(mf, "intensive", "regular", "minimal")
    ar_desc = _describe_level(ar, "strict", "moderate", "light")

    template = _STATE_TEMPLATES[template_idx % len(_STATE_TEMPLATES)]
    return template(dd, ti, mf, ar, dd_desc, ti_desc, mf_desc, ar_desc)


_ACTION_TEMPLATES = [
    lambda parts: "Treatment plan: " + ", ".join(parts) + ".",
    lambda parts: "Clinical orders — " + "; ".join(parts) + ".",
    lambda parts: "Adjust treatment: " + ", ".join(parts) + ".",
    lambda parts: "Prescribe: " + ", ".join(parts) + ".",
]

_NO_ACTION_TEMPLATES = [
    "No changes to the treatment plan.",
    "No clinical interventions planned.",
    "Keep all treatment parameters unchanged.",
    "No treatment adjustments.",
]

_ACTION_LEVEL_DESCRIPTORS = {
    "drug_dosage": ("low", "standard", "high"),
    "treatment_intensity": ("conservative", "standard", "aggressive"),
    "monitoring_frequency": ("minimal", "regular", "intensive"),
    "activity_restriction": ("light", "moderate", "strict"),
}


def action_to_text(mask: np.ndarray, values: np.ndarray, template_idx: int = 0) -> str:
    """Convert an intervention to a natural language action description."""
    parts = []
    for i in range(NUM_CONTROLLABLE):
        if mask[i] > 0.5:
            name = CONTROLLABLE_NAMES[i].replace("_", " ")
            new_val = values[i]
            descs = _ACTION_LEVEL_DESCRIPTORS[CONTROLLABLE_NAMES[i]]
            if new_val > 0.7:
                level = descs[2]
            elif new_val > 0.4:
                level = descs[1]
            else:
                level = descs[0]
            parts.append(f"set {name} to {level} ({new_val:.2f})")

    if not parts:
        return _NO_ACTION_TEMPLATES[template_idx % len(_NO_ACTION_TEMPLATES)]

    template = _ACTION_TEMPLATES[template_idx % len(_ACTION_TEMPLATES)]
    return template(parts)


class TextClinicalDataset(Dataset):
    """
    Dataset that encodes clinical SCM states as text through a real Transformer.

    Precomputes all hidden states to avoid re-encoding during training.
    Same interface as TextSCMDataset.
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
        print(f"Generating {num_samples} clinical SCM samples...")
        scm = ClinicalSCM(noise_scale=noise_scale, seed=scm_seed)
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

        self.h = torch.cat(all_h, dim=0)
        self.actions = torch.cat(all_a, dim=0)

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
