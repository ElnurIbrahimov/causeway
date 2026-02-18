"""
Self-Supervised Causeway Dataset.

Generates scenario/intervention pairs across multiple domains, encodes them
through a frozen Transformer, and uses the representation shift (h_delta)
as the training signal. No external ground truth needed — the Transformer's
own representations provide supervision.

The core insight: when a Transformer encodes "deploy with no tests under
heavy load" vs "deploy with full tests under low load," the hidden states
shift. That shift IS the Transformer's implicit estimate of what changed.
It's noisy, it conflates correlation with causation, but it's a signal.

Causeway learns to decompose these representation shifts through its causal
bottleneck (state encoder → causal graph → do-operator → structured delta).
If the causal bottleneck outperforms a flat MLP at predicting h_delta,
the causal structure is extracting something real.

Flow:
    scenario_base      → Transformer → h_base
    scenario_intervened → Transformer → h_intervened
    intervention_text   → Transformer → action
    h_delta = h_intervened - h_base   ← training target

    Train: Causeway(h_base, action) → structured delta → reconstruct h_delta

Domains:
    1. Software Deployment  — code quality, testing, infrastructure
    2. Clinical Treatment   — patient state, medication, vitals
    3. Business Operations  — market conditions, inventory, pricing
    4. Infrastructure Mgmt  — system health, security, capacity
    5. Project Management   — team state, technical debt, deadlines
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
from tqdm import tqdm


# ── Domain Definitions ──────────────────────────────────────────────
# Each variable: (short_name, display_name, [low_label, mid_label, high_label])

DOMAINS = [
    {
        'name': 'deployment',
        'context': 'software deployment',
        'vars': [
            ('complexity', 'code complexity', ['low', 'moderate', 'high']),
            ('coverage', 'test coverage', ['minimal', 'partial', 'comprehensive']),
            ('load', 'server load', ['light', 'moderate', 'heavy']),
            ('rollback', 'rollback readiness', ['unprepared', 'partial', 'ready']),
            ('experience', 'team experience', ['junior', 'mid-level', 'senior']),
            ('monitoring', 'monitoring depth', ['minimal', 'basic', 'comprehensive']),
        ],
    },
    {
        'name': 'clinical',
        'context': 'patient treatment',
        'vars': [
            ('severity', 'symptom severity', ['mild', 'moderate', 'severe']),
            ('dose', 'medication dose', ['low', 'standard', 'high']),
            ('vitals', 'vital stability', ['unstable', 'variable', 'stable']),
            ('labs', 'lab results', ['abnormal', 'borderline', 'normal']),
            ('compliance', 'treatment adherence', ['poor', 'partial', 'good']),
            ('comorbidity', 'comorbidity risk', ['minimal', 'moderate', 'significant']),
        ],
    },
    {
        'name': 'business',
        'context': 'business operations',
        'vars': [
            ('demand', 'market demand', ['weak', 'moderate', 'strong']),
            ('inventory', 'inventory level', ['depleted', 'adequate', 'surplus']),
            ('pricing', 'pricing strategy', ['discount', 'standard', 'premium']),
            ('competition', 'competitive pressure', ['low', 'moderate', 'fierce']),
            ('retention', 'customer retention', ['declining', 'stable', 'growing']),
            ('costs', 'operational efficiency', ['poor', 'moderate', 'excellent']),
        ],
    },
    {
        'name': 'infrastructure',
        'context': 'infrastructure management',
        'vars': [
            ('utilization', 'system utilization', ['light', 'moderate', 'heavy']),
            ('hardware', 'hardware condition', ['degraded', 'fair', 'excellent']),
            ('redundancy', 'redundancy level', ['none', 'partial', 'full']),
            ('security', 'security posture', ['vulnerable', 'patched', 'hardened']),
            ('bandwidth', 'network capacity', ['constrained', 'adequate', 'abundant']),
            ('backlog', 'maintenance backlog', ['critical', 'manageable', 'clear']),
        ],
    },
    {
        'name': 'project',
        'context': 'project management',
        'vars': [
            ('velocity', 'team velocity', ['slow', 'steady', 'fast']),
            ('debt', 'technical debt', ['severe', 'moderate', 'minimal']),
            ('clarity', 'requirements clarity', ['ambiguous', 'partial', 'well-defined']),
            ('alignment', 'stakeholder alignment', ['conflicting', 'partial', 'strong']),
            ('resources', 'resource availability', ['scarce', 'adequate', 'abundant']),
            ('pressure', 'schedule pressure', ['critical', 'moderate', 'relaxed']),
        ],
    },
]

N_DOMAINS = len(DOMAINS)
N_VARS_PER_DOMAIN = 6


# ── Text Rendering ──────────────────────────────────────────────────

def _level_idx(val: float) -> int:
    """Map float in [0, 1] to level index 0 (low), 1 (mid), 2 (high)."""
    if val > 0.67:
        return 2
    if val > 0.33:
        return 1
    return 0


def _var_desc(var_def: tuple, val: float) -> str:
    """Render a single variable as 'display_name: level (value)'."""
    _, display, levels = var_def
    return f"{display}: {levels[_level_idx(val)]} ({val:.2f})"


# State templates: take (context_string, list_of_var_descriptions) → sentence
_STATE_TEMPLATES = [
    lambda ctx, parts: f"{ctx.title()} assessment — " + ", ".join(parts) + ".",
    lambda ctx, parts: f"Current {ctx} status: " + "; ".join(parts) + ".",
    lambda ctx, parts: f"Situation report ({ctx}): " + ", ".join(parts) + ".",
    lambda ctx, parts: f"{ctx.title()} snapshot — " + ", ".join(parts) + ".",
    lambda ctx, parts: f"Evaluating {ctx}: " + ", ".join(parts) + ".",
    lambda ctx, parts: f"The {ctx} state is: " + ", ".join(parts) + ".",
]

# Action templates: take (context_string, list_of_change_descriptions) → sentence
_ACTION_TEMPLATES = [
    lambda ctx, parts: f"Proposed {ctx} action: " + ", ".join(parts) + ".",
    lambda ctx, parts: f"Intervention plan — " + "; ".join(parts) + ".",
    lambda ctx, parts: f"Adjust {ctx}: " + ", ".join(parts) + ".",
    lambda ctx, parts: f"Changes to apply: " + ", ".join(parts) + ".",
    lambda ctx, parts: f"Recommended {ctx} modifications: " + ", ".join(parts) + ".",
    lambda ctx, parts: f"Action items — " + "; ".join(parts) + ".",
]

_NO_ACTION_TEMPLATES = [
    lambda ctx: f"No changes to {ctx}.",
    lambda ctx: f"Maintain current {ctx} parameters.",
    lambda ctx: f"No {ctx} interventions planned.",
    lambda ctx: f"Hold steady on {ctx}.",
]


def render_state(domain: dict, values: np.ndarray, template_idx: int = 0) -> str:
    """Render a domain state as natural language text."""
    parts = [_var_desc(var, val) for var, val in zip(domain['vars'], values)]
    template = _STATE_TEMPLATES[template_idx % len(_STATE_TEMPLATES)]
    return template(domain['context'], parts)


def render_intervention(
    domain: dict,
    old_values: np.ndarray,
    new_values: np.ndarray,
    changed_indices: List[int],
    template_idx: int = 0,
) -> str:
    """Render an intervention as natural language text."""
    if len(changed_indices) == 0:
        return _NO_ACTION_TEMPLATES[template_idx % len(_NO_ACTION_TEMPLATES)](domain['context'])

    parts = []
    for idx in changed_indices:
        _, display, levels = domain['vars'][idx]
        old_level = levels[_level_idx(old_values[idx])]
        new_level = levels[_level_idx(new_values[idx])]
        parts.append(f"set {display} from {old_level} to {new_level} ({new_values[idx]:.2f})")

    template = _ACTION_TEMPLATES[template_idx % len(_ACTION_TEMPLATES)]
    return template(domain['context'], parts)


# ── Dataset ─────────────────────────────────────────────────────────

class SelfSupervisedDataset(Dataset):
    """
    Self-supervised dataset: Transformer representation shifts as training signal.

    For each sample, generates a base scenario and an intervened variant across
    one of 5 domains, encodes both through a frozen Transformer, and provides
    the hidden state difference h_delta = h_intervened - h_base as the target.

    Returns (h_base, action, h_delta, domain_idx) per sample.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        num_samples: int = 50000,
        batch_encode_size: int = 32,
        device: str = "cuda",
        cache_path: Optional[str] = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name

        # ── Load from cache if available ─────────────────
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached self-supervised dataset from {cache_path}")
            data = torch.load(cache_path, weights_only=False)
            self.h_base = data["h_base"]
            self.actions = data["actions"]
            self.h_delta = data["h_delta"]
            self.domain_indices = data["domain_indices"]
            self.d_model = data["d_model"]
            print(f"Loaded {len(self.h_base)} samples, d_model={self.d_model}")
            return

        # ── Generate scenario/intervention pairs ─────────
        print(f"Generating {num_samples} scenario pairs across {N_DOMAINS} domains...")
        rng = np.random.RandomState(seed)

        texts_base = []
        texts_intervened = []
        texts_action = []
        domain_indices = []

        samples_per_domain = num_samples // N_DOMAINS

        for d_idx, domain in enumerate(DOMAINS):
            # Last domain absorbs any remainder from integer division
            n = samples_per_domain if d_idx < N_DOMAINS - 1 else num_samples - d_idx * samples_per_domain
            n_vars = len(domain['vars'])

            for i in range(n):
                template_idx = (d_idx * samples_per_domain + i)

                # Sample base state: all variables in [0.05, 0.95]
                base_vals = rng.uniform(0.05, 0.95, n_vars)

                # Sample intervention: change 1-3 variables
                n_change = rng.randint(1, min(4, n_vars + 1))
                changed = sorted(rng.choice(n_vars, n_change, replace=False).tolist())

                # New values for changed variables (ensure they actually change)
                new_vals = base_vals.copy()
                for idx in changed:
                    # Shift by at least 0.15 to ensure a meaningful text-level change
                    shift = rng.uniform(0.2, 0.6) * rng.choice([-1, 1])
                    new_vals[idx] = np.clip(base_vals[idx] + shift, 0.05, 0.95)

                # Render all three texts
                texts_base.append(render_state(domain, base_vals, template_idx))
                texts_intervened.append(render_state(domain, new_vals, template_idx))
                texts_action.append(render_intervention(
                    domain, base_vals, new_vals, changed, template_idx
                ))
                domain_indices.append(d_idx)

        # Shuffle across domains so batches are mixed
        order = rng.permutation(len(texts_base))
        texts_base = [texts_base[i] for i in order]
        texts_intervened = [texts_intervened[i] for i in order]
        texts_action = [texts_action[i] for i in order]
        domain_indices = [domain_indices[i] for i in order]

        print(f"  Samples per domain: ~{samples_per_domain}")
        print(f"  Example base:       {texts_base[0][:80]}...")
        print(f"  Example intervened:  {texts_intervened[0][:80]}...")
        print(f"  Example action:      {texts_action[0][:80]}...")

        # ── Encode through Transformer ───────────────────
        print(f"\nEncoding {num_samples} x 3 texts with {model_name}...")
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # float16 for large models, float32 for GPT-2
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

        def _last_token_pool(hidden_states, attention_mask):
            """Extract the last non-padding token's hidden state."""
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, seq_lengths]

        all_h_base = []
        all_h_intervened = []
        all_actions = []

        for start in tqdm(range(0, num_samples, batch_encode_size), desc="Encoding"):
            end = min(start + batch_encode_size, num_samples)

            with torch.no_grad():
                # Encode base scenarios
                b_tok = tokenizer(
                    texts_base[start:end], return_tensors="pt",
                    padding=True, truncation=True, max_length=128,
                ).to(device)
                b_out = transformer(**b_tok)
                all_h_base.append(
                    _last_token_pool(b_out.last_hidden_state, b_tok.attention_mask).float().cpu()
                )

                # Encode intervened scenarios
                i_tok = tokenizer(
                    texts_intervened[start:end], return_tensors="pt",
                    padding=True, truncation=True, max_length=128,
                ).to(device)
                i_out = transformer(**i_tok)
                all_h_intervened.append(
                    _last_token_pool(i_out.last_hidden_state, i_tok.attention_mask).float().cpu()
                )

                # Encode action descriptions
                a_tok = tokenizer(
                    texts_action[start:end], return_tensors="pt",
                    padding=True, truncation=True, max_length=64,
                ).to(device)
                a_out = transformer(**a_tok)
                all_actions.append(
                    _last_token_pool(a_out.last_hidden_state, a_tok.attention_mask).float().cpu()
                )

        # Free Transformer memory
        del transformer
        torch.cuda.empty_cache()

        self.h_base = torch.cat(all_h_base, dim=0)
        h_intervened = torch.cat(all_h_intervened, dim=0)
        self.actions = torch.cat(all_actions, dim=0)
        self.h_delta = h_intervened - self.h_base
        self.domain_indices = torch.tensor(domain_indices, dtype=torch.long)

        print(f"\nDataset ready: {len(self.h_base)} samples")
        print(f"  h_base shape:     {self.h_base.shape}")
        print(f"  action shape:     {self.actions.shape}")
        print(f"  h_delta shape:    {self.h_delta.shape}")
        print(f"  h_delta mean norm: {self.h_delta.norm(dim=-1).mean():.4f}")
        print(f"  h_delta std norm:  {self.h_delta.norm(dim=-1).std():.4f}")

        # Per-domain stats
        for d_idx, domain in enumerate(DOMAINS):
            mask = self.domain_indices == d_idx
            mn = self.h_delta[mask].norm(dim=-1).mean()
            print(f"  {domain['name']:15s} h_delta norm: {mn:.4f} ({mask.sum()} samples)")

        # ── Cache ────────────────────────────────────────
        if cache_path:
            print(f"Caching to {cache_path}")
            torch.save({
                "h_base": self.h_base,
                "actions": self.actions,
                "h_delta": self.h_delta,
                "domain_indices": self.domain_indices,
                "d_model": self.d_model,
            }, cache_path)

    def __len__(self) -> int:
        return len(self.h_base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (h_base, action, h_delta, domain_index)."""
        return self.h_base[idx], self.actions[idx], self.h_delta[idx], self.domain_indices[idx]
