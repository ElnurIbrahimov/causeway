"""
Confounded Structural Causal Model Environment.

Provides ground-truth counterfactuals for training and evaluating Causeway.
Third domain designed to test causal reasoning under competing paths,
confounders, nonlinear thresholds, and Simpson's paradox.

Domain: Abstract Confounded System (neutral variable names — no semantic leakage)
    Variables:
        0. alpha        - Abstract controllable input A (Controllable)
        1. beta         - Abstract controllable input B (Controllable)
        2. gamma        - Abstract controllable input C (Controllable)
        3. delta_var    - Abstract controllable input D (Controllable)
        4. epsilon_var  - Intermediate effect variable (EFFECT)
        5. zeta         - Intermediate effect variable (EFFECT)
        6. eta          - Intermediate effect variable (EFFECT)
        7. theta        - Final outcome variable (EFFECT)

    Causal graph (known):
        alpha       -> epsilon_var  (+0.5)   # alpha has competing paths
        alpha       -> zeta         (-0.4)   # positive via epsilon, negative via zeta
        beta        -> epsilon_var  (-0.3)
        beta        -> eta          (+0.6)
        gamma       -> zeta         (+0.5)
        gamma       -> eta          (-0.3)
        epsilon_var -> theta        (+0.7)
        zeta        -> theta        (-0.5, FLIPS to +0.8 when zeta > 0.7)
        eta         -> theta        (+0.4)
        delta_var   -> theta        (-0.3, protective but weak)

    Key properties:
        - Competing paths: alpha influences theta through both epsilon_var
          (positive) and zeta (negative), making net effect state-dependent.
        - Nonlinear threshold: When zeta > 0.7, the zeta->theta coefficient
          flips from -0.5 to +0.8 (cascading failure regime).
        - Simpson's paradox: beta correlates with good outcomes observationally
          (high beta -> high eta -> good theta) but intervening on beta ALSO
          increases epsilon_var negatively, which can be net harmful.
        - Semantic opacity: An LLM reading "set alpha to 0.85" has NO way to
          determine if this is good or bad without the causal graph.

    Propagation order (valid topological sort):
        1. epsilon_var = f(alpha, beta) + eps
        2. zeta        = f(alpha, gamma) + eps
        3. eta         = f(beta, gamma) + eps
        4. theta       = f(epsilon_var, zeta, eta, delta_var) + eps
                         (with nonlinear threshold on zeta)

    Structured output mapping:
        risk_shift           <- delta[theta]
        goal_progress        <- -delta[theta]
        constraint_violation <- clip(max(delta[zeta], delta[epsilon_var]) * 3, 0, 1)
        resource_cost        <- abs(delta[epsilon_var]) + abs(delta[zeta])
        success_probability  <- -delta[theta] * 0.5 + 0.5 * (1 - clip(delta[zeta], 0, 1))
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional


# Causal variable indices
ALPHA = 0
BETA = 1
GAMMA = 2
DELTA_VAR = 3
EPSILON_VAR = 4
ZETA = 5
ETA = 6
THETA = 7

NUM_VARIABLES = 8
NUM_CONTROLLABLE = 4  # first 4 are controllable
NUM_OBSERVABLE = 4    # last 4 are effects


class ConfoundedSCM:
    """
    Ground-truth structural causal model with competing paths, confounders,
    nonlinear thresholds, and Simpson's paradox structure.

    All functional relationships are known, so we can compute exact
    counterfactuals for any intervention. Variable names are intentionally
    neutral (Greek letters) to prevent semantic leakage — an LLM cannot
    determine intervention quality from variable names alone.
    """

    def __init__(self, noise_scale: float = 0.05, seed: Optional[int] = None):
        """
        Args:
            noise_scale: Scale of exogenous noise (epsilon).
            seed: Random seed for reproducibility.
        """
        self.noise_scale = noise_scale
        self.rng = np.random.RandomState(seed)

        # Ground-truth adjacency matrix
        self.adjacency = np.zeros((NUM_VARIABLES, NUM_VARIABLES))
        self.adjacency[ALPHA, EPSILON_VAR] = 0.5
        self.adjacency[ALPHA, ZETA] = -0.4
        self.adjacency[BETA, EPSILON_VAR] = -0.3
        self.adjacency[BETA, ETA] = 0.6
        self.adjacency[GAMMA, ZETA] = 0.5
        self.adjacency[GAMMA, ETA] = -0.3
        self.adjacency[EPSILON_VAR, THETA] = 0.7
        self.adjacency[ZETA, THETA] = -0.5  # linear regime; flips at threshold
        self.adjacency[ETA, THETA] = 0.4
        self.adjacency[DELTA_VAR, THETA] = -0.3  # protective but weak

    def sample_state(self, batch_size: int = 1) -> np.ndarray:
        """
        Sample a random state from the SCM.

        Returns:
            state: shape (batch_size, NUM_VARIABLES)
        """
        eps = self.rng.randn(batch_size, NUM_VARIABLES) * self.noise_scale
        state = np.zeros((batch_size, NUM_VARIABLES))

        # Controllable variables: sample from reasonable distributions
        state[:, ALPHA] = self.rng.uniform(0.1, 1.0, batch_size)
        state[:, BETA] = self.rng.uniform(0.1, 1.0, batch_size)
        state[:, GAMMA] = self.rng.uniform(0.0, 1.0, batch_size)
        state[:, DELTA_VAR] = self.rng.uniform(0.0, 0.8, batch_size)

        # Effect variables: computed from causal parents + noise
        state = self._propagate(state, eps)
        return state

    def _propagate(self, state: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """Forward-propagate through the causal graph (topological order)."""
        s = state.copy()

        # 1. epsilon_var = f(alpha, beta) + eps
        s[:, EPSILON_VAR] = np.clip(
            0.5 * s[:, ALPHA]
            - 0.3 * s[:, BETA]
            + eps[:, EPSILON_VAR],
            0.0, 1.0,
        )

        # 2. zeta = f(alpha, gamma) + eps
        s[:, ZETA] = np.clip(
            -0.4 * s[:, ALPHA]
            + 0.5 * s[:, GAMMA]
            + eps[:, ZETA],
            0.0, 1.0,
        )

        # 3. eta = f(beta, gamma) + eps
        s[:, ETA] = np.clip(
            0.6 * s[:, BETA]
            - 0.3 * s[:, GAMMA]
            + eps[:, ETA],
            0.0, 1.0,
        )

        # 4. theta = f(epsilon_var, zeta, eta, delta_var) + eps
        #    Nonlinear threshold: when zeta > 0.7, the zeta->theta effect
        #    FLIPS from -0.5 to +0.8 (cascading failure regime).
        zeta_coeff = np.where(s[:, ZETA] > 0.7, 0.8, -0.5)
        s[:, THETA] = np.clip(
            0.7 * s[:, EPSILON_VAR]
            + zeta_coeff * s[:, ZETA]
            + 0.4 * s[:, ETA]
            - 0.3 * s[:, DELTA_VAR]
            + eps[:, THETA],
            0.0, 1.0,
        )

        return s

    def intervene(
        self,
        state: np.ndarray,
        intervention_mask: np.ndarray,
        intervention_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply an intervention and compute the counterfactual state.

        This is the ground-truth do-operator.

        Args:
            state: Current state, shape (batch, NUM_VARIABLES).
            intervention_mask: Binary mask, shape (batch, NUM_VARIABLES).
                              1 = intervene on this variable.
            intervention_values: Values to set, shape (batch, NUM_VARIABLES).

        Returns:
            counterfactual_state: State after intervention, shape (batch, NUM_VARIABLES).
            delta: counterfactual_state - state, shape (batch, NUM_VARIABLES).
        """
        # Abduct: recover noise from observed state
        eps = self._abduce(state)

        # Apply intervention
        s_int = state.copy()
        s_int = np.where(intervention_mask > 0.5, intervention_values, s_int)

        # Propagate with original noise
        s_cf = self._propagate(s_int, eps)

        # Re-apply intervention (intervened variables stay fixed)
        s_cf = np.where(intervention_mask > 0.5, intervention_values, s_cf)

        delta = s_cf - state
        return s_cf, delta

    def _abduce(self, state: np.ndarray) -> np.ndarray:
        """Recover exogenous noise from observed state (inverse of _propagate).

        Note: For the nonlinear zeta->theta relationship, abduction uses the
        same threshold logic applied to the *observed* zeta values, ensuring
        the recovered noise is consistent with the observed state.
        """
        s = state
        eps = np.zeros_like(state)

        # Reverse the functional relationships (same topological order)
        # 1. epsilon_var
        eps[:, EPSILON_VAR] = (
            s[:, EPSILON_VAR]
            - 0.5 * s[:, ALPHA]
            + 0.3 * s[:, BETA]
        )
        # 2. zeta
        eps[:, ZETA] = (
            s[:, ZETA]
            + 0.4 * s[:, ALPHA]
            - 0.5 * s[:, GAMMA]
        )
        # 3. eta
        eps[:, ETA] = (
            s[:, ETA]
            - 0.6 * s[:, BETA]
            + 0.3 * s[:, GAMMA]
        )
        # 4. theta (must use same nonlinear threshold as _propagate)
        zeta_coeff = np.where(s[:, ZETA] > 0.7, 0.8, -0.5)
        eps[:, THETA] = (
            s[:, THETA]
            - 0.7 * s[:, EPSILON_VAR]
            - zeta_coeff * s[:, ZETA]
            - 0.4 * s[:, ETA]
            + 0.3 * s[:, DELTA_VAR]
        )

        return eps

    def compute_structured_delta(self, delta: np.ndarray) -> np.ndarray:
        """
        Map raw variable deltas to structured output dimensions.

        Output dims:
            0. risk_shift           = delta[THETA]
            1. goal_progress        = -delta[THETA]
            2. constraint_violation = clip(max(delta[ZETA], delta[EPSILON_VAR]) * 3, 0, 1)
            3. resource_cost        = abs(delta[EPSILON_VAR]) + abs(delta[ZETA])
            4. success_probability  = -delta[THETA] * 0.5 + 0.5 * (1 - clip(delta[ZETA], 0, 1))

        Args:
            delta: Raw state delta, shape (batch, NUM_VARIABLES).

        Returns:
            structured: shape (batch, 5).
        """
        batch_size = delta.shape[0]
        structured = np.zeros((batch_size, 5))

        # risk_shift: theta is the outcome — higher theta = more risk
        structured[:, 0] = delta[:, THETA]

        # goal_progress: reducing theta = progress
        structured[:, 1] = -delta[:, THETA]

        # constraint_violation: soft threshold on worst intermediate effect
        structured[:, 2] = np.clip(
            np.maximum(delta[:, ZETA], delta[:, EPSILON_VAR]) * 3,
            0, 1,
        )

        # resource_cost: interventions that move intermediates have costs
        structured[:, 3] = (
            np.abs(delta[:, EPSILON_VAR]) + np.abs(delta[:, ZETA])
        )

        # success_probability: composite of outcome improvement and zeta safety
        structured[:, 4] = (
            -delta[:, THETA] * 0.5
            + 0.5 * (1.0 - np.clip(delta[:, ZETA], 0, 1))
        )

        return structured

    def get_adjacency_tensor(self) -> torch.Tensor:
        """Return the ground-truth adjacency as a torch tensor."""
        return torch.tensor(self.adjacency, dtype=torch.float32)


class ConfoundedSCMDataset(Dataset):
    """
    PyTorch Dataset that generates (state, action, target_delta) triples
    from the confounded SCM.

    Each sample:
        - state: a sampled SCM state (serves as simulated "Transformer hidden state")
        - action: a random intervention on controllable variables
        - target_delta: the ground-truth structured delta from the SCM

    This provides perfect supervision for training Causeway.
    """

    def __init__(
        self,
        scm: ConfoundedSCM,
        num_samples: int = 10000,
        d_model: int = 64,
        d_action: int = 32,
        seed: Optional[int] = None,
    ):
        """
        Args:
            scm: The ground-truth SCM.
            num_samples: Number of samples to generate.
            d_model: Dimension to project states into (simulates Transformer dim).
            d_action: Dimension to project actions into.
            seed: Random seed.
        """
        super().__init__()
        # Use a dedicated rng for projections (consumed first, always same)
        # and a separate rng for data generation (depends on num_samples)
        proj_rng = np.random.RandomState(seed)
        data_rng = np.random.RandomState(seed + 1000 if seed is not None else None)

        # Create random projections FIRST so they're deterministic regardless
        # of num_samples. This ensures the same seed always produces the same
        # projection space (simulating a consistent frozen Transformer).
        self.state_projection = proj_rng.randn(NUM_VARIABLES, d_model).astype(np.float32) * 0.1
        self.action_projection = proj_rng.randn(NUM_CONTROLLABLE, d_action).astype(np.float32) * 0.1
        action_proj = proj_rng.randn(NUM_CONTROLLABLE * 2, d_action).astype(np.float32) * 0.1

        # Generate all data upfront
        states = scm.sample_state(num_samples)

        # Random interventions on controllable variables
        intervention_mask = np.zeros((num_samples, NUM_VARIABLES))
        intervention_values = np.zeros((num_samples, NUM_VARIABLES))

        for i in range(num_samples):
            # Randomly choose 1-3 variables to intervene on
            n_intervene = data_rng.randint(1, NUM_CONTROLLABLE)
            targets = data_rng.choice(NUM_CONTROLLABLE, n_intervene, replace=False)
            intervention_mask[i, targets] = 1.0
            intervention_values[i, targets] = data_rng.uniform(0.0, 1.0, n_intervene)

        # Compute ground-truth counterfactuals
        _, raw_deltas = scm.intervene(states, intervention_mask, intervention_values)
        structured_deltas = scm.compute_structured_delta(raw_deltas)

        # Project states into "Transformer hidden state" space
        self.h = torch.tensor(
            states @ self.state_projection, dtype=torch.float32
        )

        # Project interventions into "action" space
        # Action = concat of mask and values for controllable vars, projected
        action_raw = np.concatenate([
            intervention_mask[:, :NUM_CONTROLLABLE],
            intervention_values[:, :NUM_CONTROLLABLE],
        ], axis=1)
        self.actions = torch.tensor(
            action_raw @ action_proj, dtype=torch.float32
        )

        # Ground truth structured deltas
        self.targets = torch.tensor(structured_deltas, dtype=torch.float32)

        # Store raw data for analysis
        self.raw_states = torch.tensor(states, dtype=torch.float32)
        self.raw_deltas = torch.tensor(raw_deltas, dtype=torch.float32)
        self.intervention_masks = torch.tensor(intervention_mask, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.h)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (hidden_state, action, target_delta)."""
        return self.h[idx], self.actions[idx], self.targets[idx]
