"""
Synthetic Structural Causal Model Environment.

Provides ground-truth counterfactuals for training and evaluating Causeway.

The environment defines a known causal graph with known functional relationships.
For any state and intervention, we can compute the exact counterfactual outcome,
giving us perfect supervision signal.

Domain: Software Deployment System
    Variables:
        0. code_complexity     - Complexity of the change being deployed
        1. test_coverage       - Fraction of code covered by tests
        2. deploy_load         - Current system load at deployment time
        3. rollback_readiness  - How prepared the rollback procedure is
        4. error_rate          - Post-deployment error rate (EFFECT)
        5. latency_impact      - Change in system latency (EFFECT)
        6. user_impact         - Impact on user experience (EFFECT)
        7. resource_usage      - Change in compute/memory usage (EFFECT)

    Causal graph (known):
        code_complexity → error_rate
        code_complexity → latency_impact
        code_complexity → resource_usage
        test_coverage → error_rate (protective)
        deploy_load → latency_impact
        deploy_load → error_rate
        rollback_readiness → user_impact (protective)
        error_rate → user_impact
        latency_impact → user_impact
        resource_usage → latency_impact

    Actions: Interventions on the first 4 variables (things you can control).
    Deltas: Changes in the last 4 variables (things you observe).

    Structured output mapping:
        risk_shift           ← error_rate delta
        goal_progress        ← negative of user_impact delta (less impact = progress)
        constraint_violation ← 1 if any variable exceeds threshold
        resource_cost        ← resource_usage delta
        success_probability  ← derived from error_rate + user_impact
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict


# Causal variable indices
CODE_COMPLEXITY = 0
TEST_COVERAGE = 1
DEPLOY_LOAD = 2
ROLLBACK_READINESS = 3
ERROR_RATE = 4
LATENCY_IMPACT = 5
USER_IMPACT = 6
RESOURCE_USAGE = 7

NUM_VARIABLES = 8
NUM_CONTROLLABLE = 4  # first 4 are controllable
NUM_OBSERVABLE = 4    # last 4 are effects


class SyntheticSCM:
    """
    Ground-truth structural causal model for a software deployment system.

    All functional relationships are known, so we can compute exact
    counterfactuals for any intervention.
    """

    def __init__(self, noise_scale: float = 0.05, seed: Optional[int] = None):
        """
        Args:
            noise_scale: Scale of exogenous noise (ε).
            seed: Random seed for reproducibility.
        """
        self.noise_scale = noise_scale
        self.rng = np.random.RandomState(seed)

        # Ground-truth adjacency matrix (for reference/evaluation)
        self.adjacency = np.zeros((NUM_VARIABLES, NUM_VARIABLES))
        self.adjacency[CODE_COMPLEXITY, ERROR_RATE] = 0.6
        self.adjacency[CODE_COMPLEXITY, LATENCY_IMPACT] = 0.3
        self.adjacency[CODE_COMPLEXITY, RESOURCE_USAGE] = 0.4
        self.adjacency[TEST_COVERAGE, ERROR_RATE] = -0.5  # protective
        self.adjacency[DEPLOY_LOAD, LATENCY_IMPACT] = 0.5
        self.adjacency[DEPLOY_LOAD, ERROR_RATE] = 0.3
        self.adjacency[ROLLBACK_READINESS, USER_IMPACT] = -0.4  # protective
        self.adjacency[ERROR_RATE, USER_IMPACT] = 0.7
        self.adjacency[LATENCY_IMPACT, USER_IMPACT] = 0.5
        self.adjacency[RESOURCE_USAGE, LATENCY_IMPACT] = 0.3

    def sample_state(self, batch_size: int = 1) -> np.ndarray:
        """
        Sample a random state from the SCM.

        Returns:
            state: shape (batch_size, NUM_VARIABLES)
        """
        eps = self.rng.randn(batch_size, NUM_VARIABLES) * self.noise_scale
        state = np.zeros((batch_size, NUM_VARIABLES))

        # Controllable variables: sample from reasonable distributions
        state[:, CODE_COMPLEXITY] = self.rng.uniform(0.1, 1.0, batch_size)
        state[:, TEST_COVERAGE] = self.rng.uniform(0.2, 1.0, batch_size)
        state[:, DEPLOY_LOAD] = self.rng.uniform(0.0, 1.0, batch_size)
        state[:, ROLLBACK_READINESS] = self.rng.uniform(0.3, 1.0, batch_size)

        # Effect variables: computed from causal parents + noise
        state = self._propagate(state, eps)
        return state

    def _propagate(self, state: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """Forward-propagate through the causal graph (topological order)."""
        s = state.copy()

        # resource_usage = f(code_complexity) + ε
        s[:, RESOURCE_USAGE] = (
            0.4 * s[:, CODE_COMPLEXITY]
            + eps[:, RESOURCE_USAGE]
        )

        # error_rate = f(code_complexity, test_coverage, deploy_load) + ε
        s[:, ERROR_RATE] = np.clip(
            0.6 * s[:, CODE_COMPLEXITY]
            - 0.5 * s[:, TEST_COVERAGE]
            + 0.3 * s[:, DEPLOY_LOAD]
            + eps[:, ERROR_RATE],
            0.0, 1.0,
        )

        # latency_impact = f(code_complexity, deploy_load, resource_usage) + ε
        s[:, LATENCY_IMPACT] = (
            0.3 * s[:, CODE_COMPLEXITY]
            + 0.5 * s[:, DEPLOY_LOAD]
            + 0.3 * s[:, RESOURCE_USAGE]
            + eps[:, LATENCY_IMPACT]
        )

        # user_impact = f(error_rate, latency_impact, rollback_readiness) + ε
        s[:, USER_IMPACT] = np.clip(
            0.7 * s[:, ERROR_RATE]
            + 0.5 * s[:, LATENCY_IMPACT]
            - 0.4 * s[:, ROLLBACK_READINESS]
            + eps[:, USER_IMPACT],
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
        """Recover exogenous noise from observed state (inverse of _propagate)."""
        s = state
        eps = np.zeros_like(state)

        # Reverse the functional relationships
        eps[:, RESOURCE_USAGE] = (
            s[:, RESOURCE_USAGE] - 0.4 * s[:, CODE_COMPLEXITY]
        )
        eps[:, ERROR_RATE] = (
            s[:, ERROR_RATE]
            - 0.6 * s[:, CODE_COMPLEXITY]
            + 0.5 * s[:, TEST_COVERAGE]
            - 0.3 * s[:, DEPLOY_LOAD]
        )
        eps[:, LATENCY_IMPACT] = (
            s[:, LATENCY_IMPACT]
            - 0.3 * s[:, CODE_COMPLEXITY]
            - 0.5 * s[:, DEPLOY_LOAD]
            - 0.3 * s[:, RESOURCE_USAGE]
        )
        eps[:, USER_IMPACT] = (
            s[:, USER_IMPACT]
            - 0.7 * s[:, ERROR_RATE]
            - 0.5 * s[:, LATENCY_IMPACT]
            + 0.4 * s[:, ROLLBACK_READINESS]
        )

        return eps

    def compute_structured_delta(self, delta: np.ndarray) -> np.ndarray:
        """
        Map raw variable deltas to structured output dimensions.

        Output dims:
            0. risk_shift           = delta[ERROR_RATE]
            1. goal_progress        = -delta[USER_IMPACT]
            2. constraint_violation = max(0, delta[ERROR_RATE]) > 0.2 (soft)
            3. resource_cost        = delta[RESOURCE_USAGE]
            4. success_probability  = -(delta[ERROR_RATE] + delta[USER_IMPACT]) / 2

        Args:
            delta: Raw state delta, shape (batch, NUM_VARIABLES).

        Returns:
            structured: shape (batch, 5).
        """
        batch_size = delta.shape[0]
        structured = np.zeros((batch_size, 5))

        structured[:, 0] = delta[:, ERROR_RATE]                  # risk_shift
        structured[:, 1] = -delta[:, USER_IMPACT]                # goal_progress
        structured[:, 2] = np.clip(delta[:, ERROR_RATE] * 3, 0, 1)  # constraint_violation (soft)
        structured[:, 3] = delta[:, RESOURCE_USAGE]              # resource_cost
        structured[:, 4] = -(
            delta[:, ERROR_RATE] + delta[:, USER_IMPACT]
        ) / 2  # success_probability change

        return structured

    def get_adjacency_tensor(self) -> torch.Tensor:
        """Return the ground-truth adjacency as a torch tensor."""
        return torch.tensor(self.adjacency, dtype=torch.float32)


class SCMDataset(Dataset):
    """
    PyTorch Dataset that generates (state, action, target_delta) triples.

    Each sample:
        - state: a sampled SCM state (serves as simulated "Transformer hidden state")
        - action: a random intervention on controllable variables
        - target_delta: the ground-truth structured delta from the SCM

    This provides perfect supervision for training Causeway.
    """

    def __init__(
        self,
        scm: SyntheticSCM,
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
