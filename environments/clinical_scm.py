"""
Clinical Treatment Structural Causal Model Environment.

Provides ground-truth counterfactuals for training and evaluating Causeway.
Second domain to prove architecture generalization beyond software deployment.

Domain: Clinical Treatment System
    Variables:
        0. drug_dosage           - Medication dose level (Controllable)
        1. treatment_intensity   - Aggressiveness of treatment protocol (Controllable)
        2. monitoring_frequency  - How often patient vitals are checked (Controllable)
        3. activity_restriction  - How much physical activity is restricted (Controllable)
        4. adverse_events        - Side effects and complications (EFFECT)
        5. recovery_rate         - How fast patient improves (EFFECT)
        6. organ_stress          - Load on major organs (EFFECT)
        7. care_cost             - Resource consumption (EFFECT)

    Causal graph (known):
        drug_dosage         -> adverse_events    (+0.5)
        drug_dosage         -> recovery_rate     (+0.4)
        drug_dosage         -> organ_stress      (+0.6)
        treatment_intensity -> recovery_rate     (+0.5)
        treatment_intensity -> adverse_events    (+0.3)
        monitoring_freq     -> adverse_events    (-0.4, protective)
        activity_restrict   -> recovery_rate     (-0.3)
        adverse_events      -> organ_stress      (+0.5)
        organ_stress        -> recovery_rate     (-0.4)
        adverse_events      -> care_cost         (+0.6)

    Propagation order (valid DAG):
        1. adverse_events  = f(drug_dosage, treatment_intensity, monitoring_frequency)
        2. organ_stress    = f(drug_dosage, adverse_events)
        3. care_cost       = f(adverse_events)
        4. recovery_rate   = f(drug_dosage, treatment_intensity, activity_restriction, organ_stress)

    Structured output mapping:
        risk_shift           <- adverse_events delta
        goal_progress        <- recovery_rate delta
        constraint_violation <- clip(organ_stress delta * 3, 0, 1)
        resource_cost        <- care_cost delta
        success_probability  <- (recovery_rate - adverse_events) / 2
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional


# Causal variable indices
DRUG_DOSAGE = 0
TREATMENT_INTENSITY = 1
MONITORING_FREQUENCY = 2
ACTIVITY_RESTRICTION = 3
ADVERSE_EVENTS = 4
RECOVERY_RATE = 5
ORGAN_STRESS = 6
CARE_COST = 7

NUM_VARIABLES = 8
NUM_CONTROLLABLE = 4
NUM_OBSERVABLE = 4


class ClinicalSCM:
    """
    Ground-truth structural causal model for a clinical treatment system.

    All functional relationships are known, so we can compute exact
    counterfactuals for any intervention.
    """

    def __init__(self, noise_scale: float = 0.05, seed: Optional[int] = None):
        self.noise_scale = noise_scale
        self.rng = np.random.RandomState(seed)

        # Ground-truth adjacency matrix
        self.adjacency = np.zeros((NUM_VARIABLES, NUM_VARIABLES))
        self.adjacency[DRUG_DOSAGE, ADVERSE_EVENTS] = 0.5
        self.adjacency[DRUG_DOSAGE, RECOVERY_RATE] = 0.4
        self.adjacency[DRUG_DOSAGE, ORGAN_STRESS] = 0.6
        self.adjacency[TREATMENT_INTENSITY, RECOVERY_RATE] = 0.5
        self.adjacency[TREATMENT_INTENSITY, ADVERSE_EVENTS] = 0.3
        self.adjacency[MONITORING_FREQUENCY, ADVERSE_EVENTS] = -0.4  # protective
        self.adjacency[ACTIVITY_RESTRICTION, RECOVERY_RATE] = -0.3
        self.adjacency[ADVERSE_EVENTS, ORGAN_STRESS] = 0.5
        self.adjacency[ORGAN_STRESS, RECOVERY_RATE] = -0.4
        self.adjacency[ADVERSE_EVENTS, CARE_COST] = 0.6

    def sample_state(self, batch_size: int = 1) -> np.ndarray:
        eps = self.rng.randn(batch_size, NUM_VARIABLES) * self.noise_scale
        state = np.zeros((batch_size, NUM_VARIABLES))

        # Controllable variables
        state[:, DRUG_DOSAGE] = self.rng.uniform(0.1, 0.9, batch_size)
        state[:, TREATMENT_INTENSITY] = self.rng.uniform(0.1, 1.0, batch_size)
        state[:, MONITORING_FREQUENCY] = self.rng.uniform(0.2, 1.0, batch_size)
        state[:, ACTIVITY_RESTRICTION] = self.rng.uniform(0.0, 0.8, batch_size)

        # Effect variables: computed from causal parents + noise
        state = self._propagate(state, eps)
        return state

    def _propagate(self, state: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """Forward-propagate through the causal graph (topological order)."""
        s = state.copy()

        # 1. adverse_events = f(drug_dosage, treatment_intensity, monitoring_frequency) + eps
        s[:, ADVERSE_EVENTS] = np.clip(
            0.5 * s[:, DRUG_DOSAGE]
            + 0.3 * s[:, TREATMENT_INTENSITY]
            - 0.4 * s[:, MONITORING_FREQUENCY]
            + eps[:, ADVERSE_EVENTS],
            0.0, 1.0,
        )

        # 2. organ_stress = f(drug_dosage, adverse_events) + eps
        s[:, ORGAN_STRESS] = np.clip(
            0.6 * s[:, DRUG_DOSAGE]
            + 0.5 * s[:, ADVERSE_EVENTS]
            + eps[:, ORGAN_STRESS],
            0.0, 1.0,
        )

        # 3. care_cost = f(adverse_events) + eps
        s[:, CARE_COST] = (
            0.6 * s[:, ADVERSE_EVENTS]
            + eps[:, CARE_COST]
        )

        # 4. recovery_rate = f(drug_dosage, treatment_intensity, activity_restriction, organ_stress) + eps
        s[:, RECOVERY_RATE] = np.clip(
            0.4 * s[:, DRUG_DOSAGE]
            + 0.5 * s[:, TREATMENT_INTENSITY]
            - 0.3 * s[:, ACTIVITY_RESTRICTION]
            - 0.4 * s[:, ORGAN_STRESS]
            + eps[:, RECOVERY_RATE],
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

        Args:
            state: Current state, shape (batch, NUM_VARIABLES).
            intervention_mask: Binary mask, shape (batch, NUM_VARIABLES).
            intervention_values: Values to set, shape (batch, NUM_VARIABLES).

        Returns:
            counterfactual_state: State after intervention.
            delta: counterfactual_state - state.
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

        # Reverse the functional relationships (same topological order)
        # 1. adverse_events
        eps[:, ADVERSE_EVENTS] = (
            s[:, ADVERSE_EVENTS]
            - 0.5 * s[:, DRUG_DOSAGE]
            - 0.3 * s[:, TREATMENT_INTENSITY]
            + 0.4 * s[:, MONITORING_FREQUENCY]
        )
        # 2. organ_stress
        eps[:, ORGAN_STRESS] = (
            s[:, ORGAN_STRESS]
            - 0.6 * s[:, DRUG_DOSAGE]
            - 0.5 * s[:, ADVERSE_EVENTS]
        )
        # 3. care_cost
        eps[:, CARE_COST] = (
            s[:, CARE_COST]
            - 0.6 * s[:, ADVERSE_EVENTS]
        )
        # 4. recovery_rate
        eps[:, RECOVERY_RATE] = (
            s[:, RECOVERY_RATE]
            - 0.4 * s[:, DRUG_DOSAGE]
            - 0.5 * s[:, TREATMENT_INTENSITY]
            + 0.3 * s[:, ACTIVITY_RESTRICTION]
            + 0.4 * s[:, ORGAN_STRESS]
        )

        return eps

    def compute_structured_delta(self, delta: np.ndarray) -> np.ndarray:
        """
        Map raw variable deltas to structured output dimensions.

        Output dims:
            0. risk_shift           = delta[ADVERSE_EVENTS]
            1. goal_progress        = delta[RECOVERY_RATE]
            2. constraint_violation = clip(delta[ORGAN_STRESS] * 3, 0, 1)
            3. resource_cost        = delta[CARE_COST]
            4. success_probability  = (delta[RECOVERY_RATE] - delta[ADVERSE_EVENTS]) / 2
        """
        batch_size = delta.shape[0]
        structured = np.zeros((batch_size, 5))

        structured[:, 0] = delta[:, ADVERSE_EVENTS]                         # risk_shift
        structured[:, 1] = delta[:, RECOVERY_RATE]                          # goal_progress
        structured[:, 2] = np.clip(delta[:, ORGAN_STRESS] * 3, 0, 1)       # constraint_violation
        structured[:, 3] = delta[:, CARE_COST]                              # resource_cost
        structured[:, 4] = (
            delta[:, RECOVERY_RATE] - delta[:, ADVERSE_EVENTS]
        ) / 2                                                               # success_probability

        return structured

    def get_adjacency_tensor(self) -> torch.Tensor:
        """Return the ground-truth adjacency as a torch tensor."""
        return torch.tensor(self.adjacency, dtype=torch.float32)


class ClinicalSCMDataset(Dataset):
    """
    PyTorch Dataset that generates (state, action, target_delta) triples
    from the clinical treatment SCM.
    """

    def __init__(
        self,
        scm: ClinicalSCM,
        num_samples: int = 10000,
        d_model: int = 64,
        d_action: int = 32,
        seed: Optional[int] = None,
    ):
        super().__init__()
        proj_rng = np.random.RandomState(seed)
        data_rng = np.random.RandomState(seed + 1000 if seed is not None else None)

        self.state_projection = proj_rng.randn(NUM_VARIABLES, d_model).astype(np.float32) * 0.1
        self.action_projection = proj_rng.randn(NUM_CONTROLLABLE, d_action).astype(np.float32) * 0.1
        action_proj = proj_rng.randn(NUM_CONTROLLABLE * 2, d_action).astype(np.float32) * 0.1

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

        self.h = torch.tensor(
            states @ self.state_projection, dtype=torch.float32
        )

        action_raw = np.concatenate([
            intervention_mask[:, :NUM_CONTROLLABLE],
            intervention_values[:, :NUM_CONTROLLABLE],
        ], axis=1)
        self.actions = torch.tensor(
            action_raw @ action_proj, dtype=torch.float32
        )

        self.targets = torch.tensor(structured_deltas, dtype=torch.float32)
        self.raw_states = torch.tensor(states, dtype=torch.float32)
        self.raw_deltas = torch.tensor(raw_deltas, dtype=torch.float32)
        self.intervention_masks = torch.tensor(intervention_mask, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.h)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.h[idx], self.actions[idx], self.targets[idx]
