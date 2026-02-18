"""
Intervention Engine: Implements Pearl's do-operator for counterfactual inference.

Mathematical formulation:

    Given an observed state z ∈ R^{d_causal} and a candidate action a,
    the intervention engine performs the three-step counterfactual procedure
    from structural causal models (Pearl 2009, DeepSCM Pawlowski et al. 2020):

    Step 1 — Abduction:
        Infer the exogenous noise variables ε that explain the observed state
        under the current causal graph.

        ε = f_abduce(z, W)

        In our implementation, ε is the residual not explained by the
        causal parents: ε_j = z_j - Σ_i W_ij * z_i

    Step 2 — Action (Intervention):
        Apply the do-operator. For intervened variables I:
            z'_j = a_j          if j ∈ I    (set to intervention value)
            z'_j = z_j          if j ∉ I    (keep observed value)

        The action encoder maps the raw action representation to
        (intervention_mask, intervention_values).

    Step 3 — Prediction:
        Forward-propagate through the causal graph with the intervention
        applied, using the abducted noise to maintain consistency:

        z''_j = Σ_i W_ij * z'_i + ε_j    if j ∉ I
        z''_j = a_j                         if j ∈ I

        Intervened variables are fixed; non-intervened variables update
        based on their causal parents (which may have changed due to
        the intervention) plus their original noise.

Parameter budget: ~1-3M parameters.

V2 changes (capacity scaling):
    The action encoder's hidden_dim now defaults to max(4 * d_causal, d_action // 2)
    instead of 2 * d_causal. For d_action=768 (full GPT-2 embeddings) and d_causal=48,
    this means 384 instead of 96 — matching the information content of the input.

    LayerNorm is added between action encoder layers for training stability when
    processing high-dimensional (768-dim) action embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class InterventionEngine(nn.Module):
    """Pearl's do-operator: abduction-action-prediction for counterfactuals."""

    def __init__(
        self,
        d_causal: int,
        d_action: int,
        hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            d_causal: Number of causal variables.
            d_action: Dimension of the action representation.
            hidden_dim: Hidden dim for action encoder.
                        Defaults to max(4 * d_causal, d_action // 2) to match
                        the information content of full-dim action embeddings.
        """
        super().__init__()
        self.d_causal = d_causal
        self.d_action = d_action
        # Scale hidden dim relative to both d_causal and d_action for adequate
        # capacity when decoding full 768-dim Transformer action embeddings.
        hidden_dim = hidden_dim or max(4 * d_causal, d_action // 2)

        # Action encoder: maps action to (which variables to intervene, what values)
        # LayerNorm between layers stabilizes training with high-dim inputs.
        self.action_encoder = nn.Sequential(
            nn.Linear(d_action, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Intervention mask head: which variables does this action affect?
        # Output: logits for each causal variable (sigmoid → soft mask)
        self.mask_head = nn.Linear(hidden_dim, d_causal)

        # Intervention value head: what values does the action set?
        self.value_head = nn.Linear(hidden_dim, d_causal)

    def encode_action(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map an action representation to intervention targets.

        Args:
            action: Action representation, shape (batch, d_action).

        Returns:
            mask: Soft intervention mask, shape (batch, d_causal).
                  Values in [0, 1]: 1 = fully intervened, 0 = untouched.
            values: Intervention values, shape (batch, d_causal).
        """
        h = self.action_encoder(action)
        mask = torch.sigmoid(self.mask_head(h))
        values = self.value_head(h)
        return mask, values

    def abduce(
        self, z: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Step 1 — Abduction: Infer exogenous noise from observed state.

        ε_j = z_j - Σ_i W_ij * z_i

        The noise is what the causal parents don't explain.

        Args:
            z: Observed causal variables, shape (batch, d_causal).
            adjacency: Causal adjacency matrix, shape (d_causal, d_causal).

        Returns:
            epsilon: Exogenous noise, shape (batch, d_causal).
        """
        # z_predicted_by_parents[j] = sum_i W_ij * z_i
        z_from_parents = z @ adjacency  # (batch, d_causal)
        epsilon = z - z_from_parents
        return epsilon

    def intervene(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step 2 — Action: Apply the do-operator.

        z'_j = mask_j * values_j + (1 - mask_j) * z_j

        Soft interpolation allows gradient flow through the mask.

        Args:
            z: Pre-intervention causal variables, shape (batch, d_causal).
            mask: Intervention mask, shape (batch, d_causal).
            values: Intervention values, shape (batch, d_causal).

        Returns:
            z_intervened: Post-intervention variables, shape (batch, d_causal).
        """
        return mask * values + (1.0 - mask) * z

    def propagate(
        self,
        z_intervened: torch.Tensor,
        epsilon: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 3,
    ) -> torch.Tensor:
        """
        Step 3 — Prediction: Forward-propagate intervention effects.

        Iteratively updates non-intervened variables based on their
        causal parents (which may have changed) plus abducted noise.

        Multiple steps allow effects to cascade through the graph.

        Args:
            z_intervened: State after do-operator, shape (batch, d_causal).
            epsilon: Abducted exogenous noise, shape (batch, d_causal).
            adjacency: Causal adjacency matrix, shape (d_causal, d_causal).
            mask: Intervention mask, shape (batch, d_causal).
            num_steps: Number of propagation steps (depth of causal chain).

        Returns:
            z_final: Post-propagation state, shape (batch, d_causal).
        """
        z = z_intervened

        for _ in range(num_steps):
            # Compute what parents predict for each variable
            z_from_parents = z @ adjacency  # (batch, d_causal)

            # Non-intervened variables update; intervened stay fixed
            z_updated = z_from_parents + epsilon
            z = mask * z_intervened + (1.0 - mask) * z_updated

        return z

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        adjacency: torch.Tensor,
        num_propagation_steps: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full counterfactual inference: abduction → action → prediction.

        Args:
            z: Observed causal variables, shape (batch, d_causal).
            action: Action representation, shape (batch, d_action).
            adjacency: Causal adjacency matrix, shape (d_causal, d_causal).
            num_propagation_steps: Depth of effect propagation.

        Returns:
            z_counterfactual: State under intervention, shape (batch, d_causal).
            mask: Intervention mask (which variables were affected).
            epsilon: Abducted noise (for inspection/debugging).
        """
        # Encode action into intervention targets
        mask, values = self.encode_action(action)

        # Step 1: Abduction
        epsilon = self.abduce(z, adjacency)

        # Step 2: Intervention
        z_intervened = self.intervene(z, mask, values)

        # Step 3: Propagation
        z_counterfactual = self.propagate(
            z_intervened, epsilon, adjacency, mask, num_propagation_steps
        )

        return z_counterfactual, mask, epsilon
