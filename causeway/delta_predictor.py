"""
Delta Predictor: Computes structured counterfactual deltas.

Mathematical formulation:

    Given pre-intervention state z and post-intervention state z',
    the raw delta is:

        δ_raw = z' - z

    This raw delta in causal variable space is then projected to a
    structured output space of interpretable dimensions:

        δ_structured = f_project(δ_raw, z, z')

    Output dimensions (configurable, defaults):
        - risk_shift:              How much risk increases/decreases
        - goal_progress:           Change in progress toward goals
        - constraint_violation:    Probability of violating constraints
        - resource_cost:           Change in resource consumption
        - success_probability:     Change in likelihood of success

    The projection is context-dependent: the same raw delta may have
    different structured interpretations depending on the current state
    (e.g., a small shift near a constraint boundary is more dangerous
    than the same shift far from any boundary).

    Loss function uses asymmetric weighting: underestimating risk is
    penalized more heavily than overestimating it.

Parameter budget: ~4-12M parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from dataclasses import dataclass


# Default output dimensions for the delta vector
DEFAULT_DELTA_DIMS = [
    "risk_shift",
    "goal_progress",
    "constraint_violation",
    "resource_cost",
    "success_probability",
]


@dataclass
class DeltaVector:
    """Structured output from the Delta Predictor."""
    values: torch.Tensor       # (batch, n_dims) raw values
    dim_names: List[str]       # names of each dimension
    confidence: torch.Tensor   # (batch, n_dims) confidence per dimension

    def to_dict(self, batch_idx: int = 0) -> Dict[str, float]:
        """Convert a single batch element to a named dictionary."""
        return {
            name: self.values[batch_idx, i].item()
            for i, name in enumerate(self.dim_names)
        }

    def to_dict_with_confidence(self, batch_idx: int = 0) -> Dict[str, dict]:
        """Convert with confidence scores."""
        return {
            name: {
                "value": self.values[batch_idx, i].item(),
                "confidence": self.confidence[batch_idx, i].item(),
            }
            for i, name in enumerate(self.dim_names)
        }


class DeltaPredictor(nn.Module):
    """Maps raw causal deltas to structured, interpretable output."""

    def __init__(
        self,
        d_causal: int,
        delta_dims: Optional[List[str]] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_causal: Number of causal variables.
            delta_dims: Names of structured output dimensions.
                        Defaults to DEFAULT_DELTA_DIMS.
            hidden_dim: Hidden dimension for projection MLP.
                        Defaults to 4 * d_causal.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_causal = d_causal
        self.delta_dims = delta_dims or DEFAULT_DELTA_DIMS
        self.n_dims = len(self.delta_dims)
        hidden_dim = hidden_dim or 4 * d_causal

        # Input: concat(z_pre, z_post, delta_raw) = 3 * d_causal
        input_dim = 3 * d_causal

        # Context-dependent projection
        # Uses both states AND the delta for context-aware mapping
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Value head: predicted delta for each structured dimension
        self.value_head = nn.Linear(hidden_dim, self.n_dims)

        # Confidence head: epistemic uncertainty per dimension
        # Higher = more confident in the prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, self.n_dims),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_pre: torch.Tensor,
        z_post: torch.Tensor,
    ) -> DeltaVector:
        """
        Compute structured delta from pre/post intervention states.

        Args:
            z_pre: Pre-intervention causal variables, (batch, d_causal).
            z_post: Post-intervention causal variables, (batch, d_causal).

        Returns:
            DeltaVector with structured predictions and confidence.
        """
        delta_raw = z_post - z_pre

        # Context-dependent projection
        x = torch.cat([z_pre, z_post, delta_raw], dim=-1)
        h = self.projector(x)

        # Predict structured deltas and confidence
        values = self.value_head(h)
        confidence = self.confidence_head(h)

        return DeltaVector(
            values=values,
            dim_names=self.delta_dims,
            confidence=confidence,
        )
