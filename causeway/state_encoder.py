"""
State Encoder: Projects Transformer hidden states into causal variable space.

Mathematical formulation:
    Given a Transformer hidden state h ∈ R^{d_model}, we learn a rotation
    matrix R ∈ R^{d_model x d_causal} that extracts causally meaningful
    subspaces from the distributed representation.

    z = R^T h + b

    where z ∈ R^{d_causal} is the causal variable vector.

    R is kept approximately orthogonal via a soft constraint:
        L_ortho = ||R^T R - I||_F^2

    This is inspired by Distributed Alignment Search (Geiger, Wu et al. 2023),
    which finds causal variables in distributed neural representations via
    learned rotation matrices.

    The encoder also learns a nonlinear refinement after the rotation to
    capture variable interactions not expressible as pure linear projections:

    z = MLP(R^T h)

Parameter budget: ~2-5M parameters depending on d_model and d_causal.

V2 changes (capacity scaling):
    The refinement MLP's hidden_dim now defaults to max(4 * d_causal, d_model // 2)
    instead of 2 * d_causal. For GPT-2 (d_model=768, d_causal=48), this means
    384 instead of 96 — enough capacity to decode the 768-dim rotation output.

    A third linear layer is added: d_causal → hidden → hidden → d_causal.
    This provides the depth needed for the MLP to learn non-trivial variable
    interactions from the high-dimensional Transformer representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StateEncoder(nn.Module):
    """Projects Transformer hidden states to causal variable space."""

    def __init__(
        self,
        d_model: int,
        d_causal: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Dimension of the Transformer hidden state.
            d_causal: Number of causal variables to extract.
            hidden_dim: Hidden dimension for the refinement MLP.
                        Defaults to max(4 * d_causal, d_model // 2) to ensure
                        sufficient capacity for decoding high-dim representations.
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.d_model = d_model
        self.d_causal = d_causal
        # Scale hidden dim relative to both d_causal and d_model so the MLP
        # has enough capacity to decode the 768-dim rotation output.
        hidden_dim = hidden_dim or max(4 * d_causal, d_model // 2)

        # Learned rotation matrix: extracts causal subspace from h
        # Initialized near-orthogonal via QR decomposition
        raw = torch.randn(d_model, d_causal)
        Q, _ = torch.linalg.qr(raw)
        self.rotation = nn.Parameter(Q[:, :d_causal].clone())

        # 3-layer nonlinear refinement MLP
        # Deeper network captures variable interactions beyond linear projection.
        # Architecture: d_causal → hidden → hidden → d_causal
        self.refine = nn.Sequential(
            nn.Linear(d_causal, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_causal),
        )

        # Layer norm on causal variables for stable downstream processing
        self.norm = nn.LayerNorm(d_causal)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Project Transformer hidden state to causal variable space.

        Args:
            h: Transformer hidden state, shape (batch, d_model)
               or (batch, seq_len, d_model).

        Returns:
            z: Causal variables, shape (batch, d_causal)
               or (batch, seq_len, d_causal).
        """
        # Ensure fp32 (Mistral/LLaMA output fp16 hidden states)
        h = h.float()

        # Linear rotation into causal subspace
        z = h @ self.rotation  # (..., d_causal)

        # Nonlinear refinement with residual connection
        z = z + self.refine(z)

        # Normalize
        z = self.norm(z)

        return z

    def orthogonality_loss(self) -> torch.Tensor:
        """
        Soft orthogonality constraint on the rotation matrix.

        Returns ||R^T R - I||_F^2

        This encourages the rotation to preserve distances and
        extract independent causal directions.
        """
        RtR = self.rotation.T @ self.rotation
        I = torch.eye(self.d_causal, device=self.rotation.device)
        return torch.norm(RtR - I, p="fro") ** 2
