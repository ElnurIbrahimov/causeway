"""
Causeway Loss Functions.

Combined loss for training the Causeway module:

    L_total = L_delta + λ_acyclic * L_acyclic + λ_sparse * L_sparse + λ_ortho * L_ortho

Where:
    L_delta:    Asymmetric prediction loss on structured deltas.
                Underestimating risk is penalized more than overestimating.

    L_acyclic:  NOTEARS acyclicity constraint.
                tr(e^{W∘W}) - d = 0 when graph is a DAG.

    L_sparse:   L1 sparsity on adjacency weights.
                Encourages minimal causal edges.

    L_ortho:    Orthogonality of the rotation matrix.
                ||R^T R - I||_F^2 → 0 for distance preservation.

The asymmetric delta loss is the key innovation for risk-sensitive
counterfactual estimation:

    L_asym(pred, target) = {
        α_over  * (pred - target)^2   if pred > target  (overestimate)
        α_under * (pred - target)^2   if pred ≤ target  (underestimate)
    }

For risk dimensions (risk_shift, constraint_violation), α_under > α_over,
so the model is penalized more for failing to predict danger.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from causeway.delta_predictor import DeltaVector, DEFAULT_DELTA_DIMS


class AsymmetricMSE(nn.Module):
    """
    Asymmetric mean squared error.

    Penalizes underestimation and overestimation differently per dimension.
    """

    def __init__(
        self,
        n_dims: int,
        overestimate_weight: float = 1.0,
        underestimate_weight: float = 2.0,
        risk_dims: Optional[list] = None,
        risk_underestimate_weight: float = 4.0,
    ):
        """
        Args:
            n_dims: Number of output dimensions.
            overestimate_weight: Base weight for overestimation errors.
            underestimate_weight: Base weight for underestimation errors.
            risk_dims: Indices of risk-related dimensions (get extra penalty).
            risk_underestimate_weight: Weight for underestimating risk dims.
        """
        super().__init__()
        self.n_dims = n_dims

        # Per-dimension weights: (n_dims, 2) for [overestimate, underestimate]
        weights = torch.full((n_dims, 2), overestimate_weight)
        weights[:, 1] = underestimate_weight

        if risk_dims:
            for idx in risk_dims:
                weights[idx, 1] = risk_underestimate_weight

        self.register_buffer("weights", weights)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted deltas, shape (batch, n_dims).
            target: Ground truth deltas, shape (batch, n_dims).

        Returns:
            Scalar loss.
        """
        error = pred - target
        sq_error = error ** 2

        # Select weight based on sign of error
        # overestimate: pred > target → error > 0 → weights[:, 0]
        # underestimate: pred ≤ target → error ≤ 0 → weights[:, 1]
        overestimate_mask = (error > 0).float()
        weights = self.weights.to(pred.device)
        w = (
            overestimate_mask * weights[:, 0]
            + (1 - overestimate_mask) * weights[:, 1]
        )

        return (w * sq_error).mean()


class ConfidenceLoss(nn.Module):
    """
    Calibration loss for confidence predictions.

    Confidence should be high when predictions are accurate
    and low when predictions are inaccurate.

    L_conf = -log(c) * 1(|error| < τ) - log(1-c) * 1(|error| ≥ τ)

    where c is predicted confidence and τ is the accuracy threshold.
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        confidence: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            confidence: Predicted confidence, shape (batch, n_dims), values in [0, 1].
            pred: Predicted deltas, shape (batch, n_dims).
            target: Ground truth deltas, shape (batch, n_dims).

        Returns:
            Scalar calibration loss.
        """
        error = (pred - target).abs()
        accurate = (error < self.threshold).float()

        # Binary cross-entropy where the "label" is whether prediction was accurate
        eps = 1e-7
        loss = -(
            accurate * torch.log(confidence + eps)
            + (1 - accurate) * torch.log(1 - confidence + eps)
        )
        return loss.mean()


class CausewayLoss(nn.Module):
    """
    Combined loss for training the full Causeway module.

    Components:
        1. Asymmetric delta prediction loss
        2. Confidence calibration loss
        3. NOTEARS acyclicity constraint
        4. L1 sparsity on causal graph
        5. Orthogonality of state encoder rotation
    """

    def __init__(
        self,
        delta_dims: Optional[list] = None,
        lambda_acyclic: float = 1.0,
        lambda_sparse: float = 0.01,
        lambda_edge_count: float = 0.1,
        lambda_ortho: float = 0.1,
        lambda_confidence: float = 0.5,
        risk_underestimate_weight: float = 4.0,
    ):
        """
        Args:
            delta_dims: Names of delta dimensions (to identify risk dims).
            lambda_acyclic: Weight for acyclicity constraint.
            lambda_sparse: Weight for L1 sparsity penalty.
            lambda_edge_count: Weight for L0 edge count penalty.
            lambda_ortho: Weight for orthogonality penalty.
            lambda_confidence: Weight for confidence calibration.
            risk_underestimate_weight: Extra penalty for underestimating risk.
        """
        super().__init__()
        self.delta_dims = delta_dims or DEFAULT_DELTA_DIMS
        n_dims = len(self.delta_dims)

        # Identify risk-related dimensions
        risk_keywords = {"risk", "constraint", "violation"}
        risk_dims = [
            i for i, name in enumerate(self.delta_dims)
            if any(kw in name.lower() for kw in risk_keywords)
        ]

        self.delta_loss = AsymmetricMSE(
            n_dims=n_dims,
            risk_dims=risk_dims,
            risk_underestimate_weight=risk_underestimate_weight,
        )
        self.confidence_loss = ConfidenceLoss()

        self.lambda_acyclic = lambda_acyclic
        self.lambda_sparse = lambda_sparse
        self.lambda_edge_count = lambda_edge_count
        self.lambda_ortho = lambda_ortho
        self.lambda_confidence = lambda_confidence

    def forward(
        self,
        delta_pred: DeltaVector,
        delta_target: torch.Tensor,
        reg_losses: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with breakdown.

        Args:
            delta_pred: Predicted DeltaVector from Causeway.
            delta_target: Ground truth delta values, shape (batch, n_dims).
            reg_losses: Dict from causeway.get_regularization_losses().

        Returns:
            Dict with 'total' and individual loss components.
        """
        # Primary: delta prediction
        l_delta = self.delta_loss(delta_pred.values, delta_target)

        # Confidence calibration
        l_conf = self.confidence_loss(
            delta_pred.confidence, delta_pred.values, delta_target
        )

        # Structural constraints
        l_acyclic = reg_losses["acyclicity"]
        l_sparse = reg_losses["sparsity"]
        l_edge_count = reg_losses["edge_count"]
        l_ortho = reg_losses["orthogonality"]

        # Total
        total = (
            l_delta
            + self.lambda_confidence * l_conf
            + self.lambda_edge_count * l_edge_count
            + self.lambda_acyclic * l_acyclic
            + self.lambda_sparse * l_sparse
            + self.lambda_ortho * l_ortho
        )

        return {
            "total": total,
            "delta": l_delta,
            "confidence": l_conf,
            "acyclicity": l_acyclic,
            "sparsity": l_sparse,
            "edge_count": l_edge_count,
            "orthogonality": l_ortho,
        }
