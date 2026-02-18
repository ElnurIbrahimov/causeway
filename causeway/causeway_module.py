"""
Causeway: The composed module.

Wires StateEncoder → CausalGraphLayer → InterventionEngine → DeltaPredictor
into a single nn.Module with a clean forward pass.

Usage:
    causeway = Causeway(d_model=768, d_causal=32, d_action=64)

    # h: Transformer hidden state (batch, d_model)
    # action: candidate action representation (batch, d_action)
    delta = causeway(h, action)

    # delta.values: (batch, 5) structured shifts
    # delta.confidence: (batch, 5) per-dimension confidence
    # delta.to_dict(0) -> {"risk_shift": 0.23, "goal_progress": -0.11, ...}
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict

from causeway.state_encoder import StateEncoder
from causeway.causal_graph import CausalGraphLayer
from causeway.intervention_engine import InterventionEngine
from causeway.delta_predictor import DeltaPredictor, DeltaVector


class Causeway(nn.Module):
    """
    Lightweight Counterfactual World Interface.

    A parameter-efficient causal adapter that takes a Transformer's
    hidden state and a candidate action, and returns a structured
    delta vector describing what would change under intervention.
    """

    def __init__(
        self,
        d_model: int,
        d_causal: int = 32,
        d_action: int = 64,
        graph_layers: int = 2,
        propagation_steps: int = 3,
        delta_dims: Optional[List[str]] = None,
        dropout: float = 0.1,
        encoder_hidden_dim: Optional[int] = None,
        intervention_hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            d_model: Dimension of the Transformer hidden state.
            d_causal: Number of causal variables (graph nodes).
            d_action: Dimension of the action representation.
            graph_layers: Number of message-passing layers in causal graph.
            propagation_steps: Depth of intervention effect propagation.
            delta_dims: Names of structured output dimensions.
            dropout: Dropout rate throughout.
            encoder_hidden_dim: Hidden dim for StateEncoder refinement MLP.
                If None, auto-scales to max(4*d_causal, d_model//2).
            intervention_hidden_dim: Hidden dim for InterventionEngine action encoder.
                If None, auto-scales to max(4*d_causal, d_action//2).
        """
        super().__init__()
        self.d_model = d_model
        self.d_causal = d_causal
        self.d_action = d_action
        self.propagation_steps = propagation_steps

        # Component 1: Project Transformer state → causal variable space
        self.state_encoder = StateEncoder(
            d_model=d_model,
            d_causal=d_causal,
            hidden_dim=encoder_hidden_dim,  # None → auto-scale inside StateEncoder
            dropout=dropout,
        )

        # Component 2: Learned causal graph over variables
        self.causal_graph = CausalGraphLayer(
            d_causal=d_causal,
            num_layers=graph_layers,
            dropout=dropout,
        )

        # Component 3: do-operator for counterfactual inference
        self.intervention_engine = InterventionEngine(
            d_causal=d_causal,
            d_action=d_action,
            hidden_dim=intervention_hidden_dim,  # None → auto-scale inside InterventionEngine
        )

        # Component 4: Raw delta → structured output
        self.delta_predictor = DeltaPredictor(
            d_causal=d_causal,
            delta_dims=delta_dims,
            dropout=dropout,
        )

    def forward(
        self,
        h: torch.Tensor,
        action: torch.Tensor,
    ) -> DeltaVector:
        """
        Full Causeway forward pass.

        Args:
            h: Transformer hidden state, shape (batch, d_model).
            action: Candidate action representation, shape (batch, d_action).

        Returns:
            DeltaVector containing structured counterfactual deltas.
        """
        # 1. Encode: Transformer state → causal variables
        z = self.state_encoder(h)

        # 2. Graph processing: refine through causal structure
        z_refined = self.causal_graph(z)

        # 3. Intervene: apply counterfactual via do-operator
        adjacency = self.causal_graph.adjacency
        z_counterfactual, mask, epsilon = self.intervention_engine(
            z_refined,
            action,
            adjacency,
            num_propagation_steps=self.propagation_steps,
        )

        # 4. Delta: compute structured output
        delta = self.delta_predictor(z_refined, z_counterfactual)

        return delta

    def get_diagnostics(self) -> Dict:
        """Return diagnostic info about the module's internal state."""
        graph_stats = self.causal_graph.get_graph_stats()
        n_params = sum(p.numel() for p in self.parameters())
        return {
            "total_parameters": n_params,
            "total_parameters_human": f"{n_params / 1e6:.2f}M",
            "graph": graph_stats,
            "orthogonality_loss": self.state_encoder.orthogonality_loss().item(),
        }

    def get_regularization_losses(self) -> Dict[str, torch.Tensor]:
        """Return all regularization loss terms for the training loop."""
        return {
            "acyclicity": self.causal_graph.acyclicity_loss(),
            "sparsity": self.causal_graph.sparsity_loss(),
            "edge_count": self.causal_graph.edge_count_loss(),
            "orthogonality": self.state_encoder.orthogonality_loss(),
        }
