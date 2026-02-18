"""
Causal Graph Layer: Learns and maintains a sparse directed acyclic graph
over causal variables, with differentiable message passing.

Mathematical formulation:

    The causal graph is parameterized by a weighted adjacency matrix
    W ∈ R^{d_causal x d_causal}, where W_ij != 0 means variable i
    directly causes variable j.

    Acyclicity constraint (NOTEARS, Zheng et al. 2018):
        h(W) = tr(e^{W ∘ W}) - d = 0

    Sparsity mechanism (v2): Gumbel-sigmoid edge gating.
        Each edge has a learnable logit α_ij. During training, edges are
        sampled via Gumbel-sigmoid with temperature τ:

            g_ij = σ((α_ij + Gumbel(0,1) - Gumbel(0,1)) / τ)

        As τ → 0, this approaches hard binary selection.
        During eval, edges are hard-thresholded: g_ij = 1(σ(α_ij) > 0.5).

        The effective adjacency is: W_eff = g ∘ W_raw

    This achieves genuinely sparse graphs rather than many weak edges.

    Additional: L0 penalty on expected edge count:
        L_edges = Σ_ij σ(α_ij)  (expected number of active edges)

Parameter budget: ~3-10M parameters depending on d_causal and num_layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CausalGraphLayer(nn.Module):
    """Sparse DAG over causal variables with Gumbel-sigmoid edge gating."""

    def __init__(
        self,
        d_causal: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        initial_temperature: float = 1.0,
        edge_prior: float = -2.0,
    ):
        """
        Args:
            d_causal: Number of causal variables (graph nodes).
            num_layers: Number of message-passing layers.
            dropout: Dropout rate on edges during training.
            initial_temperature: Starting Gumbel-sigmoid temperature.
            edge_prior: Initial logit for edge gates (negative = prior toward no edge).
        """
        super().__init__()
        self.d_causal = d_causal
        self.num_layers = num_layers

        # Raw adjacency weights (edge strengths)
        self.W_raw = nn.Parameter(torch.randn(d_causal, d_causal) * 0.01)

        # Edge gate logits — initialized negative to bias toward sparsity
        # α_ij: probability of edge i→j existing = σ(α_ij)
        self.edge_logits = nn.Parameter(
            torch.full((d_causal, d_causal), edge_prior)
        )

        # Temperature for Gumbel-sigmoid (annealed during training)
        self.register_buffer("temperature", torch.tensor(initial_temperature))

        # Per-layer bias and scaling for message passing
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d_causal)) for _ in range(num_layers)
        ])
        self.scales = nn.ParameterList([
            nn.Parameter(torch.ones(d_causal)) for _ in range(num_layers)
        ])

        # Edge-level nonlinearity parameters (per layer)
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_causal, d_causal * 2),
                nn.GELU(),
                nn.Linear(d_causal * 2, d_causal),
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_causal)

        # Diagonal mask: no self-causation
        self.register_buffer(
            "diag_mask",
            1.0 - torch.eye(d_causal),
        )

    def _gumbel_sigmoid(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Gumbel-sigmoid: differentiable approximation to Bernoulli sampling.

        During training: soft samples with temperature annealing.
        During eval: hard threshold at 0.5.
        """
        if self.training:
            # Sample Gumbel noise
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gumbel_noise = torch.log(u) - torch.log(1 - u)
            return torch.sigmoid((logits + gumbel_noise) / self.temperature)
        else:
            return (torch.sigmoid(logits) > 0.5).float()

    @property
    def edge_gates(self) -> torch.Tensor:
        """Gumbel-sigmoid gates, shape (d_causal, d_causal)."""
        gates = self._gumbel_sigmoid(self.edge_logits)
        return gates * self.diag_mask  # no self-loops

    @property
    def adjacency(self) -> torch.Tensor:
        """
        Effective adjacency matrix: gated and masked.

        Returns:
            W: shape (d_causal, d_causal), sparse gated adjacency.
        """
        return self.edge_gates * self.W_raw * self.diag_mask

    @property
    def adjacency_probs(self) -> torch.Tensor:
        """Edge existence probabilities (no Gumbel noise)."""
        return torch.sigmoid(self.edge_logits) * self.diag_mask

    def set_temperature(self, temperature: float):
        """Set the Gumbel-sigmoid temperature (call during training to anneal)."""
        self.temperature.fill_(temperature)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward message passing through the causal graph.

        Args:
            z: Causal variables, shape (batch, d_causal).

        Returns:
            z_out: Updated causal variables after message passing,
                   shape (batch, d_causal).
        """
        W = self.adjacency
        h = z

        for i in range(self.num_layers):
            messages = h @ W  # (batch, d_causal)
            messages = self.edge_mlps[i](messages)
            messages = messages * self.scales[i] + self.biases[i]
            h = h + self.dropout(F.gelu(messages))

        return self.norm(h)

    def acyclicity_loss(self) -> torch.Tensor:
        """
        NOTEARS acyclicity constraint on the gated adjacency.
        h(W) = tr(e^{W ∘ W}) - d
        """
        W = self.adjacency
        W_sq = W * W
        d = self.d_causal
        I = torch.eye(d, device=W.device)
        M = I
        term = I
        for k in range(1, d + 1):
            term = term @ W_sq / k
            M = M + term
        return torch.trace(M) - d

    def sparsity_loss(self) -> torch.Tensor:
        """L1 penalty on gated adjacency weights."""
        return torch.sum(torch.abs(self.adjacency))

    def edge_count_loss(self) -> torch.Tensor:
        """
        L0-style penalty: expected number of active edges.
        Minimizing this directly pressures the graph toward fewer edges.
        """
        return self.adjacency_probs.sum()

    def get_graph_stats(self) -> dict:
        """Return interpretable statistics about the learned graph."""
        with torch.no_grad():
            probs = self.adjacency_probs
            hard_edges = (probs > 0.5).sum().item()
            soft_edges = probs.sum().item()
            density = hard_edges / (self.d_causal * (self.d_causal - 1))
            max_prob = probs.max().item()
            max_weight = (self.W_raw * self.diag_mask).abs().max().item()
            return {
                "hard_edges": int(hard_edges),
                "expected_edges": f"{soft_edges:.1f}",
                "density": round(density, 4),
                "max_edge_prob": round(max_prob, 4),
                "max_weight": round(max_weight, 6),
                "acyclicity": round(self.acyclicity_loss().item(), 6),
                "temperature": round(self.temperature.item(), 4),
            }

    def get_top_edges(self, top_k: int = 20) -> list:
        """Return top edges by probability."""
        with torch.no_grad():
            probs = self.adjacency_probs
            weights = (self.W_raw * self.diag_mask).abs()
            d = self.d_causal
            edges = []
            for i in range(d):
                for j in range(d):
                    if i != j and probs[i, j] > 0.1:
                        edges.append((
                            i, j,
                            round(probs[i, j].item(), 4),
                            round(weights[i, j].item(), 6),
                        ))
            edges.sort(key=lambda x: -x[2])
            return edges[:top_k]
