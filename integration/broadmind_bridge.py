"""
BroadMind-Causeway Integration Bridge
======================================

Connects two complementary systems:
  - Causeway:  "What would change if I did X?"  (causal counterfactual reasoning)
  - BroadMind: "How do I execute this program?"  (latent program execution)

Three integration modules:

1. CausalWisdomBridge
   Converts Causeway's learned causal graph into BroadMind wisdom codes.
   The causal structure becomes compressed procedural knowledge that
   guides BroadMind's latent program generation.

2. AdaptiveInterventionEngine
   Replaces Causeway's fixed 3-step intervention propagation with
   BroadMind-style adaptive compute. Simple interventions propagate
   in 1 step; complex causal chains get up to max_depth steps.
   Uses a learned halter to decide when propagation has converged.

3. CausalProgramExecutor
   The joint module. Uses Causeway to understand causal consequences
   of candidate actions, then feeds those consequences as wisdom
   signals into BroadMind's solver. BroadMind generates and executes
   latent programs that are causally informed — it knows what will
   change before it acts.

Dimensions (happy coincidence):
  - Causeway d_causal = 48
  - BroadMind d_wisdom = 48
  Both use 48-dim compressed representations, making the bridge natural.

Usage:
    # Standalone: causal graph -> wisdom
    bridge = CausalWisdomBridge(d_causal=48, d_wisdom=48)
    wisdom = bridge(adjacency, causal_vars)

    # Standalone: adaptive intervention propagation
    engine = AdaptiveInterventionEngine(d_causal=48, d_action=768)
    z_cf, mask, eps, steps = engine(z, action, adjacency)

    # Joint: full causal program executor
    executor = CausalProgramExecutor(causeway, broadmind, d_model=768)
    result = executor(h, programs, initial_states)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


# ============================================================================
# OUTPUT CONTAINERS
# ============================================================================

@dataclass
class CausalExecutionResult:
    """Output of the joint CausalProgramExecutor."""
    predictions: torch.Tensor        # (batch, n_steps, n_vars) BroadMind state predictions
    delta_values: torch.Tensor       # (batch, 5) Causeway structured deltas
    delta_confidence: torch.Tensor   # (batch, 5) per-dimension confidence
    causal_wisdom: torch.Tensor      # (batch, d_wisdom) wisdom derived from causal graph
    matched_wisdom: torch.Tensor     # (batch, d_wisdom) BroadMind's matched wisdom
    fused_wisdom: torch.Tensor       # (batch, d_wisdom) blended causal + matched wisdom
    compute_cost: float              # BroadMind compute cost
    propagation_steps: Optional[int] # steps used by adaptive intervention (if used)


# ============================================================================
# MODULE 1: CAUSAL WISDOM BRIDGE
# ============================================================================

class CausalWisdomBridge(nn.Module):
    """
    Converts Causeway's causal graph structure into BroadMind wisdom codes.

    Causeway learns a sparse DAG (adjacency matrix) over causal variables.
    This module compresses that structural knowledge into a 48-dim wisdom
    code that BroadMind's solver can consume.

    The bridge answers: "Given what the causal graph says about how variables
    relate, what procedural knowledge should guide program execution?"

    Architecture:
        adjacency (d_causal x d_causal) -> flatten + causal vars
        -> MLP -> 48-dim wisdom code (same space as BroadMind's WisdomBank)
    """

    def __init__(
        self,
        d_causal: int = 48,
        d_wisdom: int = 48,
        hidden_dim: int = 128,
        use_graph_stats: bool = True,
    ):
        """
        Args:
            d_causal: Number of causal variables (Causeway's graph size).
            d_wisdom: Wisdom code dimension (BroadMind's d_wisdom).
            hidden_dim: Hidden layer size.
            use_graph_stats: If True, also feed graph statistics (degree,
                             density, spectral features) for richer encoding.
        """
        super().__init__()
        self.d_causal = d_causal
        self.d_wisdom = d_wisdom
        self.use_graph_stats = use_graph_stats

        # Graph structure encoder: learns to read the adjacency pattern
        # Input: per-variable features from the adjacency matrix
        # Each variable gets its row (incoming edges) + column (outgoing edges)
        self.row_encoder = nn.Sequential(
            nn.Linear(d_causal, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Causal variable encoder: the current state in causal space
        self.var_encoder = nn.Sequential(
            nn.Linear(d_causal, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Graph-level statistics (optional enrichment)
        n_stats = 4 if use_graph_stats else 0  # degree_mean, degree_std, density, spectral_gap

        # Fusion: combines graph structure + variable state (+ optional stats) -> wisdom
        fusion_in = hidden_dim // 2 + hidden_dim // 2 + n_stats
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_wisdom),
            nn.Tanh(),  # match BroadMind's wisdom compressor (bounded output)
        )

    def _graph_stats(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Extract scalar graph statistics from adjacency matrix."""
        # Soft edge weights (adjacency may be continuous from Gumbel-sigmoid)
        in_degree = adjacency.sum(dim=0)   # (d_causal,)
        out_degree = adjacency.sum(dim=1)  # (d_causal,)
        total_degree = in_degree + out_degree

        degree_mean = total_degree.mean()
        degree_std = total_degree.std()
        density = adjacency.sum() / (self.d_causal * (self.d_causal - 1))

        # Spectral gap: difference between top 2 singular values
        # Indicates how "hub-like" the graph structure is
        svs = torch.linalg.svdvals(adjacency)
        spectral_gap = svs[0] - svs[1] if svs.shape[0] > 1 else svs[0]

        return torch.stack([degree_mean, degree_std, density, spectral_gap])

    def forward(
        self,
        adjacency: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert causal graph + state into a wisdom code.

        Args:
            adjacency: Causal adjacency matrix, shape (d_causal, d_causal).
                       From Causeway's CausalGraphLayer.adjacency property.
            z: Causal variables, shape (batch, d_causal).
               From Causeway's StateEncoder output.

        Returns:
            wisdom: Wisdom code, shape (batch, d_wisdom).
                    Compatible with BroadMind's wisdom input format.
        """
        batch_size = z.shape[0]

        # Encode graph structure: mean-pool row encodings
        # Each row of the adjacency = incoming edge pattern for that variable
        row_enc = self.row_encoder(adjacency)  # (d_causal, hidden//2)
        graph_enc = row_enc.mean(dim=0)        # (hidden//2,) — graph summary
        graph_enc = graph_enc.unsqueeze(0).expand(batch_size, -1)  # (batch, hidden//2)

        # Encode causal variable state
        var_enc = self.var_encoder(z)  # (batch, hidden//2)

        # Fuse structure + state (+ optional graph stats)
        if self.use_graph_stats:
            stats = self._graph_stats(adjacency)  # (4,)
            stats = stats.unsqueeze(0).expand(batch_size, -1)  # (batch, 4)
            fused = torch.cat([graph_enc, var_enc, stats], dim=-1)
        else:
            fused = torch.cat([graph_enc, var_enc], dim=-1)

        wisdom = self.fusion(fused)  # (batch, d_wisdom)
        return wisdom


# ============================================================================
# MODULE 2: ADAPTIVE INTERVENTION ENGINE
# ============================================================================

class AdaptiveInterventionEngine(nn.Module):
    """
    BroadMind-style adaptive compute for Causeway's intervention propagation.

    Replaces the fixed `num_propagation_steps=3` in Causeway's
    InterventionEngine.propagate() with a learned halter that decides
    when causal effects have fully propagated through the graph.

    Simple interventions (affecting a leaf variable) converge in 1 step.
    Complex interventions (affecting a root with long causal chains)
    may need up to max_depth steps.

    This mirrors BroadMind's Halter concept: don't waste compute on
    problems that are already solved.

    Architecture:
        Same abduction-action-prediction pipeline as Causeway, but the
        prediction step iterates with a convergence halter instead of
        running a fixed number of steps.
    """

    def __init__(
        self,
        d_causal: int,
        d_action: int,
        hidden_dim: Optional[int] = None,
        max_depth: int = 6,
        halt_threshold: float = 0.5,
    ):
        """
        Args:
            d_causal: Number of causal variables.
            d_action: Dimension of action representation.
            hidden_dim: Hidden dim for action encoder.
            max_depth: Maximum propagation steps (safety bound).
            halt_threshold: Halter sigmoid threshold for stopping.
        """
        super().__init__()
        self.d_causal = d_causal
        self.max_depth = max_depth
        self.halt_threshold = halt_threshold

        hidden_dim = hidden_dim or max(4 * d_causal, d_action // 2)

        # Action encoder (same as Causeway's InterventionEngine)
        self.action_encoder = nn.Sequential(
            nn.Linear(d_action, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.mask_head = nn.Linear(hidden_dim, d_causal)
        self.value_head = nn.Linear(hidden_dim, d_causal)

        # Convergence halter: decides when propagation is done
        # Input: current state delta (how much changed this step) + step encoding
        halter_in = d_causal + d_causal // 4  # delta_norm + step_enc
        self.halter = nn.Sequential(
            nn.Linear(halter_in, d_causal),
            nn.GELU(),
            nn.Linear(d_causal, 1),
        )

    def _sinusoidal_enc(self, step: int, d: int, batch_size: int, device) -> torch.Tensor:
        """Sinusoidal encoding for propagation step (no learnable params)."""
        position = torch.tensor([step], dtype=torch.float, device=device)
        div_term = torch.exp(
            torch.arange(0, d, 2, device=device).float() * -(math.log(10000.0) / d)
        )
        pe = torch.zeros(1, d, device=device)
        pe[0, 0::2] = torch.sin(position * div_term[:pe[0, 0::2].shape[0]])
        pe[0, 1::2] = torch.cos(position * div_term[:pe[0, 1::2].shape[0]])
        return pe.expand(batch_size, -1)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Adaptive counterfactual inference.

        Args:
            z: Observed causal variables, shape (batch, d_causal).
            action: Action representation, shape (batch, d_action).
            adjacency: Causal adjacency matrix, shape (d_causal, d_causal).

        Returns:
            z_counterfactual: Post-intervention state, (batch, d_causal).
            mask: Intervention mask, (batch, d_causal).
            epsilon: Abducted noise, (batch, d_causal).
            steps_used: Number of propagation steps taken.
        """
        batch_size = z.shape[0]

        # Encode action -> intervention targets
        h = self.action_encoder(action)
        mask = torch.sigmoid(self.mask_head(h))
        values = self.value_head(h)

        # Abduction: infer exogenous noise
        z_from_parents = z @ adjacency
        epsilon = z - z_from_parents

        # Intervention: apply do-operator
        z_state = mask * values + (1.0 - mask) * z

        # Adaptive propagation with halter
        steps_used = 0
        for step in range(self.max_depth):
            z_prev = z_state

            # One propagation step
            z_from_parents = z_state @ adjacency
            z_updated = z_from_parents + epsilon
            z_state = mask * (mask * values + (1.0 - mask) * z) + (1.0 - mask) * z_updated

            steps_used += 1

            # Check convergence (only at eval; train always runs max_depth)
            if not self.training:
                delta = (z_state - z_prev).abs()
                step_enc = self._sinusoidal_enc(
                    step, self.d_causal // 4, batch_size, z.device
                )
                halt_input = torch.cat([delta, step_enc], dim=-1)
                halt_prob = torch.sigmoid(self.halter(halt_input)).squeeze(-1)

                if halt_prob.mean() > self.halt_threshold:
                    break

        return z_state, mask, epsilon, steps_used


# ============================================================================
# MODULE 3: CAUSAL PROGRAM EXECUTOR (Joint Module)
# ============================================================================

class CausalProgramExecutor(nn.Module):
    """
    The joint Causeway + BroadMind system.

    Flow:
        1. Transformer hidden state h enters Causeway
        2. Causeway extracts causal variables z, learns the causal graph,
           and predicts counterfactual deltas for a candidate action
        3. CausalWisdomBridge converts the causal graph into a wisdom code
        4. The causal wisdom is fused with BroadMind's matched wisdom
        5. BroadMind's solver executes programs using the fused wisdom

    The result: BroadMind generates latent programs that are causally
    informed. It knows what will change (from Causeway) before it
    decides how to act (via its own latent program induction).

    Parameter budget:
        Causeway:           ~794K
        BroadMind:          ~447K
        Bridge + fusion:    ~40K
        Total:              ~1.28M
    """

    def __init__(
        self,
        causeway: nn.Module,
        broadmind: nn.Module,
        d_model: int = 768,
        d_causal: int = 48,
        d_wisdom: int = 48,
        d_action: Optional[int] = None,
        fusion_mode: str = "gate",
    ):
        """
        Args:
            causeway: Trained Causeway module (or None to skip causal reasoning).
            broadmind: Trained BroadMind module (BroadMindV077 or similar).
            d_model: Transformer hidden state dimension.
            d_causal: Causeway's causal variable count.
            d_wisdom: BroadMind's wisdom code dimension.
            d_action: Action representation dimension (defaults to d_model).
            fusion_mode: How to combine causal and matched wisdom.
                "gate"    - learned gating (default, most flexible)
                "add"     - simple addition
                "replace" - use only causal wisdom (ignore BroadMind's matcher)
        """
        super().__init__()
        self.causeway = causeway
        self.broadmind = broadmind
        self.d_model = d_model
        self.d_causal = d_causal
        self.d_wisdom = d_wisdom
        self.d_action = d_action or d_model
        self.fusion_mode = fusion_mode

        # Causal graph -> wisdom bridge
        self.causal_bridge = CausalWisdomBridge(
            d_causal=d_causal,
            d_wisdom=d_wisdom,
        )

        # Delta vector -> supplementary wisdom signal
        # Causeway's 5 delta dims + 5 confidence dims = 10 -> d_wisdom
        self.delta_to_wisdom = nn.Sequential(
            nn.Linear(10, d_wisdom),
            nn.GELU(),
            nn.Linear(d_wisdom, d_wisdom),
            nn.Tanh(),
        )

        # Wisdom fusion
        if fusion_mode == "gate":
            # Learned gate: how much to trust causal vs matched wisdom
            # Input: causal_wisdom + matched_wisdom -> gate in [0,1]
            self.gate = nn.Sequential(
                nn.Linear(d_wisdom * 2, d_wisdom),
                nn.Sigmoid(),
            )

    def fuse_wisdom(
        self,
        causal_wisdom: torch.Tensor,
        matched_wisdom: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine causal wisdom (from Causeway's graph) with BroadMind's
        matched wisdom (from its WisdomBank).

        Args:
            causal_wisdom: (batch, d_wisdom) from CausalWisdomBridge
            matched_wisdom: (batch, d_wisdom) from BroadMind's WisdomMatcher

        Returns:
            fused: (batch, d_wisdom) blended wisdom
        """
        if self.fusion_mode == "replace":
            return causal_wisdom
        elif self.fusion_mode == "add":
            return causal_wisdom + matched_wisdom
        elif self.fusion_mode == "gate":
            gate_input = torch.cat([causal_wisdom, matched_wisdom], dim=-1)
            g = self.gate(gate_input)  # (batch, d_wisdom), values in [0,1]
            return g * causal_wisdom + (1 - g) * matched_wisdom
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

    def forward(
        self,
        h: torch.Tensor,
        action: torch.Tensor,
        programs: torch.Tensor,
        initial_states: torch.Tensor,
        width_mult: float = 1.0,
        depth_mask: Optional[torch.Tensor] = None,
    ) -> CausalExecutionResult:
        """
        Full causal program execution pipeline.

        Args:
            h: Transformer hidden state, (batch, d_model).
                Source of causal information. Pass through Causeway.
            action: Candidate action representation, (batch, d_action).
                What we're considering doing.
            programs: Operation indices, (batch, n_steps).
                BroadMind program to execute.
            initial_states: Starting state, (batch, n_variables).
                BroadMind initial state.
            width_mult: Elastic width for BroadMind solver.
            depth_mask: Elastic depth mask for BroadMind solver.

        Returns:
            CausalExecutionResult with predictions, deltas, and wisdom info.
        """
        # === Stage 1: Causal reasoning (Causeway) ===
        delta = self.causeway(h, action)

        # Extract causal internals for the bridge
        z = self.causeway.state_encoder(h)
        adjacency = self.causeway.causal_graph.adjacency

        # === Stage 2: Build causal wisdom ===
        # From graph structure
        graph_wisdom = self.causal_bridge(adjacency, z)

        # From delta predictions (what will change)
        delta_input = torch.cat([delta.values, delta.confidence], dim=-1)
        delta_wisdom = self.delta_to_wisdom(delta_input)

        # Combine both causal signals
        causal_wisdom = graph_wisdom + delta_wisdom  # additive in wisdom space

        # === Stage 3: Get BroadMind's own wisdom ===
        matched_wisdom, _ = self.broadmind.get_wisdom(programs, initial_states)

        # === Stage 4: Fuse causal + procedural wisdom ===
        fused = self.fuse_wisdom(causal_wisdom, matched_wisdom)

        # === Stage 5: Execute with BroadMind (using fused wisdom) ===
        preds, states, compute_cost = self.broadmind.solver(
            programs, initial_states, fused,
            width_mult=width_mult,
            depth_mask=depth_mask,
        )

        return CausalExecutionResult(
            predictions=preds,
            delta_values=delta.values,
            delta_confidence=delta.confidence,
            causal_wisdom=causal_wisdom,
            matched_wisdom=matched_wisdom,
            fused_wisdom=fused,
            compute_cost=compute_cost,
            propagation_steps=None,
        )

    def forward_causal_only(
        self,
        h: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run only the Causeway side. Useful for evaluating causal
        predictions without executing a program.

        Returns dict with delta values, confidence, and causal wisdom.
        """
        delta = self.causeway(h, action)
        z = self.causeway.state_encoder(h)
        adjacency = self.causeway.causal_graph.adjacency
        wisdom = self.causal_bridge(adjacency, z)

        return {
            "delta_values": delta.values,
            "delta_confidence": delta.confidence,
            "causal_wisdom": wisdom,
            "adjacency": adjacency,
        }

    def forward_broadmind_only(
        self,
        programs: torch.Tensor,
        initial_states: torch.Tensor,
        width_mult: float = 1.0,
    ) -> torch.Tensor:
        """
        Run only BroadMind with its own wisdom (no causal input).
        Useful for ablation studies comparing with/without causal wisdom.
        """
        return self.broadmind.forward_all_steps(
            programs, initial_states,
            width_mult=width_mult,
        )


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class CausalExecutorLoss(nn.Module):
    """
    Joint loss for the CausalProgramExecutor.

    Combines:
        1. BroadMind's task loss (state prediction MSE)
        2. Causeway's delta loss (counterfactual accuracy)
        3. Wisdom alignment loss (causal wisdom should be useful)
        4. Causeway's structural regularization (DAG, sparsity, ortho)
    """

    def __init__(
        self,
        lambda_task: float = 1.0,
        lambda_delta: float = 1.0,
        lambda_alignment: float = 0.1,
        lambda_structural: float = 0.01,
    ):
        super().__init__()
        self.lambda_task = lambda_task
        self.lambda_delta = lambda_delta
        self.lambda_alignment = lambda_alignment
        self.lambda_structural = lambda_structural

    def forward(
        self,
        result: CausalExecutionResult,
        target_states: torch.Tensor,
        target_deltas: Optional[torch.Tensor],
        causeway: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute joint loss.

        Args:
            result: CausalExecutionResult from the executor.
            target_states: Ground-truth intermediate states, (batch, n_steps, n_vars).
            target_deltas: Ground-truth counterfactual deltas, (batch, 5) or None.
            causeway: Causeway module (for regularization losses).

        Returns:
            Dict of named loss components + total.
        """
        losses = {}

        # 1. Task loss: BroadMind state prediction accuracy
        losses["task"] = F.mse_loss(result.predictions, target_states)

        # 2. Delta loss: Causeway counterfactual accuracy (if targets available)
        if target_deltas is not None:
            losses["delta"] = F.mse_loss(result.delta_values, target_deltas)
        else:
            losses["delta"] = torch.tensor(0.0, device=result.predictions.device)

        # 3. Wisdom alignment: fused wisdom should differ from matched wisdom
        # (otherwise the causal bridge isn't contributing anything)
        # Minimize negative cosine similarity = maximize contribution
        cos_sim = F.cosine_similarity(result.fused_wisdom, result.matched_wisdom, dim=-1)
        losses["alignment"] = cos_sim.mean()  # penalize being identical

        # 4. Structural regularization from Causeway
        reg = causeway.get_regularization_losses()
        losses["structural"] = sum(reg.values())

        # Total
        losses["total"] = (
            self.lambda_task * losses["task"]
            + self.lambda_delta * losses["delta"]
            + self.lambda_alignment * losses["alignment"]
            + self.lambda_structural * losses["structural"]
        )

        return losses


# ============================================================================
# FACTORY: convenient construction
# ============================================================================

def build_causal_executor(
    causeway_checkpoint: str,
    broadmind_checkpoint: str,
    d_model: int = 768,
    d_causal: int = 48,
    d_wisdom: int = 48,
    d_action: Optional[int] = None,
    fusion_mode: str = "gate",
    device: str = "cpu",
) -> CausalProgramExecutor:
    """
    Build a CausalProgramExecutor from saved checkpoints.

    Args:
        causeway_checkpoint: Path to Causeway .pt checkpoint.
        broadmind_checkpoint: Path to BroadMind .pt checkpoint.
        d_model: Transformer hidden state dim (768 for GPT-2).
        d_causal: Causeway causal variable count.
        d_wisdom: BroadMind wisdom code dimension.
        d_action: Action representation dim (defaults to d_model).
        fusion_mode: Wisdom fusion strategy ("gate", "add", "replace").
        device: Target device.

    Returns:
        CausalProgramExecutor ready for inference or fine-tuning.
    """
    import sys
    import os

    d_action = d_action or d_model

    # Load Causeway
    # Assumes causeway package is importable
    from causeway import Causeway
    causeway = Causeway(
        d_model=d_model,
        d_causal=d_causal,
        d_action=d_action,
    )
    ckpt = torch.load(causeway_checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        causeway.load_state_dict(ckpt["model_state_dict"])
    else:
        causeway.load_state_dict(ckpt)

    # Load BroadMind
    # The checkpoint contains the full model under 'model_state_dict'
    bm_ckpt = torch.load(broadmind_checkpoint, map_location=device, weights_only=False)

    # BroadMind needs its config — import or reconstruct
    # This assumes BroadMind_v077_elastic is importable
    broadmind_module = None
    for path in [
        os.path.join(os.path.dirname(broadmind_checkpoint), "BroadMind_v077_elastic"),
        "BroadMind_v077_elastic",
    ]:
        try:
            parent = os.path.dirname(broadmind_checkpoint)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            from BroadMind_v077_elastic import BroadMindV077, Config as BMConfig
            bm_config = BMConfig()
            bm_config.device = torch.device(device)
            broadmind_module = BroadMindV077(bm_config).to(device)
            if "model_state_dict" in bm_ckpt:
                broadmind_module.load_state_dict(bm_ckpt["model_state_dict"])
            else:
                broadmind_module.load_state_dict(bm_ckpt)
            break
        except ImportError:
            continue

    if broadmind_module is None:
        raise ImportError(
            "Could not import BroadMind_v077_elastic. "
            "Ensure the BroadMind directory is on sys.path."
        )

    # Build joint executor
    executor = CausalProgramExecutor(
        causeway=causeway.to(device),
        broadmind=broadmind_module,
        d_model=d_model,
        d_causal=d_causal,
        d_wisdom=d_wisdom,
        d_action=d_action,
        fusion_mode=fusion_mode,
    ).to(device)

    return executor
