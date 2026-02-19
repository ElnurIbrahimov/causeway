"""
CausewayLayer: Surgically inserted into a Transformer's layer stack.

Instead of generating prefix tokens that the model can ignore, this module
is spliced between two decoder layers so every forward pass flows through
Causeway's causal pipeline. The model MUST compute through it.

Architecture:
    hidden_states (batch, seq_len, d_model)
        -> [Attention Pooling] -> h_state, h_action
        -> [StateEncoder -> CausalGraphLayer -> InterventionEngine] (warm-started)
        -> z_counterfactual
        -> [WriteBack projection]
        -> delta_h (batch, d_model)
        -> [Scalar gate, broadcast to all positions]
        -> hidden_states + gated_delta_h (residual addition)

The gate is initialized at sigmoid(-5.0) ~ 0.007 so the layer starts as
near-identity, preventing catastrophic disruption on insertion.

Usage:
    from causeway.causeway_module import Causeway
    layer = CausewayLayerGPT2(causeway, d_model=768, d_causal=48)
    # Insert into model.transformer.h at index 6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from causeway.causeway_module import Causeway


class AttentionPooler(nn.Module):
    """Cross-attention pooler: learned queries attend over sequence positions.

    Uses bottleneck projections (d_model -> d_pool) for K and V to keep
    parameter count low. Two separate queries extract state and action
    summaries from the residual stream.
    """

    def __init__(self, d_model: int, d_pool: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_pool = d_pool
        self.n_heads = n_heads
        assert d_pool % n_heads == 0, f"d_pool={d_pool} must be divisible by n_heads={n_heads}"
        self.head_dim = d_pool // n_heads

        # Bottleneck projections for K and V
        self.k_proj = nn.Linear(d_model, d_pool, bias=False)
        self.v_proj = nn.Linear(d_model, d_pool, bias=False)

        # Learned query vectors
        self.state_query = nn.Parameter(torch.randn(1, 1, d_pool) * 0.02)
        self.action_query = nn.Parameter(torch.randn(1, 1, d_pool) * 0.02)

        # Project pooled output back to d_model
        self.state_out = nn.Linear(d_pool, d_model)
        self.action_out = nn.Linear(d_pool, d_model)

        self.scale = self.head_dim ** -0.5

    def _attend(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multi-head cross-attention: (batch, 1, d_pool) query over (batch, seq, d_pool) KV."""
        batch, seq_len, _ = key.shape

        # Reshape to (batch, n_heads, seq/1, head_dim)
        q = query.view(batch, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = key.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = value.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, 1, seq)

        if attention_mask is not None:
            # attention_mask: (batch, seq_len) with 1=attend, 0=mask
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)  # (batch, heads, 1, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, 1, self.n_heads * self.head_dim)
        return out.squeeze(1)  # (batch, d_pool)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract state and action summaries from the residual stream.

        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len), 1=attend, 0=mask

        Returns:
            h_state: (batch, d_model) state summary
            h_action: (batch, d_model) action summary
        """
        hidden_states = hidden_states.float()
        batch = hidden_states.shape[0]

        k = self.k_proj(hidden_states)  # (batch, seq, d_pool)
        v = self.v_proj(hidden_states)  # (batch, seq, d_pool)

        # Expand queries to batch
        sq = self.state_query.expand(batch, -1, -1).squeeze(1).unsqueeze(1)
        aq = self.action_query.expand(batch, -1, -1).squeeze(1).unsqueeze(1)

        h_state_pool = self._attend(sq, k, v, attention_mask)   # (batch, d_pool)
        h_action_pool = self._attend(aq, k, v, attention_mask)  # (batch, d_pool)

        h_state = self.state_out(h_state_pool)    # (batch, d_model)
        h_action = self.action_out(h_action_pool)  # (batch, d_model)

        return h_state, h_action


class WriteBack(nn.Module):
    """Projects causal delta back to residual stream dimension.

    Uses a bottleneck: 2*d_causal -> bottleneck_dim -> d_model
    """

    def __init__(self, d_causal: int, d_model: int, bottleneck_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_causal, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, d_model),
        )

    def forward(self, z_refined: torch.Tensor, z_counterfactual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_refined: (batch, d_causal)
            z_counterfactual: (batch, d_causal)

        Returns:
            delta_h: (batch, d_model)
        """
        x = torch.cat([z_refined, z_counterfactual], dim=-1)
        return self.net(x)


class CausewayLayer(nn.Module):
    """Base CausewayLayer inserted between Transformer decoder layers.

    Reads the residual stream, runs the causal pipeline, and writes a
    gated modification back. Subclasses match specific model signatures.

    Args:
        causeway: Pre-trained Causeway module (warm-started from oracle).
        d_model: Transformer hidden dimension.
        d_causal: Causeway's causal variable dimension.
        d_pool: Bottleneck dimension for attention pooler K/V projections.
        bottleneck_dim: WriteBack projection bottleneck dimension.
        gate_init: Initial gate logit. sigmoid(-5.0) ~ 0.007.
        n_heads: Number of attention heads in the pooler.
    """

    def __init__(
        self,
        causeway: Causeway,
        d_model: int,
        d_causal: int,
        d_pool: Optional[int] = None,
        bottleneck_dim: int = 256,
        gate_init: float = -5.0,
        n_heads: int = 4,
    ):
        super().__init__()
        d_pool = d_pool or d_causal * 4
        # Round d_pool to be divisible by n_heads
        d_pool = (d_pool // n_heads) * n_heads

        self.causeway = causeway
        self.d_model = d_model
        self.d_causal = d_causal

        # Attention-pooled extraction
        self.pooler = AttentionPooler(d_model, d_pool, n_heads=n_heads)

        # Write-back projection
        self.write_back = WriteBack(d_causal, d_model, bottleneck_dim)

        # Scalar gate initialized near zero for safe insertion
        self.gate_logit = nn.Parameter(torch.tensor(gate_init))

    @property
    def gate(self) -> torch.Tensor:
        """Current gate value (scalar in [0, 1])."""
        return torch.sigmoid(self.gate_logit)

    def _apply_causeway(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Core pipeline: pool -> causeway -> write-back -> gated residual.

        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len), 1=attend, 0=mask

        Returns:
            modified hidden_states: (batch, seq_len, d_model)
        """
        # 1. Attention-pooled extraction
        h_state, h_action = self.pooler(hidden_states, attention_mask)

        # 2. Run Causeway's causal pipeline
        z = self.causeway.state_encoder(h_state)
        z_refined = self.causeway.causal_graph(z)
        adjacency = self.causeway.causal_graph.adjacency
        z_counterfactual, mask, epsilon = self.causeway.intervention_engine(
            z_refined, h_action, adjacency,
            num_propagation_steps=self.causeway.propagation_steps,
        )

        # 3. Write-back to residual stream dimension
        delta_h = self.write_back(z_refined, z_counterfactual)  # (batch, d_model)

        # 4. Gated residual addition, broadcast to all positions
        gate = self.gate
        delta_h = gate * delta_h  # (batch, d_model)
        delta_h = delta_h.unsqueeze(1)  # (batch, 1, d_model)

        # Match dtype (fp16/bf16 models)
        delta_h = delta_h.to(hidden_states.dtype)

        return hidden_states + delta_h

    def _apply_causeway_with_internals(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Like _apply_causeway but also returns internal representations for distillation."""
        h_state, h_action = self.pooler(hidden_states, attention_mask)

        z = self.causeway.state_encoder(h_state)
        z_refined = self.causeway.causal_graph(z)
        adjacency = self.causeway.causal_graph.adjacency
        z_counterfactual, mask, epsilon = self.causeway.intervention_engine(
            z_refined, h_action, adjacency,
            num_propagation_steps=self.causeway.propagation_steps,
        )

        delta_h = self.write_back(z_refined, z_counterfactual)
        gate = self.gate
        gated_delta_h = gate * delta_h

        delta_h_broadcast = gated_delta_h.unsqueeze(1).to(hidden_states.dtype)
        modified = hidden_states + delta_h_broadcast

        internals = {
            'h_state': h_state,
            'h_action': h_action,
            'z_refined': z_refined,
            'z_counterfactual': z_counterfactual,
            'delta_h': delta_h,
            'gate': gate,
            'gated_delta_h': gated_delta_h,
        }
        return modified, internals

    def get_regularization_losses(self) -> Dict[str, torch.Tensor]:
        """Return regularization losses from the underlying Causeway module."""
        return self.causeway.get_regularization_losses()

    def get_diagnostics(self) -> Dict:
        """Return diagnostic info including gate value."""
        diag = self.causeway.get_diagnostics()
        diag['gate_value'] = self.gate.item()
        diag['gate_logit'] = self.gate_logit.item()
        n_new_params = (
            sum(p.numel() for p in self.pooler.parameters())
            + sum(p.numel() for p in self.write_back.parameters())
            + 1  # gate_logit
        )
        diag['new_params'] = n_new_params
        diag['new_params_human'] = f"{n_new_params / 1e6:.3f}M"
        return diag


class CausewayLayerGPT2(CausewayLayer):
    """CausewayLayer matching GPT-2 decoder layer forward signature.

    GPT-2 calls blocks with positional args:
        block(hidden_states, past_key_values, cache_position, attention_mask,
              head_mask, encoder_hidden_states, encoder_attention_mask=..., ...)

    Returns (hidden_states, present_key_value) or
    (hidden_states, present_key_value, attentions).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        # attention_mask from GPT-2 is a 4D causal mask, not a simple 2D mask.
        # We pass None to our pooler (it works fine without a mask for full sequences).
        modified = self._apply_causeway(hidden_states, attention_mask=None)
        # GPT-2 returns (hidden_states, present_key_value)
        # present is None since CausewayLayer has no KV cache
        outputs = (modified, None)
        if output_attentions:
            outputs = outputs + (None,)
        return outputs


class CausewayLayerLlama(CausewayLayer):
    """CausewayLayer matching Llama/Mistral decoder layer forward signature.

    Modern transformers: LlamaDecoderLayer.forward returns hidden_states directly.
    Called as: hidden_states = decoder_layer(hidden_states, attention_mask=..., ...)
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple] = None,
        **kwargs,
    ):
        # attention_mask from Llama is a 4D causal mask; pass None to our pooler.
        modified = self._apply_causeway(hidden_states, attention_mask=None)
        return modified
