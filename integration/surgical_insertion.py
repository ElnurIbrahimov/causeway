"""
Surgical Insertion: Insert/remove CausewayLayer into Transformer layer stacks.

Supports GPT-2, Llama, and Mistral architectures. The CausewayLayer is
inserted between two existing decoder layers so that the residual stream
flows through Causeway's causal pipeline on every forward pass.

Usage:
    from integration.surgical_insertion import insert_causeway_layer, remove_causeway_layer

    # Insert at layer 6 of GPT-2 (12 layers total)
    causeway_layer = insert_causeway_layer(
        lm_model, causeway, layer_idx=6, architecture="gpt2"
    )

    # Remove to restore original model
    remove_causeway_layer(lm_model, layer_idx=6, architecture="gpt2")
"""

import torch
import torch.nn as nn
from typing import Optional

from causeway.causeway_module import Causeway
from integration.causeway_layer import (
    CausewayLayer,
    CausewayLayerGPT2,
    CausewayLayerLlama,
)


def detect_architecture(model: nn.Module) -> str:
    """Detect Transformer architecture from model structure."""
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return "gpt2"
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return "llama"
    else:
        raise ValueError(
            f"Cannot detect architecture from {type(model).__name__}. "
            "Supported: GPT-2 (transformer.h), Llama/Mistral (model.layers)."
        )


def get_layer_list(model: nn.Module, architecture: str) -> nn.ModuleList:
    """Get the ModuleList containing decoder layers."""
    if architecture == "gpt2":
        return model.transformer.h
    elif architecture in ("llama", "mistral"):
        return model.model.layers
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def get_default_insertion_index(model: nn.Module, architecture: str) -> int:
    """Return the midpoint layer index (default insertion point)."""
    layers = get_layer_list(model, architecture)
    return len(layers) // 2


def insert_causeway_layer(
    model: nn.Module,
    causeway: Causeway,
    layer_idx: Optional[int] = None,
    architecture: Optional[str] = None,
    d_pool: Optional[int] = None,
    bottleneck_dim: int = 256,
    gate_init: float = -5.0,
    n_heads: int = 4,
    device: Optional[torch.device] = None,
) -> CausewayLayer:
    """Insert a CausewayLayer into a Transformer's layer stack.

    The CausewayLayer is inserted AFTER the specified layer index,
    so data flows: layer[idx] -> CausewayLayer -> layer[idx+1].

    Args:
        model: HuggingFace causal LM (GPT2LMHeadModel, LlamaForCausalLM, etc.)
        causeway: Pre-trained Causeway module.
        layer_idx: Insert after this layer. Defaults to midpoint.
        architecture: "gpt2", "llama", or "mistral". Auto-detected if None.
        d_pool: Attention pooler bottleneck dim. Defaults to d_causal * 4.
        bottleneck_dim: WriteBack bottleneck dim.
        gate_init: Initial gate logit (sigmoid(-5) ~ 0.007).
        n_heads: Number of attention heads in the pooler.
        device: Device for the new layer. Inferred from model if None.

    Returns:
        The inserted CausewayLayer instance.
    """
    if architecture is None:
        architecture = detect_architecture(model)

    layers = get_layer_list(model, architecture)
    n_layers = len(layers)

    if layer_idx is None:
        layer_idx = n_layers // 2

    if layer_idx < 0 or layer_idx >= n_layers:
        raise ValueError(f"layer_idx={layer_idx} out of range [0, {n_layers})")

    d_model = causeway.d_model
    d_causal = causeway.d_causal

    # Create appropriate subclass
    if architecture == "gpt2":
        causeway_layer = CausewayLayerGPT2(
            causeway=causeway,
            d_model=d_model,
            d_causal=d_causal,
            d_pool=d_pool,
            bottleneck_dim=bottleneck_dim,
            gate_init=gate_init,
            n_heads=n_heads,
        )
    elif architecture in ("llama", "mistral"):
        causeway_layer = CausewayLayerLlama(
            causeway=causeway,
            d_model=d_model,
            d_causal=d_causal,
            d_pool=d_pool,
            bottleneck_dim=bottleneck_dim,
            gate_init=gate_init,
            n_heads=n_heads,
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Device handling
    if device is None:
        device = next(model.parameters()).device
    causeway_layer = causeway_layer.to(device)

    # Insert into layer list: layers becomes [..., layer[idx], CausewayLayer, layer[idx+1], ...]
    new_layers = list(layers)
    new_layers.insert(layer_idx + 1, causeway_layer)

    # Replace the ModuleList and update config.n_layer / num_hidden_layers
    new_module_list = nn.ModuleList(new_layers)
    if architecture == "gpt2":
        model.transformer.h = new_module_list
        model.config.n_layer = len(new_layers)
    elif architecture in ("llama", "mistral"):
        model.model.layers = new_module_list
        model.config.num_hidden_layers = len(new_layers)

    print(f"Inserted CausewayLayer after layer {layer_idx} "
          f"({architecture}, {n_layers} -> {n_layers + 1} layers)")
    diag = causeway_layer.get_diagnostics()
    print(f"  Gate: {diag['gate_value']:.6f}, New params: {diag['new_params_human']}")

    return causeway_layer


def remove_causeway_layer(
    model: nn.Module,
    layer_idx: Optional[int] = None,
    architecture: Optional[str] = None,
) -> None:
    """Remove a CausewayLayer from a Transformer's layer stack.

    Args:
        model: The modified model.
        layer_idx: Index of the CausewayLayer in the current layer list.
            If None, finds the first CausewayLayer automatically.
        architecture: "gpt2", "llama", or "mistral". Auto-detected if None.
    """
    if architecture is None:
        architecture = detect_architecture(model)

    layers = get_layer_list(model, architecture)

    if layer_idx is None:
        # Find the first CausewayLayer
        for i, layer in enumerate(layers):
            if isinstance(layer, CausewayLayer):
                layer_idx = i
                break
        if layer_idx is None:
            raise ValueError("No CausewayLayer found in the model.")

    if not isinstance(layers[layer_idx], CausewayLayer):
        raise ValueError(f"Layer at index {layer_idx} is not a CausewayLayer "
                         f"(got {type(layers[layer_idx]).__name__})")

    new_layers = list(layers)
    removed = new_layers.pop(layer_idx)
    n_original = len(new_layers)

    new_module_list = nn.ModuleList(new_layers)
    if architecture == "gpt2":
        model.transformer.h = new_module_list
        model.config.n_layer = n_original
    elif architecture in ("llama", "mistral"):
        model.model.layers = new_module_list
        model.config.num_hidden_layers = n_original

    print(f"Removed CausewayLayer from index {layer_idx} "
          f"({n_original + 1} -> {n_original} layers)")


def freeze_transformer_weights(model: nn.Module, architecture: Optional[str] = None) -> None:
    """Freeze all Transformer parameters, leaving CausewayLayer trainable.

    Args:
        model: HuggingFace causal LM with an inserted CausewayLayer.
        architecture: Auto-detected if None.
    """
    if architecture is None:
        architecture = detect_architecture(model)

    layers = get_layer_list(model, architecture)

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze CausewayLayer(s)
    for layer in layers:
        if isinstance(layer, CausewayLayer):
            for p in layer.parameters():
                p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Frozen Transformer: {n_trainable:,} trainable / {n_total:,} total "
          f"({100 * n_trainable / n_total:.3f}%)")


def get_causeway_layer(model: nn.Module, architecture: Optional[str] = None) -> Optional[CausewayLayer]:
    """Find and return the CausewayLayer in a model, or None."""
    if architecture is None:
        architecture = detect_architecture(model)

    layers = get_layer_list(model, architecture)
    for layer in layers:
        if isinstance(layer, CausewayLayer):
            return layer
    return None


def verify_identity(
    model: nn.Module,
    tokenizer,
    test_text: str = "The quick brown fox jumps over the lazy dog.",
    device: Optional[torch.device] = None,
    architecture: Optional[str] = None,
) -> bool:
    """Verify CausewayLayer acts as near-identity when gate is small.

    Compares model output with the current (small) gate value against
    a baseline with gate forced to exactly 0.

    Returns:
        True if the gate~0 output is close to the current output.
    """
    if architecture is None:
        architecture = detect_architecture(model)
    if device is None:
        device = next(model.parameters()).device

    cl = get_causeway_layer(model, architecture)
    if cl is None:
        print("No CausewayLayer found.")
        return False

    inputs = tokenizer(test_text, return_tensors="pt").to(device)

    # Run with current gate
    with torch.no_grad():
        logits_current = model(**inputs).logits

    # Temporarily set gate to effectively 0
    original_logit = cl.gate_logit.data.clone()
    cl.gate_logit.data.fill_(-100.0)
    with torch.no_grad():
        logits_zero_gate = model(**inputs).logits
    cl.gate_logit.data.copy_(original_logit)

    # Compare
    max_diff = (logits_current - logits_zero_gate).abs().max().item()
    passed = max_diff < 0.1  # small gate should produce small diff
    print(f"Identity test: max_diff={max_diff:.6f}, gate={cl.gate.item():.6f} "
          f"{'PASS' if passed else 'FAIL'}")
    return passed
