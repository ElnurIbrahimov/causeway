"""
Causeway: Lightweight Counterfactual World Interface

A parameter-efficient causal adapter that bolts onto frozen Transformers
to provide structured counterfactual reasoning. Estimates causal displacement
under intervention, not correlation.

Architecture:
    StateEncoder -> CausalGraphLayer -> InterventionEngine -> DeltaPredictor
"""

from causeway.state_encoder import StateEncoder
from causeway.causal_graph import CausalGraphLayer
from causeway.intervention_engine import InterventionEngine
from causeway.delta_predictor import DeltaPredictor
from causeway.causeway_module import Causeway
from causeway.losses import CausewayLoss

__version__ = "0.1.0"
__all__ = [
    "Causeway",
    "StateEncoder",
    "CausalGraphLayer",
    "InterventionEngine",
    "DeltaPredictor",
    "CausewayLoss",
]
