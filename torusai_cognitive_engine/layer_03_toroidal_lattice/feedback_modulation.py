from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph # For type hinting

# From User Part 3 - Labelled "Layer 3: Feedback Modulation"
# Corresponds to L3 Feedback Modulation in the 15-layer specification.
# Adjusts node activations globally based on a sentiment signal.

class FeedbackModulationLayer:
    """
    Modulates the activation of all concept nodes based on a global sentiment value.
    Positive sentiment slightly increases activation, negative sentiment decreases it.
    This layer simulates how overall mood or feedback can influence cognitive processing.
    """
    DEFAULT_SENTIMENT_DELTA: float = 0.03

    def __init__(self, sentiment_delta: Optional[float] = None):
        """
        Initializes the FeedbackModulationLayer.

        Args:
            sentiment_delta: The amount by which to change activation based on sentiment.
                             Defaults to DEFAULT_SENTIMENT_DELTA.
        """
        self.last_sentiment: float = 0.1  # Default initial sentiment, can be updated by TorusEngine
        self.sentiment_modulation_delta: float = sentiment_delta if sentiment_delta is not None else self.DEFAULT_SENTIMENT_DELTA

    def run(self, state: Dict[str, Any]) -> None:
        """
        Applies feedback modulation to node activations in the concept graph.
        The `self.last_sentiment` attribute of this layer instance should be updated
        by the TorusEngine with the current global valence before this method is called.

        Args:
            state: The shared engine state, expected to contain 'cg' (EnhancedConceptGraph).
        """
        cg: Optional['EnhancedConceptGraph'] = state.get('cg')
        if not cg:
            # print("Warning: Concept graph 'cg' not found in state for FeedbackModulationLayer.")
            return

        # Determine the change in activation based on the layer's current sentiment
        # This sentiment is typically set by the TorusEngine based on state['valence']
        delta = self.sentiment_modulation_delta if self.last_sentiment > 0 else -self.sentiment_modulation_delta
        
        for cid in cg.nodes: # cg.nodes is a dictionary, iterating over its keys
            current_activation = cg.getActivation(cid)
            # setActivation in EnhancedConceptGraph already handles clamping between 0.0 and 1.0
            cg.setActivation(cid, current_activation + delta)
