from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph # For type hinting

# From User Part 3 - Labelled "Layer 10: Social Mind Layer"
# Corresponds to L10 Social Mind in the 15-layer specification.
# Modulates cognitive processing based on the overall affective state (valence).

class SocialMindLayer:
    """
    Simulates a "social mind" influence by damping overall cognitive activity
    (node activations) if the engine's current affective valence is negative.
    This represents how a negative mood might suppress or alter thinking.
    """
    DEFAULT_NEGATIVE_VALENCE_DAMPING_FACTOR: float = 0.95

    def __init__(self, damping_factor: Optional[float] = None):
        """
        Initializes the SocialMindLayer.

        Args:
            damping_factor: The factor by which to multiply activations if valence is negative.
                            Defaults to DEFAULT_NEGATIVE_VALENCE_DAMPING_FACTOR.
        """
        self.damping_factor: float = damping_factor if damping_factor is not None else self.DEFAULT_NEGATIVE_VALENCE_DAMPING_FACTOR

    def run(self, state_dict: Dict[str, Any]) -> None:
        """
        Applies activation damping if the current valence in the shared state is negative.

        Args:
            state_dict: The shared engine state, expected to contain 'cg' (EnhancedConceptGraph)
                        and 'valence'.
        """
        cg_instance: Optional['EnhancedConceptGraph'] = state_dict.get('cg')
        current_valence: float = state_dict.get("valence", 0.0)

        if not cg_instance:
            # print("Warning: Concept graph 'cg' not found in state for SocialMindLayer.")
            return

        if current_valence < 0:
            for cid in list(cg_instance.nodes.keys()): # Iterate over a copy of keys
                current_activation = cg_instance.getActivation(cid)
                # setActivation in EnhancedConceptGraph handles clamping
                cg_instance.setActivation(cid, current_activation * self.damping_factor)
