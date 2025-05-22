from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph # For type hinting

# From User Part 3 - Labelled "Layer 6: Symbolic Reflection"
# Corresponds to L6 Symbolic Reflection in the 15-layer specification.
# Identifies and records the most salient (highly active) concepts.

class SymbolicReflectionLayer:
    """
    Identifies the most highly activated concepts in the graph, forming a "reflection"
    of the current cognitive focus. This reflection is stored in the shared engine state.
    It uses the `getActiveConceptPath` method of the concept graph.
    """
    DEFAULT_REFLECTION_THRESHOLD: float = 0.7

    def __init__(self, reflection_threshold: Optional[float] = None):
        """
        Initializes the SymbolicReflectionLayer.

        Args:
            reflection_threshold: The minimum activation for a concept to be included
                                  in the reflection. Defaults to DEFAULT_REFLECTION_THRESHOLD.
        """
        self.reflection_threshold: float = reflection_threshold if reflection_threshold is not None else self.DEFAULT_REFLECTION_THRESHOLD

    def run(self, state: Dict[str, Any]) -> None:
        """
        Executes the symbolic reflection process. It calls `getActiveConceptPath` on the
        concept graph using the configured threshold and updates `state['reflection']`.

        Args:
            state: The shared engine state, expected to contain 'cg' (EnhancedConceptGraph)
                   and will be updated with 'reflection'.
        """
        cg: Optional['EnhancedConceptGraph'] = state.get('cg')
        if not cg:
            # print("Warning: Concept graph 'cg' not found in state for SymbolicReflectionLayer.")
            return
        
        # getActiveConceptPath is a method of EnhancedConceptGraph
        reflection = cg.getActiveConceptPath(threshold=self.reflection_threshold)
        state['reflection'] = reflection
