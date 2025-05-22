from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .concept_graph import EnhancedConceptGraph # For type hinting

# From User Part 3 - Labelled "Layer 1: Symbolic Tagging"
# Corresponds to L1 Symbolic Tagging & Typing in the 15-layer specification.
# Ensures basic structural integrity of nodes regarding their type.

class SymbolicTaggingLayer:
    """
    Ensures that every node in the concept graph has a 'type' property.
    If a node lacks a 'type', it defaults to "concept".
    This layer helps maintain data consistency within the concept graph.
    """
    DEFAULT_NODE_TYPE: str = "concept"

    def run(self, state: Dict[str, Any]) -> None:
        """
        Executes the symbolic tagging process on the concept graph in the given state.

        Args:
            state: The shared engine state dictionary, expected to contain a 'cg'
                   key referencing an EnhancedConceptGraph instance.
        """
        cg: Optional['EnhancedConceptGraph'] = state.get('cg')
        
        if not cg:
            # print("Warning: Concept graph 'cg' not found in state for SymbolicTaggingLayer.")
            # Consider formal logging if this becomes a common or critical issue.
            return
            
        for cid in cg.nodes: # Iterates over keys of the nodes dictionary
            # setdefault on the node's dictionary itself to add 'type' if not present.
            cg.nodes[cid].setdefault("type", self.DEFAULT_NODE_TYPE)
