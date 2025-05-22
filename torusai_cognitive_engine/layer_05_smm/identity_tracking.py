from typing import Dict, Any, List, Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph # For type hinting

# From User Part 3 - Labelled "Layer 5: Identity Tracking"
# Corresponds to L5 Identity Tracking in the 15-layer specification.
# Groups nodes by their 'type' property.

class IdentityTrackingLayer:
    """
    Tracks identities of concepts by grouping them into clusters based on their 'type' property.
    The resulting clusters are stored in the shared engine state. This can be used for
    type-based reasoning or selective processing in other layers.
    """
    DEFAULT_UNKNOWN_TYPE: str = "unknown"

    def run(self, state: Dict[str, Any]) -> None:
        """
        Executes the identity tracking process. It iterates through all nodes in the
        concept graph, groups them by their 'type' property, and updates the
        'clusters' key in the shared engine state.

        Args:
            state: The shared engine state dictionary, expected to contain 'cg'
                   (EnhancedConceptGraph) and will be updated with 'clusters'.
        """
        cg: Optional['EnhancedConceptGraph'] = state.get('cg')
        if not cg:
            # print("Warning: Concept graph 'cg' not found in state for IdentityTrackingLayer.")
            return

        # Use defaultdict for cleaner cluster building
        clusters: Dict[str, List[str]] = defaultdict(list)
        for cid, node_data in cg.nodes.items():
            node_type = node_data.get("type", self.DEFAULT_UNKNOWN_TYPE)
            clusters[node_type].append(cid)
        
        state['clusters'] = dict(clusters) # Convert back to dict if defaultdict is not desired in state
