from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph # For type hinting

# From User Part 3 - Labelled "Layer 7: Metrics Observation"
# Corresponds to L7 Metrics Observation in the 15-layer specification.
# Calculates and records key metrics about the current state of the concept graph.

class MetricsObservationLayer:
    """
    Observes the concept graph and computes various metrics such as average node
    activation, total number of nodes, and total number of edges. These metrics
    are stored in the shared engine state and can be used for monitoring,
    telemetry, or influencing other layers.
    """
    def run(self, state: Dict[str, Any]) -> None:
        """
        Calculates graph metrics and updates `state['metrics']`.

        Args:
            state: The shared engine state, expected to contain 'cg' (EnhancedConceptGraph)
                   and will be updated with 'metrics'.
        """
        cg: Optional['EnhancedConceptGraph'] = state.get('cg')
        if not cg:
            # print("Warning: Concept graph 'cg' not found in state for MetricsObservationLayer.")
            state['metrics'] = { # Ensure metrics key exists with default values
                "avg_activation": 0.0,
                "node_count": 0,
                "edge_count": 0
            }
            return

        activations: List[float] = [node_data.get("activation", 0.0) for node_data in cg.nodes.values()]
        avg_activation: float = sum(activations) / len(activations) if activations else 0.0
        
        state['metrics'] = {
            "avg_activation": round(avg_activation, 4),
            "node_count": len(cg.nodes),
            "edge_count": len(cg.edges)
        }
