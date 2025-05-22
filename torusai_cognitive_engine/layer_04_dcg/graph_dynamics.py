from typing import TYPE_CHECKING, Optional # Added Optional
from collections import defaultdict
# import random # Not directly used in the provided HebbianLayer logic that defaults new edges

if TYPE_CHECKING:
    from .concept_graph import EnhancedConceptGraph # For type hinting

# From User Part 2 - GeometryLayer, DynamicsLayer, HebbianLayer
# Maps to README Layer 4: Dynamic Concept Graph (DCG) - as operational dynamics

class GeometryLayer:
    """
    Handles the decay of activation in the concept graph over time.
    This simulates forgetting or the fading of unreinforced concepts.
    Corresponds to L0 Geometry Runtime Substrate.
    """
    def __init__(self, decay_rate: float = 0.02):
        """
        Initializes the GeometryLayer.

        Args:
            decay_rate: The amount by which activation decays each cycle.
        """
        self.decay_rate: float = decay_rate

    def run(self, cg: 'EnhancedConceptGraph') -> None:
        """
        Applies activation decay to all nodes in the concept graph.

        Args:
            cg: The EnhancedConceptGraph instance to operate on.
        """
        for cid in list(cg.nodes.keys()): # Iterate over a copy of keys for safety
            current_activation = cg.getActivation(cid)
            cg.setActivation(cid, current_activation - self.decay_rate)

class DynamicsLayer:
    """
    Propagates activation energy through the concept graph based on node
    activations and weighted edges.
    Corresponds to L2 Activation Dynamics (CA).
    """
    ACTIVATION_PROPAGATION_FACTOR: float = 0.3 # Class constant for propagation factor

    def run(self, cg: 'EnhancedConceptGraph') -> None:
        """
        Calculates and applies changes in activation to nodes based on
        the activation of their connected source nodes and edge weights.

        Args:
            cg: The EnhancedConceptGraph instance to operate on.
        """
        delta: defaultdict[str, float] = defaultdict(float)
        for (s, t), edge_data in cg.edges.items():
            source_activation = cg.getActivation(s)
            # Use .get for safety, default weight to 0 if not present
            edge_weight = edge_data.get("weight", 0.0)
            delta[t] += source_activation * edge_weight * self.ACTIVATION_PROPAGATION_FACTOR
        
        for t_cid, val_delta in delta.items():
            current_activation = cg.getActivation(t_cid)
            cg.setActivation(t_cid, current_activation + val_delta) # Activation is clamped in setActivation

class HebbianLayer:
    """
    Implements Hebbian learning ("neurons that fire together, wire together").
    It strengthens connections between co-active nodes and prunes weak connections.
    Corresponds to L4 Hebbian Edge Memory.
    """
    DEFAULT_NEW_EDGE_WEIGHT: float = 0.5
    DEFAULT_NEW_EDGE_RELATION: str = "assoc_hebbian"
    MAX_EDGE_WEIGHT: float = 5.0
    DEFAULT_CO_ACTIVATION_THRESHOLD: float = 0.6


    def __init__(self, multiplier: float = 1.1, prune_threshold: float = 0.2,
                 co_activation_threshold: Optional[float] = None):
        """
        Initializes the HebbianLayer.

        Args:
            multiplier: Factor by which to multiply weights of co-active nodes.
            prune_threshold: Activation threshold below which edges are removed.
            co_activation_threshold: Minimum activation for a node to be considered 'active'
                                     for Hebbian learning. Defaults to 0.6.
        """
        self.multiplier: float = multiplier
        self.prune_threshold: float = prune_threshold
        self.co_activation_threshold: float = co_activation_threshold if co_activation_threshold is not None else self.DEFAULT_CO_ACTIVATION_THRESHOLD


    def run(self, cg: 'EnhancedConceptGraph') -> None:
        """
        Applies Hebbian learning rules to the concept graph.
        Strengthens edges between highly co-activated nodes and prunes weak edges.

        Args:
            cg: The EnhancedConceptGraph instance to operate on.
        """
        active_nodes = [
            cid for cid, data in cg.nodes.items()
            if data.get("activation", 0.0) > self.co_activation_threshold
        ]
        
        # Strengthen existing edges or add new ones between co-active nodes
        # This is O(N^2) for active_nodes, can be computationally intensive.
        for i in range(len(active_nodes)):
            for j in range(i + 1, len(active_nodes)):
                s_cid, t_cid = active_nodes[i], active_nodes[j]
                
                # Consider both directions for Hebbian association if graph is conceptually undirected
                # For now, creating/strengthening one way based on pair order.
                # If (t_cid, s_cid) should also be strengthened, this loop needs adjustment or
                # the graph needs to handle undirected edges more explicitly for this layer.
                
                edge_key = (s_cid, t_cid)
                current_edge_data = cg.edges.get(edge_key)

                if current_edge_data:
                    new_weight = current_edge_data["weight"] * self.multiplier
                    relation = current_edge_data.get("relation", self.DEFAULT_NEW_EDGE_RELATION)
                else: # New edge
                    new_weight = self.DEFAULT_NEW_EDGE_WEIGHT # Start with a base weight
                    relation = self.DEFAULT_NEW_EDGE_RELATION
                
                cg.addEdge(s_cid, t_cid, relation=relation, weight=min(new_weight, self.MAX_EDGE_WEIGHT))

        # Prune weak edges
        edges_to_prune = [
            edge_key for edge_key, edge_data in cg.edges.items()
            if edge_data.get("weight", 0.0) < self.prune_threshold
        ]
        for edge_key in edges_to_prune:
            if edge_key in cg.edges: # Check if still exists (though unlikely to change in this loop)
                 del cg.edges[edge_key]
