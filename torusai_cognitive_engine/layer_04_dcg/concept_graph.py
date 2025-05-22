from typing import Dict, Any, Tuple, List, Optional

# From User Part 1 - EnhancedConceptGraph
# Maps to README Layer 4: Dynamic Concept Graph (DCG)
# import numpy as np # Not strictly needed by this class as provided

class EnhancedConceptGraph:
    """
    Represents a dynamic concept graph where nodes are concepts and edges
    are relationships between them. Nodes have properties including activation
    levels and linguistic information. Edges have types (relations) and weights.
    """
    def __init__(self):
        """
        Initializes an empty concept graph.
        self.nodes stores node data, keyed by concept ID (cid).
        self.edges stores edge data, keyed by a tuple of (source_cid, target_cid).
        """
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def addEnhancedNode(self, cid: str, properties: Optional[Dict[str, Any]] = None, linguistics: Optional[Dict[str, Any]] = None) -> None:
        """
        Adds a new node or updates an existing node in the graph.

        Args:
            cid: The unique concept ID for the node.
            properties: A dictionary of properties for the node (e.g., type).
                        Activation is initialized to 0.0 if not provided.
            linguistics: A dictionary of linguistic information, e.g.,
                         {"wordForms": {"base": "concept"}, "syntacticRoles": ["noun"]}.
        """
        base_properties: Dict[str, Any] = {"activation": 0.0}
        if properties:
            base_properties.update(properties)
        
        self.nodes[cid] = base_properties
        
        # Ensure linguistic properties are set, with defaults
        linguistics_data = linguistics if linguistics is not None else {}
        self.nodes[cid]["wordForms"] = linguistics_data.get("wordForms", {"base": str(cid)})
        self.nodes[cid]["syntacticRoles"] = linguistics_data.get("syntacticRoles", [])

    def setActivation(self, cid: str, value: float) -> None:
        """
        Sets the activation level for a given concept node.
        Activation is clamped between 0.0 and 1.0.

        Args:
            cid: The concept ID of the node.
            value: The new activation value (float).
        """
        if cid in self.nodes:
            self.nodes[cid]["activation"] = max(0.0, min(1.0, float(value)))
        # else:
            # Optionally handle case where node doesn't exist, e.g., print warning or auto-add
            # print(f"Warning: Node {cid} not found for setActivation.")

    def getActivation(self, cid: str) -> float:
        """
        Retrieves the activation level of a concept node.

        Args:
            cid: The concept ID of the node.

        Returns:
            The activation level (float), or 0.0 if the node or activation is not found.
        """
        return self.nodes.get(cid, {}).get("activation", 0.0)

    def addEdge(self, s_cid: str, t_cid: str, relation: str = "assoc", weight: float = 1.0) -> None:
        """
        Adds a directed edge between two existing nodes in the graph.

        Args:
            s_cid: The concept ID of the source node.
            t_cid: The concept ID of the target node.
            relation: The type of relationship (e.g., "is_a", "causes"). Defaults to "assoc".
            weight: The strength of the relationship. Defaults to 1.0.
        """
        if s_cid in self.nodes and t_cid in self.nodes:
            self.edges[(s_cid, t_cid)] = {"relation": relation, "weight": float(weight)}
        # else:
            # Optionally handle case where one or both nodes don't exist
            # print(f"Warning: Node(s) {s_cid} or {t_cid} not found for addEdge.")

    def getActiveConceptPath(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieves a list of concepts whose activation meets or exceeds a given threshold,
        sorted by activation in descending order.

        Args:
            threshold: The minimum activation level for a concept to be included.

        Returns:
            A list of dictionaries, where each dictionary represents an active concept
            and includes its 'id', 'activation', and 'type'.
        """
        # Ensure 'type' exists or provide a default
        active_nodes = [
            {
                "id": cid,
                "activation": data.get("activation", 0.0), # Use .get for safety
                "type": data.get("type", "concept") # Default type if not present
            }
            for cid, data in self.nodes.items() if data.get("activation", 0.0) >= threshold
        ]
        return sorted(active_nodes, key=lambda x: -x["activation"])
