import random
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph
    from ..layer_13_imr.language_generator import EnhancedLanguageGenerator

# From User Part 2 - DreamLayer
# Corresponds to L11 Offline Dreaming in the 15-layer specification.
# Simulates offline cognitive processing, potentially for consolidation or exploration.

class DreamLayer:
    """
    Simulates an "offline dreaming" process. During a dream cycle, this layer
    randomly activates a subset of concept nodes to a high level. It then uses
    the language generator to create a linguistic narrative of this activated state,
    which is logged as a "dream."
    """
    DEFAULT_DREAM_QUERY_STRING: str = "dream?" # Placeholder query for LG during dreaming

    def __init__(self,
                 lg_instance: Optional['EnhancedLanguageGenerator'],
                 n_candidates: int = 5,
                 keep_top_k: int = 2, # Currently unused by run logic, but kept for spec alignment
                 dream_query_string: Optional[str] = None):
        """
        Initializes the DreamLayer.

        Args:
            lg_instance: An instance of EnhancedLanguageGenerator.
            n_candidates: The number of nodes to attempt to randomly activate.
            keep_top_k: (Currently unused) Intended for selecting a subset of dream candidates.
            dream_query_string: The string passed to the language generator during dreaming.
                                Defaults to DEFAULT_DREAM_QUERY_STRING.
        """
        self.lg: Optional['EnhancedLanguageGenerator'] = lg_instance
        self.n_candidates: int = n_candidates
        self.keep_top_k: int = keep_top_k
        self.dream_query_string: str = dream_query_string if dream_query_string is not None else self.DEFAULT_DREAM_QUERY_STRING

    def run(self, cg_instance: Optional['EnhancedConceptGraph'], state_dict: Dict[str, Any]) -> None:
        """
        Executes a dream cycle. Randomly activates nodes and generates a dream narrative.

        Args:
            cg_instance: The EnhancedConceptGraph instance.
            state_dict: The shared engine state, will be updated with 'dream_log'.
        """
        if not self.lg:
            # print("Warning: Language generator not available for DreamLayer.")
            return
        if not cg_instance:
            # print("Warning: Concept graph not available for DreamLayer.")
            return

        node_ids: List[str] = list(cg_instance.nodes.keys())
        if not node_ids:
            # print("Warning: No nodes in concept graph for DreamLayer.")
            return

        sample_size: int = min(self.n_candidates, len(node_ids))
        
        if sample_size > 0:
            picked_node_ids: List[str] = random.sample(node_ids, sample_size)
            for cid in picked_node_ids:
                # Set activation to maximum for dream simulation
                cg_instance.setActivation(cid, 1.0)
        
        # Generate dream line using the language generator with a specific dream query
        dream_line: str = self.lg.generate(self.dream_query_string)
        
        # Append the generated dream narrative to the dream_log in the shared state
        state_dict.setdefault("dream_log", []).append(dream_line)
