from typing import Dict, Any, Optional, List # Added List and Optional

# Assuming these are the necessary imports for the classes used by TorusEngine
from ..layer_04_dcg.concept_graph import EnhancedConceptGraph
from ..layer_13_imr.language_generator import EnhancedLanguageGenerator
from ..layer_04_dcg.graph_dynamics import GeometryLayer, DynamicsLayer, HebbianLayer
from ..layer_04_dcg.symbolic_tagging import SymbolicTaggingLayer
from ..layer_03_toroidal_lattice.feedback_modulation import FeedbackModulationLayer
from ..layer_05_smm.identity_tracking import IdentityTrackingLayer
from ..layer_06_errb.symbolic_reflection import SymbolicReflectionLayer
from ..layer_07_mrmce.metrics_observation import MetricsObservationLayer
from ..layer_09_abm.affective_body import AffectiveBodyLayer
from ..layer_10_ire.social_mind_influence import SocialMindLayer
from ..layer_08_egdpe.dream_logic import DreamLayer # This is L11 (Offline Dreaming) in the new spec
from ..layer_12_learning_orchestrator.orchestrator import LearningOrchestrator # L12
from ..layer_13_imr.telemetry import MonitoringLayer # L13
from ..layer_14_api_embedders.output_embedding import APIEmbedderLayer # L14

class TorusEngine:
    """
    Orchestrates the execution of various cognitive layers to simulate a cognitive cycle.
    It manages the concept graph, language generator, engine parameters, and shared state.
    The engine processes queries, can run dream cycles, and updates its internal state
    based on the operations of its constituent layers.
    """
    def __init__(self, cg_instance: EnhancedConceptGraph, lg_instance: EnhancedLanguageGenerator, params: Optional[Dict[str, Any]] = None):
        """
        Initializes the TorusEngine with a concept graph, language generator, and parameters.

        Args:
            cg_instance: An instance of EnhancedConceptGraph.
            lg_instance: An instance of EnhancedLanguageGenerator (represents Layer 8).
            params: An optional dictionary of parameters to override defaults for various
                    engine and layer behaviors (e.g., decay rates, learning rates).
        """
        self.cg = cg_instance  # Concept Graph instance
        self.lg = lg_instance  # Language Generator instance (L8)
        self.params = params if params is not None else {}

        # Retrieve default parameters from the spec, allowing overrides from self.params
        self.decay_awake: float = self.params.get('decay_awake', 0.02)
        self.decay_dream: float = self.params.get('decay_dream', 0.05)
        self.hebbian_multiplier: float = self.params.get('hebbian_multiplier', 1.1)
        self.prune_threshold: float = self.params.get('prune_threshold', 0.2)
        self.dream_n_candidates: int = self.params.get('dream_n_candidates', 5)
        self.dream_keep_top_k: int = self.params.get('dream_keep_top_k', 2)
        self.low_ticks: int = self.params.get('low_ticks', 3) # General ticks for non-query cycles if needed

        # Initialize shared engine state. This state will be passed to layers that need it.
        # This dictionary holds dynamic information updated and used by various layers during a cycle.
        self.state: Dict[str, Any] = {
            "cg": self.cg, # Direct reference to the concept graph instance
            "narrative": self.lg.narrative if self.lg else [], # Reference to lg's narrative log
            "query": None, # Current input query string
            "last_response": "", # Last generated linguistic response
            "valence": 0.0, # Current affective valence (-1.0 to 1.0)
            "reflection": [], # List of concepts from symbolic reflection
            "metrics": {}, # Dictionary of graph/engine metrics
            "clusters": {}, # Dictionary of concept clusters
            "dream_log": [], # List of strings representing dream narratives
            "retrieved_patterns": [] # List of patterns retrieved by LearningOrchestrator
        }

        # Instantiate layers (L0-L14 based on 15-Layer spec)
        # L0 Geometry Runtime Substrate (handles activation decay)
        self.l0_geometry = GeometryLayer(decay_rate=self.decay_awake)
        # L1 Symbolic Tagging & Typing
        self.l1_symbolic_tagging = SymbolicTaggingLayer()
        # L2 Activation Dynamics (CA)
        self.l2_activation_dynamics = DynamicsLayer()
        # L3 Feedback Modulation
        self.l3_feedback_modulation = FeedbackModulationLayer() 
        # L4 Hebbian Edge Memory
        self.l4_hebbian_memory = HebbianLayer(multiplier=self.hebbian_multiplier, prune_threshold=self.prune_threshold)
        # L5 Identity Tracking
        self.l5_identity_tracking = IdentityTrackingLayer()
        # L6 Symbolic Reflection
        self.l6_symbolic_reflection = SymbolicReflectionLayer()
        # L7 Metrics Observation
        self.l7_metrics_observation = MetricsObservationLayer()
        # L8 Language Module (is self.lg, passed in)
        # L9 Affective Body
        self.l9_affective_body = AffectiveBodyLayer()
        # L10 Social Mind
        self.l10_social_mind = SocialMindLayer()
        # L11 Offline Dreaming
        self.l11_offline_dreaming = DreamLayer(lg_instance=self.lg, n_candidates=self.dream_n_candidates, keep_top_k=self.dream_keep_top_k)
        # L12 Learning Orchestrator
        self.l12_learning_orchestrator = LearningOrchestrator(shared_engine_state=self.state)
        # L13 Monitoring / Telemetry
        self.l13_monitoring = MonitoringLayer()
        # L14 API Embedder / Interface
        self.l14_api_embedder = APIEmbedderLayer()
            
    def run_cycle(self, query: Optional[str] = None, is_dream_cycle: bool = False) -> Optional[str]:
        """
        Executes a single cognitive cycle, processing through all defined layers.
        The behavior of the cycle can change based on whether it's a dream cycle
        or a standard cycle processing a query.

        Args:
            query: An optional string representing the user's input or query.
                   Typically provided for standard cycles.
            is_dream_cycle: A boolean indicating if this is a dream cycle.
                            Defaults to False.

        Returns:
            The linguistic response generated by the engine (if any),
            or None if no response is generated (e.g., during a dream cycle).
        """
        self.state["query"] = query
        if query and not is_dream_cycle: # Clear previous response only for new, non-dream queries
            self.state["last_response"] = ""

        # --- Layer Execution Order (based on a general cognitive flow) ---
        
        # L0 Geometry (Perception/Decay)
        # Adjust decay rate based on cycle type
        current_decay = self.decay_dream if is_dream_cycle else self.decay_awake
        original_l0_decay = self.l0_geometry.decay_rate
        try:
            self.l0_geometry.decay_rate = current_decay
            self.l0_geometry.run(self.cg)
        finally:
            self.l0_geometry.decay_rate = original_l0_decay # Ensure restoration

        # L1 Symbolic Tagging (Basic graph maintenance)
        self.l1_symbolic_tagging.run(self.state)
        
        # L2 Activation Dynamics (Graph propagation)
        self.l2_activation_dynamics.run(self.cg)
        
        # L9 Affective Body (Update valence based on previous cycle's output if any)
        # This might run earlier or later depending on when last_response is finalized.
        # For now, run it before feedback modulation that might use the new valence.
        if not is_dream_cycle: # Affective body might not run or run differently during dreams
             self.l9_affective_body.run(self.state)

        # L3 Feedback Modulation (Valence influences activations)
        # Ensure L9 (AffectiveBody) runs before L3 if L3 depends on fresh valence
        self.l3_feedback_modulation.last_sentiment = self.state.get("valence", 0.0) # Update layer's sentiment
        self.l3_feedback_modulation.run(self.state)
        
        # L4 Hebbian Edge Memory (Learning based on new activations)
        self.l4_hebbian_memory.run(self.cg)
        
        # L5 Identity Tracking (Clustering based on types)
        self.l5_identity_tracking.run(self.state)
        
        # L6 Symbolic Reflection (Capture highly active path)
        self.l6_symbolic_reflection.run(self.state)

        # L7 Metrics Observation (Observe current graph state)
        self.l7_metrics_observation.run(self.state)
        
        # Conditional Layer Execution (Dreaming vs. Query Processing)
        if is_dream_cycle:
            # L11 Offline Dreaming
            self.l11_offline_dreaming.run(self.cg, self.state) # DreamLayer logs to state['dream_log']
        elif query and self.lg:
            # L8 Language Module (Generate response if query)
            self.state["last_response"] = self.lg.generate(query)
        
        # L10 Social Mind (Dampen activations if negative valence)
        # Run after L9 has potentially updated valence
        self.l10_social_mind.run(self.state)
        
        # L12 Learning Orchestrator (Record patterns, retrieve)
        self.l12_learning_orchestrator.step() # Assumes step processes current state
        
        # L13 Monitoring / Telemetry (Output internal metrics/logs)
        # Behavior might be configured (e.g., log level)
        self.l13_monitoring.run(self.state)
        
        # L14 API Embedder / Interface (Output final response)
        # Behavior might be configured (e.g., verbosity)
        self.l14_api_embedder.run(self.state)

        return self.state.get("last_response")

    # --- Convenience methods for standard and dream cycles ---
    def standard_cycle(self, query: str) -> Optional[str]:
        """
        Runs a standard cognitive cycle to process a given query.

        Args:
            query: The input string (user query).

        Returns:
            The linguistic response from the engine.
        """
        # print("\n--- Running Standard Cycle ---") # Optional debug print
        return self.run_cycle(query=query, is_dream_cycle=False)

    def dream_cycle_execution(self) -> Optional[str]:
        """
        Runs a dream cycle, typically without a direct external query.
        This is used for internal processing, consolidation, or pattern exploration.

        Returns:
            The result of the cycle, which might be None or an internal state representation
            depending on how L14 (API Embedder) is configured for dream cycles.
            Typically, the primary output is to self.state['dream_log'].
        """
        # print("\n--- Running Dream Cycle ---") # Optional debug print
        return self.run_cycle(query=None, is_dream_cycle=True)
