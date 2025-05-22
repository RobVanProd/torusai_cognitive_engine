import time
import hashlib
import json
from collections import deque
from typing import Any, Dict, List, Tuple, Optional, Deque
import numpy as np # For embeddings

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None # type: ignore
    util = None # type: ignore
    print("Warning: sentence-transformers library not found. PatternMemory will use basic vectorization. Install via 'pip install sentence-transformers'")


class PatternMemory:
    """
    A memory buffer for storing and retrieving (trigger, response, metadata, trigger_embedding) patterns.
    It uses sentence embeddings for similarity matching if sentence-transformers is available,
    otherwise falls back to a simple character-based vectorization.
    Has a pruning mechanism to manage its size.
    """
    MAX_ITEMS: int = 50_000
    RETAIN_PCT: float = 0.8
    RECENT_N: int = 1000
    TOP_K: int = 5
    THRESH: float = 0.35 # Similarity threshold for embeddings (may need adjustment)
    DEFAULT_EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'

    def __init__(self, max_items: int = MAX_ITEMS, embedding_model_name: Optional[str] = None):
        """
        Initializes the PatternMemory.

        Args:
            max_items: The maximum number of items to store in the buffer before pruning.
            embedding_model_name: The name of the sentence-transformer model to use.
                                  Defaults to DEFAULT_EMBEDDING_MODEL.
        """
        self.max_items: int = max_items
        # Buffer now stores: (trigger_str, response_str, metadata_dict, trigger_embedding_vector)
        self.buf: Deque[Tuple[str, str, Dict[str, Any], Optional[np.ndarray]]] = deque()
        self.written: int = 0
        self.last_prune: float = 0.0
        
        self.embedding_model_name = embedding_model_name if embedding_model_name is not None else self.DEFAULT_EMBEDDING_MODEL
        self.embedding_model: Optional[SentenceTransformer] = None
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                print(f"PatternMemory: Loaded sentence-transformer model '{self.embedding_model_name}'.")
            except Exception as e:
                print(f"PatternMemory: Error loading sentence-transformer model '{self.embedding_model_name}': {e}. Falling back to basic vectorization.")
                self.embedding_model = None
        else:
            print("PatternMemory: sentence-transformers library not available. Using basic vectorization.")

    def record(self, trigger: str, response: str,
               meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Records a new pattern, including its trigger embedding.
        """
        if len(self.buf) >= self.max_items:
            self._prune()
        
        current_meta = meta if meta is not None else {}
        current_meta["ts"] = time.time()
        
        trigger_embedding: Optional[np.ndarray] = self._get_embedding(trigger)
        
        self.buf.append((trigger, response, current_meta, trigger_embedding))
        self.written += 1

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generates an embedding for the given text using the sentence transformer model."""
        if self.embedding_model and text:
            try:
                # Ensure text is a list for sentence-transformers batch processing, even if it's a single string
                return self.embedding_model.encode([text])[0]
            except Exception as e:
                print(f"PatternMemory: Error generating embedding for '{text[:30]}...': {e}")
                return None
        return None # Fallback if no model or empty text

    def _basic_vec(self, s: str) -> List[int]:
        """Fallback basic character-frequency vectorization."""
        vector = [0]*128
        for char_byte_value in s[:1024].encode('utf-8', 'ignore'):
            vector[char_byte_value % 128] += 1
        return vector

    def nearest(self, query: str, k: int = TOP_K, t: float = THRESH) -> List[Tuple[str, str, Dict[str, Any], float]]:
        """
        Finds nearest patterns using sentence embeddings if available, otherwise basic vectors.
        """
        if not query:
            return []
            
        candidate_items_tuples = list(self._recent()) if len(self.buf) >= self.RECENT_N else list(self.buf)
        if not candidate_items_tuples:
            return []

        similar_patterns: List[Tuple[str, str, Dict[str, Any], float]] = []

        if self.embedding_model and util:
            query_embedding: Optional[np.ndarray] = self._get_embedding(query)
            if query_embedding is None: # Could not embed query
                return []

            for trig_str, resp_str, meta_info, stored_trig_emb in candidate_items_tuples:
                if stored_trig_emb is not None:
                    # Calculate cosine similarity between numpy arrays
                    # util.cos_sim returns a tensor, convert to float
                    similarity_tensor = util.cos_sim(query_embedding, stored_trig_emb)
                    similarity = similarity_tensor.item() if similarity_tensor is not None else 0.0
                    
                    if similarity >= t:
                        similar_patterns.append((trig_str, resp_str, meta_info, similarity))
        else: # Fallback to basic vectorization
            query_vector_basic = self._basic_vec(query)
            for trig_str, resp_str, meta_info, _ in candidate_items_tuples: # Embedding not used here
                trigger_vector_basic = self._basic_vec(trig_str)
                similarity = self._cos_basic(query_vector_basic, trigger_vector_basic)
                if similarity >= t:
                    similar_patterns.append((trig_str, resp_str, meta_info, similarity))
        
        return sorted(similar_patterns, key=lambda x: -x[3])[:k]

    def stats(self) -> Dict[str, Any]:
        return {"items": len(self.buf), "written": self.written,
                "last_prune_timestamp": self.last_prune,
                "embedding_model_active": self.embedding_model is not None}

    def _cos_basic(self, vec_a: List[int], vec_b: List[int]) -> float:
        """Cosine similarity for basic integer vectors."""
        dot_product = sum(x * y for x, y in zip(vec_a, vec_b))
        norm_a_sq = sum(x * x for x in vec_a)
        norm_b_sq = sum(y * y for y in vec_b)
        if norm_a_sq == 0 or norm_b_sq == 0: return 0.0
        return dot_product / ((norm_a_sq**0.5) * (norm_b_sq**0.5))

    def _recent(self) -> List[Tuple[str, str, Dict[str, Any], Optional[np.ndarray]]]:
        num_recent = min(len(self.buf), self.RECENT_N)
        return [self.buf[len(self.buf) - 1 - i] for i in range(num_recent)]

    def _prune(self) -> None:
        items_to_keep = int(self.max_items * self.RETAIN_PCT)
        while len(self.buf) > items_to_keep:
            self.buf.popleft()
        self.last_prune = time.time()

class LearningOrchestrator:
    """
    Manages the learning process by recording salient (trigger, response) patterns
    from the engine's state and retrieving similar past patterns.
    Corresponds to L12 Learning Orchestrator in the 15-layer specification.
    """
    def __init__(self, shared_engine_state: Dict[str, Any]):
        """
        Initializes the LearningOrchestrator.

        Args:
            shared_engine_state: A reference to the main engine's shared state dictionary.
                                 This state is used to get reflection, last_response, etc.,
                                 and to store retrieved_patterns.
        """
        self.state: Dict[str, Any] = shared_engine_state
        self.pattern_memory: PatternMemory = PatternMemory()

    def _create_reflection_key(self, reflection_data: Any) -> str:
        """
        Creates a consistent string key from reflection data.
        Reflection data can be a string or a list of dicts (active concepts).
        If it's a list of dicts, it sorts the 'id' fields and joins them.
        """
        if isinstance(reflection_data, list) and reflection_data:
            # Create a sorted, joined string of concept IDs for consistency
            # Ensure items are dicts and have 'id' before trying to access.
            ids = sorted([
                item.get("id", "") for item in reflection_data
                if isinstance(item, dict) and item.get("id") is not None
            ])
            return "_".join(ids)
        elif isinstance(reflection_data, str):
            return reflection_data
        return "" # Default for empty or unexpected types

    def step(self) -> None:
        """
        Performs a learning step:
        1. Extracts reflection and last_response from the shared state.
        2. Creates a string key from the reflection data.
        3. Records the (reflection_key, last_response) pair in PatternMemory,
           along with current valence as metadata.
        4. Retrieves patterns from memory similar to the current reflection_key.
        5. Stores retrieved patterns back into the shared state ('retrieved_patterns').
        """
        reflection_data: Any = self.state.get("reflection")
        last_response_str: str = str(self.state.get("last_response", ""))
        current_valence: float = self.state.get("valence", 0.0)

        reflection_key: str = self._create_reflection_key(reflection_data)

        if reflection_key and last_response_str: # Only record if both are meaningful
            meta_data = {"valence_at_recording": current_valence, "timestamp": time.time()}
            self.pattern_memory.record(reflection_key, last_response_str, meta=meta_data)
        
        if reflection_key:
            retrieved_patterns: List[Tuple[str, str, Dict[str, Any], float]] = self.pattern_memory.nearest(reflection_key)
            self.state["retrieved_patterns"] = retrieved_patterns
        else:
            self.state["retrieved_patterns"] = [] # Ensure key exists even if no retrieval

# Optional: Comment out or remove the self-test block for final integration
# if __name__ == "__main__":
#     # Ensure json is imported if this block is used
#     # import json
#     example_state_list_reflection = {
#         "reflection": [{"id":"conceptA", "activation":0.8}, {"id":"conceptB", "activation":0.7}],
#         "last_response": "a generated response to conceptA and conceptB",
#         "valence": 0.5
#     }
#
#     orchestrator_list = LearningOrchestrator(example_state_list_reflection)
#     orchestrator_list.step()
#     # print("State after list reflection:", json.dumps(example_state_list_reflection, indent=2))
#
#     pm_list = orchestrator_list.pattern_memory
#     # print("\nPatternMemory Stats (list):", pm_list.stats())
#
#     reflection_as_key_list = orchestrator_list._create_reflection_key(example_state_list_reflection["reflection"])
#     # print(f"Nearest to '{reflection_as_key_list}':", pm_list.nearest(reflection_as_key_list))
#
#     example_state_str_reflection = {
#         "reflection": "simple_trigger_string",
#         "last_response": "response to simple trigger",
#         "valence": -0.2
#     }
#     orchestrator_str = LearningOrchestrator(example_state_str_reflection)
#     orchestrator_str.step()
#     # print("\nState after str reflection:", json.dumps(example_state_str_reflection, indent=2))
#     # print(f"Nearest to 'simple_trigger_string':", orchestrator_str.pattern_memory.nearest("simple_trigger_string"))
#
#     example_state_empty_reflection = {
#         "reflection": [],
#         "last_response": "response to empty reflection",
#         "valence": 0.1
#     }
#     orchestrator_empty = LearningOrchestrator(example_state_empty_reflection)
#     orchestrator_empty.step()
#     # print("\nState after empty reflection:", json.dumps(example_state_empty_reflection, indent=2))
#     reflection_as_key_empty = orchestrator_empty._create_reflection_key(example_state_empty_reflection["reflection"])
#     # print(f"Nearest to '{reflection_as_key_empty}':", orchestrator_empty.pattern_memory.nearest(reflection_as_key_empty))
