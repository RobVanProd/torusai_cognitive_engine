import time
import hashlib
import json # Added for potential serialization in LearningOrchestrator
from collections import deque
from typing import Any, Dict, List, Tuple, Optional, Deque

class PatternMemory:
    """
    A memory buffer for storing and retrieving (trigger, response, metadata) patterns.
    It uses a simple character-based vectorization for similarity matching and
    has a pruning mechanism to manage its size.
    """
    MAX_ITEMS: int = 50_000
    RETAIN_PCT: float = 0.8
    RECENT_N: int = 1000 # Number of recent items to consider for nearest neighbor search if buffer is large
    TOP_K: int = 5 # Default number of nearest neighbors to return
    THRESH: float = 0.35 # Default similarity threshold for nearest neighbors

    def __init__(self, max_items: int = MAX_ITEMS):
        """
        Initializes the PatternMemory.

        Args:
            max_items: The maximum number of items to store in the buffer before pruning.
        """
        self.max_items: int = max_items
        self.buf: Deque[Tuple[str, str, Dict[str, Any]]] = deque()
        # self.sha_idx: Dict[str, int] = {} # Index for quick lookup by trigger hash (currently unused)
        self.written: int = 0 # Total items ever written
        self.last_prune: float = 0.0 # Timestamp of the last prune operation

    def record(self, trigger: str, response: str,
               meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Records a new (trigger, response, metadata) pattern.
        If the buffer exceeds max_items, it prunes older items.

        Args:
            trigger: The trigger string.
            response: The response string associated with the trigger.
            meta: Optional dictionary of metadata associated with the pattern.
                  A timestamp 'ts' is automatically added.
        """
        if len(self.buf) >= self.max_items:
            self._prune()
        
        current_meta = meta if meta is not None else {}
        current_meta["ts"] = time.time()
        
        self.buf.append((trigger, response, current_meta))
        # If sha_idx were used for exact matching or deduplication:
        # self.sha_idx[hashlib.sha1(trigger.encode('utf-8')).hexdigest()] = len(self.buf) - 1
        self.written += 1

    def nearest(self, query: str, k: int = TOP_K, t: float = THRESH) -> List[Tuple[str, str, Dict[str, Any], float]]:
        """
        Finds the k nearest patterns to a given query string based on cosine similarity
        of their vectorized forms, above a certain threshold.

        Args:
            query: The query string to match against stored triggers.
            k: The maximum number of nearest patterns to return.
            t: The minimum similarity threshold for a pattern to be considered.

        Returns:
            A list of tuples, where each tuple contains:
            (trigger, response, metadata, similarity_score), sorted by similarity.
        """
        if not query:
            return []
            
        query_vector = self._vec(query)
        # Consider only recent items if buffer is large, otherwise all items
        candidate_items = list(self._recent()) if len(self.buf) >= self.RECENT_N else list(self.buf)
        
        if not candidate_items:
            return []

        similar_patterns: List[Tuple[str, str, Dict[str, Any], float]] = []
        for trig, resp, meta_info in candidate_items:
            trigger_vector = self._vec(trig)
            similarity = self._cos(query_vector, trigger_vector)
            if similarity >= t:
                similar_patterns.append((trig, resp, meta_info, similarity))
        
        # Sort by similarity in descending order and take top k
        return sorted(similar_patterns, key=lambda x: -x[3])[:k]

    def stats(self) -> Dict[str, Any]:
        """Returns statistics about the pattern memory."""
        return {"items": len(self.buf), "written": self.written,
                "last_prune_timestamp": self.last_prune}

    def _vec(self, s: str) -> List[int]:
        """
        Converts a string into a simple fixed-size integer vector.
        This is a basic character-frequency based vectorization.
        
        NOTE: For more sophisticated pattern matching, especially for semantic similarity,
        this should be replaced with advanced text embedding techniques (e.g.,
        Sentence Transformers, Word2Vec, FastText, or LLM-based embeddings).
        This is a critical area for future enhancement to achieve more human-like
        pattern recognition and analogy.
        """
        vector = [0]*128 # Fixed size vector
        # Consider only the first 1024 bytes to limit processing for very long strings
        for char_byte_value in s[:1024].encode('utf-8', 'ignore'):
            vector[char_byte_value % 128] += 1
        return vector

    def _cos(self, vec_a: List[int], vec_b: List[int]) -> float:
        """Calculates the cosine similarity between two integer vectors."""
        dot_product = sum(x * y for x, y in zip(vec_a, vec_b))
        
        norm_a_sq = sum(x * x for x in vec_a)
        norm_b_sq = sum(y * y for y in vec_b)
        
        if norm_a_sq == 0 or norm_b_sq == 0:
            return 0.0
            
        norm_a = norm_a_sq**0.5
        norm_b = norm_b_sq**0.5
        
        return dot_product / (norm_a * norm_b)

    def _recent(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Returns the N most recently added items from the buffer."""
        num_recent = min(len(self.buf), self.RECENT_N)
        # Iterating from the right end of the deque to get recent items
        return [self.buf[len(self.buf) - 1 - i] for i in range(num_recent)]


    def _prune(self) -> None:
        """Prunes the buffer to maintain max_items * RETAIN_PCT by removing oldest items."""
        items_to_keep = int(self.max_items * self.RETAIN_PCT)
        while len(self.buf) > items_to_keep:
            old_trigger, _, _ = self.buf.popleft()
            # If sha_idx were used for exact matches, it would need consistent updates:
            # self.sha_idx.pop(hashlib.sha1(old_trigger.encode('utf-8')).hexdigest(), None)
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
