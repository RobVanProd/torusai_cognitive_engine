import json
import time
from typing import Dict, Any, TYPE_CHECKING, Tuple, Deque
from collections import deque # For PatternMemory.buf if loaded directly

if TYPE_CHECKING:
    from ..engine.torus_engine import TorusEngine
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph
    from ..layer_12_learning_orchestrator.orchestrator import PatternMemory

def save_engine_state(engine: 'TorusEngine', file_path: str) -> bool:
    """
    Saves the crucial state of the TorusEngine to a JSON file.
    This includes the concept graph (nodes and edges) and pattern memory.

    Args:
        engine: The TorusEngine instance to save.
        file_path: The path to the file where the state will be saved.

    Returns:
        True if saving was successful, False otherwise.
    """
    try:
        state_to_save = {
            "concept_graph": {
                "nodes": engine.cg.nodes,
                "edges": {str(k): v for k, v in engine.cg.edges.items()} # Convert tuple keys to strings
            },
            "pattern_memory": {
                # PatternMemory.buf is a deque of tuples. JSON can handle lists of lists/tuples.
                "buf": list(engine.l12_learning_orchestrator.pattern_memory.buf),
                "max_items": engine.l12_learning_orchestrator.pattern_memory.max_items,
                "written": engine.l12_learning_orchestrator.pattern_memory.written,
                "last_prune": engine.l12_learning_orchestrator.pattern_memory.last_prune
            },
            "engine_params": engine.params, # Save engine construction parameters
            "timestamp": time.time()
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, indent=4, ensure_ascii=False)
        print(f"Engine state successfully saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving engine state to {file_path}: {e}")
        return False

def load_engine_state(file_path: str, engine: 'TorusEngine') -> bool:
    """
    Loads engine state from a JSON file into an existing TorusEngine instance.
    This populates the concept graph and pattern memory.
    The engine's lg_instance needs to be re-associated with the loaded cg.

    Args:
        file_path: The path to the file from which to load the state.
        engine: The TorusEngine instance to populate.

    Returns:
        True if loading was successful, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_state = json.load(f)

        # Load Concept Graph
        cg_data = loaded_state.get("concept_graph", {})
        engine.cg.nodes = cg_data.get("nodes", {})
        # Convert string keys back to tuples for edges
        engine.cg.edges = {eval(k): v for k, v in cg_data.get("edges", {}).items()}
        
        # Re-associate language generator with the loaded concept graph
        if engine.lg:
            engine.lg.cg = engine.cg # type: ignore

        # Load Pattern Memory
        pm_data = loaded_state.get("pattern_memory", {})
        # Ensure buf is a deque. It's saved as a list.
        engine.l12_learning_orchestrator.pattern_memory.buf = deque(pm_data.get("buf", []))
        engine.l12_learning_orchestrator.pattern_memory.max_items = pm_data.get("max_items", 
                                                                           engine.l12_learning_orchestrator.pattern_memory.MAX_ITEMS)
        engine.l12_learning_orchestrator.pattern_memory.written = pm_data.get("written", 0)
        engine.l12_learning_orchestrator.pattern_memory.last_prune = pm_data.get("last_prune", 0.0)
        
        # Optionally, re-initialize engine with loaded params if structure supports it,
        # or just note them. For now, we assume engine is already initialized and we're
        # just restoring the dynamic parts (CG, PM).
        # loaded_engine_params = loaded_state.get("engine_params", {})
        # print(f"Engine parameters loaded from file (for reference): {loaded_engine_params}")


        # Update engine's internal state dictionary references if necessary,
        # though direct modification of cg and pattern_memory objects should suffice.
        engine.state["cg"] = engine.cg
        # engine.state["narrative"] is tied to lg, which is now re-tied to cg.

        print(f"Engine state successfully loaded from {file_path}")
        print(f"Loaded CG: {len(engine.cg.nodes)} nodes, {len(engine.cg.edges)} edges.")
        print(f"Loaded PM: {len(engine.l12_learning_orchestrator.pattern_memory.buf)} items.")
        return True
    except Exception as e:
        print(f"Error loading engine state from {file_path}: {e}")
        return False

if __name__ == '__main__':
    # Example Usage (requires TorusEngine, EnhancedConceptGraph, etc. to be importable)
    # This is a conceptual test. Full testing requires an initialized engine.

    print("This script provides save_engine_state and load_engine_state functions.")
    print("To test, you would typically:")
    # 1. Initialize a TorusEngine instance.
    # from ..engine.torus_engine import TorusEngine
    # from ..layer_04_dcg.concept_graph import EnhancedConceptGraph
    # from ..layer_13_imr.language_generator import EnhancedLanguageGenerator
    #
    # cg_test = EnhancedConceptGraph()
    # lg_test = EnhancedLanguageGenerator(cg_test, [])
    # test_engine = TorusEngine(cg_instance=cg_test, lg_instance=lg_test)
    #
    # # 2. Populate it with some data (e.g., run some cycles, or use document processor)
    # test_engine.cg.addEnhancedNode("test_node", {"data": "sample"})
    # test_engine.cg.addEdge("test_node", "test_node") # Self-edge for testing
    # test_engine.l12_learning_orchestrator.pattern_memory.record("trigger", "response")
    #
    # # 3. Save the state
    # save_file = "test_engine_state.json"
    # if save_engine_state(test_engine, save_file):
    #     print(f"Test state saved to {save_file}")
    #
    #     # 4. Create a new engine instance (or clear the old one)
    #     cg_load = EnhancedConceptGraph()
    #     lg_load = EnhancedLanguageGenerator(cg_load, [])
    #     load_engine = TorusEngine(cg_instance=cg_load, lg_instance=lg_load)
    #
    #     # 5. Load the state
    #     if load_engine_state(save_file, load_engine):
    #         print(f"Test state loaded from {save_file}")
    #         print(f"Loaded nodes: {load_engine.cg.nodes}")
    #         print(f"Loaded pattern memory items: {len(load_engine.l12_learning_orchestrator.pattern_memory.buf)}")
    pass