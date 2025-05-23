import sys
import os

# Add project root to sys.path to allow direct execution of script
# and proper imports from torusai_cognitive_engine
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from torusai_cognitive_engine.layer_04_dcg.concept_graph import EnhancedConceptGraph
from torusai_cognitive_engine.document_processor.text_to_graph import populate_graph_from_text
# UIE_MODEL_NAME is a global in text_to_graph, its default "t5-small" will be used.

if __name__ == "__main__":
    print("--- Starting Manual UIE Test ---")
    cg = EnhancedConceptGraph()
    
    sample_text = "Albert Einstein developed the theory of relativity in Germany. He was a physicist. Einstein was born in Ulm."

    print(f"Processing sample text: '{sample_text}'")
    
    # The populate_graph_from_text function will internally try to load the UIE model (t5-small by default)
    # and use it if UIE_MODEL_LOADED is True after the load attempt.
    # It will print loading messages from _load_uie_model_if_needed().
    populate_graph_from_text(sample_text, cg, {}) # engine_state is {} for this test

    print("\n--- Concept Graph Nodes ---")
    if not cg.nodes:
        print("No nodes found in the graph.")
    for cid, data in cg.nodes.items():
        print(f"Node: {cid}, Data: {data}")

    print("\n--- Concept Graph Edges ---")
    if not cg.edges:
        print("No edges found in the graph.")
    for (s_cid, t_cid), data in cg.edges.items():
        print(f"Edge: ({s_cid} -> {t_cid}), Data: {data}")
    
    print("\n--- Manual UIE Test Finished ---")
