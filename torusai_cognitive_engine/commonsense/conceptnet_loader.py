from typing import List, TYPE_CHECKING, Set
import csv # Assuming ConceptNet dump might be in CSV or similar parsable format

if TYPE_CHECKING:
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph

# Define a set of ConceptNet relations we might be interested in
# This would be expanded based on the actual ConceptNet relation types.
DEFAULT_CONCEPTNET_RELATIONS_TO_INCLUDE: Set[str] = {
    "/r/RelatedTo",
    "/r/IsA",
    "/r/PartOf",
    "/r/CapableOf",
    "/r/Causes",
    "/r/HasProperty",
    # Add more as needed
}

def normalize_conceptnet_term(term: str) -> str:
    """Normalizes a ConceptNet term to be used as a CID."""
    # ConceptNet terms are often like /c/en/apple
    if term.startswith("/c/en/"):
        term = term[5:]
    return term.replace("_", " ").lower() # Match graph population style

def load_conceptnet_into_graph(
    cg: 'EnhancedConceptGraph', 
    conceptnet_filepath: str, 
    relations_to_include: Set[str] = DEFAULT_CONCEPTNET_RELATIONS_TO_INCLUDE,
    limit_rows: int = -1 # Process all rows by default
) -> int:
    """
    Parses a ConceptNet data file (e.g., CSV dump) and adds relevant
    concepts and relations to the EnhancedConceptGraph.

    Args:
        cg: The EnhancedConceptGraph instance to populate.
        conceptnet_filepath: Path to the ConceptNet data file.
        relations_to_include: A set of ConceptNet relation types to include.
        limit_rows: Max number of rows to process from ConceptNet file (for testing).

    Returns:
        The number of edges successfully added from ConceptNet.
    """
    print(f"Attempting to load ConceptNet data from: {conceptnet_filepath}")
    edges_added_count = 0
    try:
        with open(conceptnet_filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t') # ConceptNet often uses tabs
            for i, row in enumerate(reader):
                if limit_rows > 0 and i >= limit_rows:
                    print(f"Reached row limit of {limit_rows} for ConceptNet processing.")
                    break
                
                if len(row) < 3: # Expecting at least relation, start_node, end_node
                    continue
                
                # Example ConceptNet CSV columns (actual format might vary):
                # uri, relation, start, end, json_metadata
                # /r/RelatedTo /c/en/cat /c/en/dog "{...}"
                
                relation_uri = row[1] # e.g., /r/IsA
                start_node_uri = row[2] # e.g., /c/en/apple
                end_node_uri = row[3]   # e.g., /c/en/fruit
                
                # Optional: Parse metadata for weight/confidence if available
                # weight = 1.0 # Default weight for ConceptNet edges
                # try:
                #     metadata = json.loads(row[4])
                #     weight = metadata.get("weight", 1.0)
                # except: pass


                if relation_uri in relations_to_include:
                    s_text = normalize_conceptnet_term(start_node_uri)
                    o_text = normalize_conceptnet_term(end_node_uri)
                    relation_short = relation_uri.replace("/r/", "cn_") # e.g., cn_IsA

                    s_cid = s_text # Using normalized text as CID directly
                    o_cid = o_text

                    # Add nodes if they don't exist
                    if s_cid not in cg.nodes:
                        cg.addEnhancedNode(s_cid, properties={"source": "conceptnet"}, linguistics={"wordForms": {"base": s_text}})
                    if o_cid not in cg.nodes:
                        cg.addEnhancedNode(o_cid, properties={"source": "conceptnet"}, linguistics={"wordForms": {"base": o_text}})
                    
                    # Add edge
                    if not cg.edges.get((s_cid, o_cid)) or \
                       cg.edges.get((s_cid, o_cid), {}).get("relation") != relation_short: # Avoid duplicate relations if loading multiple times
                        cg.addEdge(s_cid, o_cid, relation=relation_short, weight=0.7) # Default weight for CN
                        edges_added_count += 1
                
                if i % 50000 == 0 and i > 0:
                    print(f"Processed {i} lines from ConceptNet...")

        print(f"Finished loading ConceptNet data. Added {edges_added_count} new edges.")
    except FileNotFoundError:
        print(f"Error: ConceptNet file not found at {conceptnet_filepath}")
    except Exception as e:
        print(f"Error processing ConceptNet file: {e}")
    return edges_added_count

if __name__ == '__main__':
    # This would require a sample ConceptNet CSV file to test.
    # Example: Create a dummy 'conceptnet_sample.csv' with tab-separated values:
    # /a/something	/r/IsA	/c/en/cat	/c/en/animal	{"dataset": "/d/conceptnet/4/en", "weight": 2.0}
    # /a/something	/r/RelatedTo	/c/en/cat	/c/en/pet	{"dataset": "/d/conceptnet/4/en", "weight": 1.5}
    # /a/something	/r/PartOf	/c/en/wheel	/c/en/car	{"dataset": "/d/conceptnet/4/en", "weight": 1.0}
    
    # from ..layer_04_dcg.concept_graph import EnhancedConceptGraph
    # test_cg = EnhancedConceptGraph()
    # sample_cn_file = "conceptnet_sample.csv" 
    # # Create the sample file for testing:
    # with open(sample_cn_file, "w", encoding="utf-8") as f:
    #     f.write("/a/something\t/r/IsA\t/c/en/cat\t/c/en/animal\t{\"weight\": 2.0}\n")
    #     f.write("/a/something\t/r/RelatedTo\t/c/en/cat\t/c/en/pet\t{\"weight\": 1.5}\n")
    #     f.write("/a/something\t/r/PartOf\t/c/en/wheel\t/c/en/car\t{\"weight\": 1.0}\n")
    #     f.write("/a/something\t/r/MadeOf\t/c/en/table\t/c/en/wood\t{\"weight\": 1.0}\n") # This relation is not in default set

    # count = load_conceptnet_into_graph(test_cg, sample_cn_file, limit_rows=10)
    # print(f"\n--- ConceptNet Test Graph ---")
    # print(f"Nodes ({len(test_cg.nodes)}):")
    # for cid, data in test_cg.nodes.items(): print(f"  {cid}: {data}")
    # print(f"\nEdges ({len(test_cg.edges)}):")
    # for (s,t), data in test_cg.edges.items(): print(f"  ({s}, {t}): {data}")
    pass