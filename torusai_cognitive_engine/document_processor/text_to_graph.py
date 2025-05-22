from typing import Dict, Any, List, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph

# Attempt to import nltk, provide guidance if missing
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    # Download necessary NLTK data if not present (run once)
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    
    STOP_WORDS = set(stopwords.words('english'))
except ImportError:
    nltk = None # type: ignore
    STOP_WORDS = set()
    print("NLTK library not found. Text processing capabilities will be limited. Install via 'pip install nltk'")


def simple_chunker(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of a specified word size with overlap.
    A very basic chunker.
    """
    words = text.split()
    if not words:
        return []
    
    chunks = []
    current_pos = 0
    while current_pos < len(words):
        end_pos = min(current_pos + chunk_size, len(words))
        chunks.append(" ".join(words[current_pos:end_pos]))
        current_pos += (chunk_size - overlap)
        if current_pos >= end_pos and end_pos < len(words): # Ensure progress if overlap is large
             current_pos = end_pos 
    return chunks

def extract_concepts_from_chunk_simple(text_chunk: str) -> List[str]:
    """
    Extracts potential concepts (nouns and important words) from a text chunk.
    Simple version: lowercase, remove stopwords, keep words longer than 2 chars.
    Uses NLTK for POS tagging if available to prioritize nouns.
    """
    if not nltk:
        # Fallback if NLTK is not available
        words = re.findall(r'\b\w+\b', text_chunk.lower())
        return [word for word in words if word not in STOP_WORDS and len(word) > 2][:20] # Limit concepts per chunk

    tokens = word_tokenize(text_chunk.lower())
    
    # POS tagging to identify nouns
    tagged_tokens = nltk.pos_tag(tokens)
    
    concepts = []
    for word, tag in tagged_tokens:
        if word.isalnum() and word not in STOP_WORDS and len(word) > 2:
            # Prioritize nouns, but include other significant words
            if tag.startswith('NN'): # NN, NNS, NNP, NNPS
                concepts.append(word)
            # Optionally, add other types like adjectives (JJ) or verbs (VB) if desired
            # elif tag.startswith('JJ'):
            #     concepts.append(word)
    
    # If too few nouns, take other content words
    if not concepts:
        concepts = [word for word, tag in tagged_tokens if word.isalnum() and word not in STOP_WORDS and len(word) > 2 and not tag.startswith('NN')]

    # Deduplicate while preserving order (somewhat) and limit
    return list(dict.fromkeys(concepts))[:15] # Limit concepts per chunk

def populate_graph_from_text(text: str, cg: 'EnhancedConceptGraph', engine_state: Dict[str, Any]) -> None:
    """
    Processes a block of text, extracts concepts, and populates the concept graph.

    Args:
        text: The input text to process.
        cg: The EnhancedConceptGraph instance to populate.
        engine_state: The shared engine state (currently unused in this basic version).
    """
    print(f"Populating graph from text (length: {len(text)})...")
    if not text.strip():
        print("No text content to process.")
        return

    # For now, using a very simple chunking and concept extraction.
    # This would be a key area for more sophisticated NLP.
    
    chunks = simple_chunker(text, chunk_size=150, overlap=30) # Smaller chunks for more connections
    
    processed_chunks = 0
    for chunk_index, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        concepts_in_chunk = extract_concepts_from_chunk_simple(chunk)
        
        if not concepts_in_chunk:
            continue

        # Add concepts as nodes
        for concept_word in concepts_in_chunk:
            # Use the word itself as CID for simplicity. Could be normalized further.
            cid = concept_word 
            if cid not in cg.nodes:
                cg.addEnhancedNode(cid, properties={"source": "document_import"}, 
                                   linguistics={"wordForms": {"base": concept_word}})
            else: # Increment a counter or update a property if node exists
                cg.nodes[cid]["doc_freq"] = cg.nodes[cid].get("doc_freq", 0) + 1


        # Add edges between co-occurring concepts in the chunk
        # This creates a densely connected component for each chunk's concepts
        for i in range(len(concepts_in_chunk)):
            for j in range(i + 1, len(concepts_in_chunk)):
                s_cid = concepts_in_chunk[i]
                t_cid = concepts_in_chunk[j]
                
                # Add edges in both directions or choose one based on desired graph structure
                # For now, adding one way. Could also adjust weight based on co-occurrence.
                if not cg.edges.get((s_cid, t_cid)):
                    cg.addEdge(s_cid, t_cid, relation="co-occurs_doc", weight=0.3) 
                else: # Increment weight if edge exists
                    current_weight = cg.edges[(s_cid, t_cid)].get("weight", 0.0)
                    cg.edges[(s_cid, t_cid)]["weight"] = min(current_weight + 0.1, 5.0)


                # Optionally, add reverse edge for undirected feel for co-occurrence
                # if not cg.edges.get((t_cid, s_cid)):
                #    cg.addEdge(t_cid, s_cid, relation="co-occurs_doc", weight=0.3)


        processed_chunks +=1
        if processed_chunks % 10 == 0:
            print(f"Processed {processed_chunks}/{len(chunks)} chunks...")

    print(f"Finished populating graph. Total nodes: {len(cg.nodes)}, Total edges: {len(cg.edges)}")


if __name__ == '__main__':
    # Example Usage (requires EnhancedConceptGraph to be importable)
    # This assumes this script is run in an environment where the parent packages are accessible.
    # For direct execution, you might need to adjust sys.path or run as a module.
    
    # Mock EnhancedConceptGraph for standalone testing if needed
    class MockEnhancedConceptGraph:
        def __init__(self):
            self.nodes = {}
            self.edges = {}
        def addEnhancedNode(self, cid, properties=None, linguistics=None):
            self.nodes[cid] = {"properties": properties or {}, "linguistics": linguistics or {}}
            self.nodes[cid]["activation"] = 0.0 # Ensure activation for getActivation
            print(f"MockAddNode: {cid}")
        def addEdge(self, s_cid, t_cid, relation="assoc", weight=1.0):
            self.edges[(s_cid, t_cid)] = {"relation": relation, "weight": weight}
            print(f"MockAddEdge: {s_cid} -> {t_cid}")
        def getActivation(self, cid): # Required by some layers if used indirectly
            return self.nodes.get(cid, {}).get("activation", 0.0)
        def setActivation(self, cid, value): # Required by some layers
             if cid in self.nodes: self.nodes[cid]["activation"] = value


    # test_cg = MockEnhancedConceptGraph()
    # If running within the package structure, use the real one:
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph
    test_cg = EnhancedConceptGraph()
    
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a test sentence.
    TorusAI aims to build a cognitive engine. The engine uses a concept graph.
    Natural language processing is key for understanding text. We will parse PDF and DOCX files.
    Learning involves creating nodes and edges from document content.
    """
    mock_engine_state = {"cg": test_cg} # Other state elements might be needed by more complex versions

    populate_graph_from_text(sample_text, test_cg, mock_engine_state)
    
    print("\n--- Test Graph Population Results ---")
    print(f"Nodes ({len(test_cg.nodes)}):")
    for cid, data in test_cg.nodes.items():
        print(f"  {cid}: {data}")
    print(f"\nEdges ({len(test_cg.edges)}):")
    for (s, t), data in test_cg.edges.items():
        print(f"  ({s}, {t}): {data}")