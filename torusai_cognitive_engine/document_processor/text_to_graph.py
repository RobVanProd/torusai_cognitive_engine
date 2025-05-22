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
    STOP_WORDS = set() # type: ignore
    # This print might be too verbose if NLTK is intentionally not used when spaCy is primary.
    # print("NLTK library not found. Some fallback text processing capabilities will be limited. Install via 'pip install nltk'")

# Attempt to import spacy, provide guidance if missing
try:
    import spacy
    # Load a spaCy model. You might want to make this configurable.
    # Small model for efficiency, larger models (md, lg) for accuracy.
    try:
        NLP_SPACY = spacy.load('en_core_web_sm')
    except OSError:
        print("spaCy 'en_core_web_sm' model not found. Please run: python -m spacy download en_core_web_sm")
        NLP_SPACY = None
except ImportError:
    NLP_SPACY = None # type: ignore
    print("spaCy library not found. Advanced NLP features will be unavailable. Install via 'pip install spacy' and download a model.")


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

def extract_concepts_relationships_spacy(text_chunk: str, cg: 'EnhancedConceptGraph') -> None:
    """
    Extracts concepts (noun chunks, named entities) and basic SVO relationships
    from a text chunk using spaCy and populates the concept graph.
    """
    if not NLP_SPACY:
        print("spaCy model not loaded. Cannot perform advanced NLP extraction.")
        # Fallback to simple NLTK-based concept extraction for this chunk
        concepts_in_chunk = extract_concepts_from_chunk_simple(text_chunk)
        for concept_word in concepts_in_chunk:
            cid = concept_word
            if cid not in cg.nodes:
                cg.addEnhancedNode(cid, properties={"source": "document_import_nltk_fallback"},
                                   linguistics={"wordForms": {"base": concept_word}})
            else:
                cg.nodes[cid]["doc_freq"] = cg.nodes[cid].get("doc_freq", 0) + 1
        # Simple co-occurrence for fallback
        for i in range(len(concepts_in_chunk)):
            for j in range(i + 1, len(concepts_in_chunk)):
                s_cid, t_cid = concepts_in_chunk[i], concepts_in_chunk[j]
                if not cg.edges.get((s_cid, t_cid)):
                    cg.addEdge(s_cid, t_cid, relation="co-occurs_doc_fallback", weight=0.2)
        return

    doc = NLP_SPACY(text_chunk)
    
    # 1. Extract Named Entities and Noun Chunks as concepts
    extracted_concept_texts = set() # To avoid duplicate node additions from same chunk text

    for ent in doc.ents:
        ent_text = ent.text.lower().strip()
        if ent_text and len(ent_text) > 1: # Basic filter
            extracted_concept_texts.add(ent_text)
            if ent_text not in cg.nodes:
                cg.addEnhancedNode(ent_text, properties={"source": "document_ner", "entity_type": ent.label_},
                                   linguistics={"wordForms": {"base": ent_text}})
            else:
                cg.nodes[ent_text]["doc_freq"] = cg.nodes[ent_text].get("doc_freq", 0) + 1
                if "entity_type" not in cg.nodes[ent_text] and ent.label_: # Add entity type if missing
                    cg.nodes[ent_text]["entity_type"] = ent.label_


    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        # Further clean noun chunks: remove leading/trailing stopwords or articles if desired
        # For now, use the chunk as is if it's meaningful
        if chunk_text and len(chunk_text) > 2 and chunk_text not in STOP_WORDS: # Basic filter
            extracted_concept_texts.add(chunk_text)
            if chunk_text not in cg.nodes:
                cg.addEnhancedNode(chunk_text, properties={"source": "document_noun_chunk"},
                                   linguistics={"wordForms": {"base": chunk_text}})
            else:
                cg.nodes[chunk_text]["doc_freq"] = cg.nodes[chunk_text].get("doc_freq", 0) + 1
    
    # 2. Extract basic SVO-like relationships using dependency parsing
    for sent in doc.sents:
        # Example: Find simple nsubj-verb-dobj relations
        # This is a very simplified SVO extraction. Real SVO is more complex.
        subjects = [token for token in sent if token.dep_ == "nsubj" or token.dep_ == "nsubjpass"]
        direct_objects = [token for token in sent if token.dep_ == "dobj" or token.dep_ == "pobj"] # pobj for prepositional objects
        
        for token in sent:
            if token.pos_ == "VERB":
                verb_text = token.lemma_.lower() # Use lemma for verb
                # Add verb as a concept if it's not a stopword (auxiliaries might be filtered)
                if verb_text not in cg.nodes and verb_text not in STOP_WORDS and len(verb_text) > 1:
                    cg.addEnhancedNode(verb_text, properties={"source": "document_verb", "pos": "VERB"},
                                       linguistics={"wordForms": {"base": verb_text}})
                
                for subj_token in subjects:
                    # Use noun chunk text if subject is part of one, otherwise lemma
                    subj_concept = next((nc.text.lower().strip() for nc in doc.noun_chunks if nc.start <= subj_token.i < nc.end), subj_token.lemma_.lower())
                    if subj_concept not in extracted_concept_texts and subj_concept not in cg.nodes: # Add if new
                         if len(subj_concept)>1 and subj_concept not in STOP_WORDS :
                            cg.addEnhancedNode(subj_concept, properties={"source":"document_subj"}, linguistics={"wordForms":{"base":subj_concept}})
                            extracted_concept_texts.add(subj_concept)


                    if subj_concept in cg.nodes and verb_text in cg.nodes:
                         # Relation: subject -> verb
                        if not cg.edges.get((subj_concept, verb_text)):
                            cg.addEdge(subj_concept, verb_text, relation="subject_of", weight=0.5)
                        else:
                            cg.edges[(subj_concept, verb_text)]["weight"] = min(cg.edges[(subj_concept, verb_text)].get("weight",0)+0.1, 5.0)


                    for obj_token in direct_objects:
                        # Use noun chunk text if object is part of one, otherwise lemma
                        obj_concept = next((nc.text.lower().strip() for nc in doc.noun_chunks if nc.start <= obj_token.i < nc.end), obj_token.lemma_.lower())
                        if obj_concept not in extracted_concept_texts and obj_concept not in cg.nodes: # Add if new
                            if len(obj_concept)>1 and obj_concept not in STOP_WORDS:
                                cg.addEnhancedNode(obj_concept, properties={"source":"document_obj"}, linguistics={"wordForms":{"base":obj_concept}})
                                extracted_concept_texts.add(obj_concept)
                        
                        if verb_text in cg.nodes and obj_concept in cg.nodes:
                            # Relation: verb -> object
                            if not cg.edges.get((verb_text, obj_concept)):
                                cg.addEdge(verb_text, obj_concept, relation="direct_object_of" if obj_token.dep_ == "dobj" else "prepositional_object_of", weight=0.5)
                            else:
                                cg.edges[(verb_text, obj_concept)]["weight"] = min(cg.edges[(verb_text, obj_concept)].get("weight",0)+0.1, 5.0)
                        
                        # Also connect subject directly to object with verb as relation for some contexts
                        if subj_concept in cg.nodes and obj_concept in cg.nodes:
                            if not cg.edges.get((subj_concept, obj_concept)) or cg.edges.get((subj_concept, obj_concept),{}).get("relation") != verb_text : # Avoid overwriting stronger relations
                                cg.addEdge(subj_concept, obj_concept, relation=f"verb_{verb_text}", weight=0.4)


def populate_graph_from_text(text: str, cg: 'EnhancedConceptGraph', engine_state: Dict[str, Any]) -> None:
    """
    Processes a block of text, extracts concepts and relationships,
    and populates the concept graph. Uses spaCy if available, otherwise NLTK fallback.

    Args:
        text: The input text to process.
        cg: The EnhancedConceptGraph instance to populate.
        engine_state: The shared engine state (currently unused in this version).
    """
    print(f"Populating graph from text (length: {len(text)})...")
    if not text.strip():
        print("No text content to process.")
        return

    # Using spaCy for sentence segmentation and richer NLP if available
    if NLP_SPACY:
        doc = NLP_SPACY(text)
        sentences = [sent.text for sent in doc.sents]
        print(f"Using spaCy for processing. Found {len(sentences)} sentences.")
    elif nltk: # Fallback to NLTK sentence tokenizer
        sentences = sent_tokenize(text)
        print(f"Using NLTK for sentence tokenization. Found {len(sentences)} sentences.")
    else: # Very basic fallback if no NLP library
        sentences = text.split('.') # Simplistic sentence split
        print(f"No NLP library (spaCy/NLTK) for sentence tokenization. Using basic split. Found {len(sentences)} potential sentences.")

    processed_sentences = 0
    for sentence_text in sentences:
        if not sentence_text.strip():
            continue
        
        if NLP_SPACY:
            extract_concepts_relationships_spacy(sentence_text, cg)
        else:
            # Fallback to simpler NLTK-based concept extraction and co-occurrence links per sentence
            concepts_in_sentence = extract_concepts_from_chunk_simple(sentence_text)
            for concept_word in concepts_in_sentence:
                cid = concept_word
                if cid not in cg.nodes:
                    cg.addEnhancedNode(cid, properties={"source": "document_import_nltk"},
                                       linguistics={"wordForms": {"base": concept_word}})
                else:
                    cg.nodes[cid]["doc_freq"] = cg.nodes[cid].get("doc_freq", 0) + 1
            
            for i in range(len(concepts_in_sentence)):
                for j in range(i + 1, len(concepts_in_sentence)):
                    s_cid, t_cid = concepts_in_sentence[i], concepts_in_sentence[j]
                    if not cg.edges.get((s_cid, t_cid)):
                        cg.addEdge(s_cid, t_cid, relation="co-occurs_sent_nltk", weight=0.1)
        
        processed_sentences += 1
        if processed_sentences % 50 == 0: # Log progress every 50 sentences
            print(f"Processed {processed_sentences}/{len(sentences)} sentences...")

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