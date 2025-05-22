from typing import Dict, Any, List, TYPE_CHECKING, Tuple # Added Tuple
import re

if TYPE_CHECKING:
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph

# Attempt to import nltk, provide guidance if missing
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
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
    # print("NLTK library not found...")

# Attempt to import spacy, provide guidance if missing
try:
    import spacy
    try:
        NLP_SPACY = spacy.load('en_core_web_sm')
    except OSError:
        print("spaCy 'en_core_web_sm' model not found. Please run: python -m spacy download en_core_web_sm")
        NLP_SPACY = None
except ImportError:
    NLP_SPACY = None # type: ignore
    # print("spaCy library not found...")

# Placeholder for UIE model integration
UIE_MODEL_LOADED = False # Set to True when a real UIE model is integrated
# from some_uie_library import UIEPipeline # Hypothetical
# uie_pipeline = None
# try:
#     # uie_pipeline = UIEPipeline(model="some-uie-model") # Hypothetical UIE model loading
#     # UIE_MODEL_LOADED = True 
#     # print("Conceptual UIE Model loaded successfully.")
#     pass # Keep UIE_MODEL_LOADED as False until a real model is chosen and integrated
# except Exception as e:
#     print(f"Conceptual UIE Model could not be loaded: {e}. Falling back to spaCy/NLTK.")
#     UIE_MODEL_LOADED = False


def normalize_cid(text: str) -> str:
    """Normalizes a text string to be used as a Concept ID."""
    if not isinstance(text, str): # Ensure text is a string
        text = str(text)
    return text.lower().replace(" ", "_").strip()

def extract_triples_via_uie(text_chunk: str) -> List[Dict[str, Any]]:
    """
    Placeholder for extracting (subject, predicate, object) triples using a UIE model.
    This function would call the actual UIE model.
    Returns a list of dictionaries, each like:
    {'subject': str, 'predicate': str, 'object': str, 'confidence': float, 'provenance_span': Tuple[int,int]}
    """
    # Mock implementation - replace with actual UIE model call
    # print(f"Conceptual UIE processing chunk: {text_chunk[:70]}...") # Keep this for debugging if UIE is active
    if "TorusAI aims to build a cognitive engine" in text_chunk and UIE_MODEL_LOADED:
        return [
            {"subject": "TorusAI", "predicate": "aims to build", "object": "cognitive engine", "confidence": 0.9, "provenance_span": (0,40)},
            {"subject": "cognitive engine", "predicate": "is_a_type_of", "object": "AI system", "confidence": 0.85, "provenance_span": (25,40)}
        ]
    # Add more mock examples if needed for testing the pipeline structure
    if "Shakespeare wrote Hamlet" in text_chunk and UIE_MODEL_LOADED:
        return [
            {"subject": "Shakespeare", "predicate": "wrote", "object": "Hamlet", "confidence": 0.95, "provenance_span": (0,24)}
        ]
    return []


def simple_chunker(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of a specified word size with overlap.
    A very basic chunker.
    """
    words = re.split(r'\s+', text) # Split by any whitespace
    if not words:
        return []
    
    chunks = []
    current_pos = 0
    while current_pos < len(words):
        end_pos = min(current_pos + chunk_size, len(words))
        chunks.append(" ".join(words[current_pos:end_pos]))
        current_pos += (chunk_size - overlap)
        if current_pos >= end_pos and end_pos < len(words): 
             current_pos = end_pos 
    return chunks

def extract_concepts_from_chunk_simple(text_chunk: str) -> List[str]:
    """
    Extracts potential concepts (nouns and important words) from a text chunk.
    Simple version: lowercase, remove stopwords, keep words longer than 2 chars.
    Uses NLTK for POS tagging if available to prioritize nouns.
    """
    if not nltk:
        words = re.findall(r'\b\w+\b', text_chunk.lower())
        return [word for word in words if word not in STOP_WORDS and len(word) > 2][:20] 

    tokens = word_tokenize(text_chunk.lower())
    tagged_tokens = nltk.pos_tag(tokens)
    
    concepts = []
    for word, tag in tagged_tokens:
        if word.isalnum() and word not in STOP_WORDS and len(word) > 2:
            if tag.startswith('NN'): 
                concepts.append(word)
    
    if not concepts: # If no nouns, take other content words
        for word, tag in tagged_tokens:
            if word.isalnum() and word not in STOP_WORDS and len(word) > 2 and not tag.startswith('NN'):
                concepts.append(word)
    
    return list(dict.fromkeys(concepts))[:15]

def extract_concepts_relationships_spacy(text_chunk: str, cg: 'EnhancedConceptGraph') -> None:
    """
    Extracts concepts (noun chunks, named entities) and basic SVO relationships
    from a text chunk using spaCy and populates the concept graph.
    """
    if not NLP_SPACY:
        # print("spaCy model not loaded. Using NLTK fallback for chunk.") # Already handled by caller
        concepts_in_chunk = extract_concepts_from_chunk_simple(text_chunk)
        for concept_word in concepts_in_chunk:
            cid = normalize_cid(concept_word)
            if cid not in cg.nodes:
                cg.addEnhancedNode(cid, properties={"source": "document_import_nltk_fallback"},
                                   linguistics={"wordForms": {"base": concept_word}})
            else:
                cg.nodes[cid]["doc_freq"] = cg.nodes[cid].get("doc_freq", 0) + 1
        for i in range(len(concepts_in_chunk)):
            for j in range(i + 1, len(concepts_in_chunk)):
                s_cid, t_cid = normalize_cid(concepts_in_chunk[i]), normalize_cid(concepts_in_chunk[j])
                if not cg.edges.get((s_cid, t_cid)):
                    cg.addEdge(s_cid, t_cid, relation="co-occurs_doc_fallback", weight=0.2)
        return

    doc = NLP_SPACY(text_chunk)
    extracted_concept_texts_in_chunk = set()

    for ent in doc.ents:
        ent_text = ent.text.lower().strip()
        if ent_text and len(ent_text) > 1:
            cid = normalize_cid(ent_text)
            extracted_concept_texts_in_chunk.add(cid)
            if cid not in cg.nodes:
                cg.addEnhancedNode(cid, properties={"source": "document_ner", "entity_type": ent.label_},
                                   linguistics={"wordForms": {"base": ent_text}})
            else:
                cg.nodes[cid]["doc_freq"] = cg.nodes[cid].get("doc_freq", 0) + 1
                if "entity_type" not in cg.nodes[cid] and ent.label_:
                    cg.nodes[cid]["entity_type"] = ent.label_

    for chunk_spacy in doc.noun_chunks: # Renamed from 'chunk' to avoid conflict
        chunk_text = chunk_spacy.text.lower().strip()
        if chunk_text and len(chunk_text) > 2 and chunk_text not in STOP_WORDS:
            cid = normalize_cid(chunk_text)
            extracted_concept_texts_in_chunk.add(cid)
            if cid not in cg.nodes:
                cg.addEnhancedNode(cid, properties={"source": "document_noun_chunk"},
                                   linguistics={"wordForms": {"base": chunk_text}})
            else:
                cg.nodes[cid]["doc_freq"] = cg.nodes[cid].get("doc_freq", 0) + 1
    
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() not in STOP_WORDS and len(token.lemma_) > 1:
            verb_cid = normalize_cid(token.lemma_)
            if verb_cid not in cg.nodes:
                 cg.addEnhancedNode(verb_cid, properties={"source": "document_verb", "pos": "VERB"},
                                   linguistics={"wordForms": {"base": token.lemma_.lower()}})
            
            # Basic SVO extraction
            subjects = [normalize_cid(child.lemma_.lower()) for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
            objects = [normalize_cid(child.lemma_.lower()) for child in token.children if child.dep_ in ("dobj", "pobj", "obj")]

            for subj_cid in subjects:
                if subj_cid in cg.nodes and verb_cid in cg.nodes:
                    if not cg.edges.get((subj_cid, verb_cid)):
                        cg.addEdge(subj_cid, verb_cid, relation="subject_of_verb", weight=0.6)
                    else:
                         cg.edges[(subj_cid, verb_cid)]["weight"] = min(cg.edges[(subj_cid, verb_cid)].get("weight",0)+0.1, 5.0)

                for obj_cid in objects:
                    if verb_cid in cg.nodes and obj_cid in cg.nodes:
                        if not cg.edges.get((verb_cid, obj_cid)):
                            cg.addEdge(verb_cid, obj_cid, relation="verb_has_object", weight=0.6)
                        else:
                            cg.edges[(verb_cid, obj_cid)]["weight"] = min(cg.edges[(verb_cid, obj_cid)].get("weight",0)+0.1, 5.0)
                    
                    # Connect subject to object via verb
                    if subj_cid in cg.nodes and obj_cid in cg.nodes:
                        rel_name = f"verb_{verb_cid}"
                        if not cg.edges.get((subj_cid, obj_cid)) or cg.edges.get((subj_cid, obj_cid),{}).get("relation") != rel_name:
                            cg.addEdge(subj_cid, obj_cid, relation=rel_name, weight=0.45)


def populate_graph_from_text(text: str, cg: 'EnhancedConceptGraph', engine_state: Dict[str, Any]) -> None:
    """
    Processes a block of text, extracts concepts and relationships,
    and populates the concept graph. Prefers UIE if available, then spaCy, then NLTK fallback.
    """
    print(f"Populating graph from text (length: {len(text)})...")
    if not text.strip():
        print("No text content to process.")
        return

    sentences: List[str] = []
    processing_method = "uie" if UIE_MODEL_LOADED else ("spacy" if NLP_SPACY else ("nltk" if nltk else "basic"))

    if processing_method == "spacy" and NLP_SPACY: # Ensure NLP_SPACY is not None
        doc = NLP_SPACY(text)
        sentences = [sent.text for sent in doc.sents]
        print(f"Using spaCy for sentence splitting. Found {len(sentences)} sentences.")
    elif processing_method == "nltk" and nltk:
        sentences = sent_tokenize(text)
        print(f"Using NLTK for sentence tokenization. Found {len(sentences)} sentences.")
    elif processing_method == "uie": 
        if NLP_SPACY: sentences = [sent.text for sent in NLP_SPACY(text).sents]
        elif nltk: sentences = sent_tokenize(text)
        else: sentences = text.split('.') # type: ignore
        print(f"Using UIE (conceptual) for triple extraction, sentence splitting via {NLP_SPACY or nltk or 'basic'}. Found {len(sentences)} sentences.")
    else: 
        sentences = text.split('.') # type: ignore
        print(f"No advanced NLP library for sentence tokenization. Using basic split. Found {len(sentences)} potential sentences.")

    processed_units = 0
    for unit_text in sentences: 
        if not unit_text.strip():
            continue
        
        if UIE_MODEL_LOADED: # Primary choice: UIE
            triples = extract_triples_via_uie(unit_text)
            if not triples and NLP_SPACY: # Fallback to spaCy if UIE returns nothing for the chunk
                 extract_concepts_relationships_spacy(unit_text, cg)
            else:
                for triple in triples:
                    subj_text = triple.get("subject","").strip()
                    pred_text = triple.get("predicate","").strip()
                    obj_text = triple.get("object","").strip()
                    conf = triple.get("confidence", 0.5)
                    # span = triple.get("provenance_span") # TODO: Store span

                    if not (subj_text and pred_text and obj_text):
                        continue

                    s_cid = normalize_cid(subj_text)
                    p_relation_str = normalize_cid(pred_text) # Predicate becomes the relation string
                    o_cid = normalize_cid(obj_text)

                    if s_cid not in cg.nodes:
                        cg.addEnhancedNode(s_cid, properties={"source": "uie_import"}, linguistics={"wordForms": {"base": subj_text}})
                    else:
                        cg.nodes[s_cid]["doc_freq"] = cg.nodes[s_cid].get("doc_freq", 0) + 1
                    
                    if o_cid not in cg.nodes:
                        cg.addEnhancedNode(o_cid, properties={"source": "uie_import"}, linguistics={"wordForms": {"base": obj_text}})
                    else:
                        cg.nodes[o_cid]["doc_freq"] = cg.nodes[o_cid].get("doc_freq", 0) + 1
                    
                    current_edge = cg.edges.get((s_cid, o_cid))
                    if not current_edge or current_edge.get("relation") != p_relation_str:
                        cg.addEdge(s_cid, o_cid, relation=p_relation_str, weight=float(conf))
                    else: 
                        existing_weight = current_edge.get("weight", 0.0)
                        cg.edges[(s_cid, o_cid)]["weight"] = max(existing_weight, float(conf))

        elif NLP_SPACY: # Secondary choice: spaCy
            extract_concepts_relationships_spacy(unit_text, cg)
        elif nltk: # Tertiary choice: NLTK basic
            concepts_in_unit = extract_concepts_from_chunk_simple(unit_text)
            for concept_word in concepts_in_unit:
                cid = normalize_cid(concept_word)
                if cid not in cg.nodes:
                    cg.addEnhancedNode(cid, properties={"source": "document_import_nltk"},
                                       linguistics={"wordForms": {"base": concept_word}})
                else:
                    cg.nodes[cid]["doc_freq"] = cg.nodes[cid].get("doc_freq", 0) + 1
            
            for i in range(len(concepts_in_unit)):
                for j in range(i + 1, len(concepts_in_unit)):
                    s_cid_text, t_cid_text = concepts_in_unit[i], concepts_in_unit[j]
                    s_cid, t_cid = normalize_cid(s_cid_text), normalize_cid(t_cid_text)
                    if not cg.edges.get((s_cid, t_cid)):
                        cg.addEdge(s_cid, t_cid, relation="co-occurs_sent_nltk", weight=0.1)
        else: # Last resort: basic co-occurrence on very simple chunks if no NLP
            pass # Covered by the fact that sentences are already split, and no further processing here.

        processed_units += 1
        if processed_units % 20 == 0: 
            print(f"Processed {processed_units}/{len(sentences)} text units ({processing_method})...")

    print(f"Finished populating graph using {processing_method} method. Total nodes: {len(cg.nodes)}, Total edges: {len(cg.edges)}")


if __name__ == '__main__':
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph
    test_cg_main = EnhancedConceptGraph()
    
    sample_text_main = """
    The quick brown fox jumps over the lazy dog. This is a test sentence.
    TorusAI aims to build a cognitive engine. The engine uses a concept graph.
    Natural language processing is key for understanding text. We will parse PDF and DOCX files.
    Learning involves creating nodes and edges from document content. Shakespeare wrote Hamlet.
    Hamlet is a tragedy.
    """
    mock_engine_state_main = {"cg": test_cg_main} 

    # Test with UIE_MODEL_LOADED = True (conceptually)
    # To actually test UIE, you'd set UIE_MODEL_LOADED = True and have a uie_pipeline
    # For now, it will use the mock extract_triples_via_uie or fallback.
    print("\n--- Testing with UIE (conceptual) ---")
    UIE_MODEL_LOADED = True # Temporarily set for this test block
    populate_graph_from_text(sample_text_main, test_cg_main, mock_engine_state_main)
    UIE_MODEL_LOADED = False # Reset
    
    print(f"\nNodes ({len(test_cg_main.nodes)}):")
    # for cid_m, data_m in test_cg_main.nodes.items(): print(f"  {cid_m}: {data_m}")
    print(f"Edges ({len(test_cg_main.edges)}):")
    # for (s_m, t_m), data_m in test_cg_main.edges.items(): print(f"  ({s_m}, {t_m}): {data_m}")

    # Test with spaCy fallback
    print("\n--- Testing with spaCy fallback ---")
    test_cg_spacy = EnhancedConceptGraph()
    NLP_SPACY_backup = NLP_SPACY # store current NLP_SPACY
    UIE_MODEL_LOADED = False # Ensure UIE is off
    # NLP_SPACY = None # Simulate spaCy not being primary for a moment if UIE was on
    populate_graph_from_text(sample_text_main, test_cg_spacy, {"cg": test_cg_spacy})
    # NLP_SPACY = NLP_SPACY_backup # restore

    print(f"\nNodes ({len(test_cg_spacy.nodes)}):")
    print(f"Edges ({len(test_cg_spacy.edges)}):")