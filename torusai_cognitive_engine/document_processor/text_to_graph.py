from typing import Dict, Any, List, TYPE_CHECKING, Tuple, Optional # Added Optional
import re
import torch # Added torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # Added transformers imports

if TYPE_CHECKING:
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph

# UIE Model Globals
UIE_MODEL = None
UIE_TOKENIZER = None
UIE_MODEL_NAME = "t5-small" # Default model, can be configured
UIE_MODEL_LOADED = False
TORCH_DEVICE = None

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

def _load_uie_model_if_needed():
    """Loads the UIE model and tokenizer if they haven't been loaded yet."""
    global UIE_MODEL, UIE_TOKENIZER, UIE_MODEL_LOADED, TORCH_DEVICE, UIE_MODEL_NAME

    if UIE_MODEL_LOADED:
        return

    try:
        if TORCH_DEVICE is None:
            TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading UIE model '{UIE_MODEL_NAME}' on device '{TORCH_DEVICE}'...")
        UIE_TOKENIZER = AutoTokenizer.from_pretrained(UIE_MODEL_NAME)
        UIE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(UIE_MODEL_NAME)
        UIE_MODEL.to(TORCH_DEVICE) # type: ignore
        UIE_MODEL.eval() # type: ignore
        UIE_MODEL_LOADED = True
        print(f"UIE model '{UIE_MODEL_NAME}' loaded successfully.")
    except Exception as e:
        print(f"Error loading UIE model '{UIE_MODEL_NAME}': {e}")
        UIE_MODEL = None
        UIE_TOKENIZER = None
        UIE_MODEL_LOADED = False

def _parse_entity_part(part_text: str) -> Tuple[str, Optional[str]]:
    """
    Parses an entity part string like "entity text [entity_type]"
    Returns (entity_text, entity_type) or (entity_text, None).
    """
    match = re.match(r"(.+?)\s*\[(.+)\]$", part_text.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return part_text.strip(), None

def normalize_cid(text: str) -> str:
    """Normalizes a text string to be used as a Concept ID."""
    if not isinstance(text, str): # Ensure text is a string
        text = str(text)
    return text.lower().replace(" ", "_").strip()

def extract_triples_via_uie(text_chunk: str) -> List[Dict[str, Any]]:
    """
    Extracts (subject, predicate, object) triples using the loaded UIE model.
    Returns a list of dictionaries, each like:
    {'subject': str, 'subject_type': Optional[str], 
     'predicate': str, 
     'object': str, 'object_type': Optional[str], 
     'confidence': float, 'provenance_span': Optional[Tuple[int,int]]}
    """
    _load_uie_model_if_needed()
    if not UIE_MODEL_LOADED or UIE_MODEL is None or UIE_TOKENIZER is None:
        # print("UIE model not loaded, skipping triple extraction.") # Already printed in _load_uie_model_if_needed
        return []

    triples: List[Dict[str, Any]] = []
    
    # Construct prompt based on common UIE practices for triple extraction
    prompt = f"Extract structured information from the following text. Output each triple as: <subject> [subject_type] | <predicate> | <object> [object_type] ||. Text: {text_chunk}"

    try:
        inputs = UIE_TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=512).to(TORCH_DEVICE) # type: ignore
        generated_ids = UIE_MODEL.generate(inputs["input_ids"], max_length=256, num_beams=4, early_stopping=True) # type: ignore
        generated_text = UIE_TOKENIZER.decode(generated_ids[0], skip_special_tokens=True)

        # print(f"UIE Input: {prompt}") # For debugging
        # print(f"UIE Output: {generated_text}") # For debugging

        extracted_parts = generated_text.split("||")
        for part in extracted_parts:
            part = part.strip()
            if not part:
                continue
            
            elements = part.split("|")
            if len(elements) == 3:
                subj_text, subj_type = _parse_entity_part(elements[0])
                pred_text = elements[1].strip()
                obj_text, obj_type = _parse_entity_part(elements[2])

                if subj_text and pred_text and obj_text:
                    triples.append({
                        "subject": subj_text, "subject_type": subj_type,
                        "predicate": pred_text,
                        "object": obj_text, "object_type": obj_type,
                        "confidence": 0.85,  # Placeholder confidence
                        "provenance_span": None # Placeholder for actual span if available
                    })
            # else:
                # print(f"Could not parse UIE triple part: '{part}'") # For debugging

    except Exception as e:
        print(f"Error during UIE model inference or parsing: {e}")
        return [] # Return empty list on error

    # Mock implementation for specific cases if needed for quick testing, remove for production
    # if "TorusAI aims to build a cognitive engine" in text_chunk:
    #     return [
    #         {"subject": "TorusAI", "subject_type": "ORG", "predicate": "aims to build", "object": "cognitive engine", "object_type": "PRODUCT", "confidence": 0.9, "provenance_span": None},
    #         {"subject": "cognitive engine", "subject_type": "PRODUCT", "predicate": "is_a_type_of", "object": "AI system", "object_type": "TECHNOLOGY", "confidence": 0.85, "provenance_span": None}
    #     ]
    # if "Shakespeare wrote Hamlet" in text_chunk:
    #     return [
    #         {"subject": "Shakespeare", "subject_type": "PERSON", "predicate": "wrote", "object": "Hamlet", "object_type": "WORK_OF_ART", "confidence": 0.95, "provenance_span": None}
    #     ]
    return triples


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
    _load_uie_model_if_needed() # Ensure UIE model is loaded (or attempted) before processing
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
                    pred_text = triple.get("predicate","").strip() # This is the relation itself
                    obj_text = triple.get("object","").strip()
                    
                    subj_type = triple.get("subject_type")
                    obj_type = triple.get("object_type")
                    
                    conf = triple.get("confidence", 0.5)
                    # span = triple.get("provenance_span") # TODO: Store span if available

                    if not (subj_text and pred_text and obj_text):
                        # print(f"Skipping incomplete triple: S:{subj_text}, P:{pred_text}, O:{obj_text}") # For debugging
                        continue

                    s_cid = normalize_cid(subj_text)
                    # Predicate becomes the relation string, can be normalized or used as is
                    p_relation_str = normalize_cid(pred_text) 
                    o_cid = normalize_cid(obj_text)

                    s_props = {"source": "uie_import"}
                    if subj_type: s_props["entity_type"] = subj_type
                    if s_cid not in cg.nodes:
                        cg.addEnhancedNode(s_cid, properties=s_props, linguistics={"wordForms": {"base": subj_text}})
                    else:
                        cg.nodes[s_cid]["doc_freq"] = cg.nodes[s_cid].get("doc_freq", 0) + 1
                        if subj_type and "entity_type" not in cg.nodes[s_cid]:
                             cg.nodes[s_cid]["entity_type"] = subj_type
                    
                    o_props = {"source": "uie_import"}
                    if obj_type: o_props["entity_type"] = obj_type
                    if o_cid not in cg.nodes:
                        cg.addEnhancedNode(o_cid, properties=o_props, linguistics={"wordForms": {"base": obj_text}})
                    else:
                        cg.nodes[o_cid]["doc_freq"] = cg.nodes[o_cid].get("doc_freq", 0) + 1
                        if obj_type and "entity_type" not in cg.nodes[o_cid]:
                            cg.nodes[o_cid]["entity_type"] = obj_type
                    
                    # Add edge for the triple
                    current_edge_data = cg.edges.get((s_cid, o_cid))
                    # Allow multiple relationships if predicates are different, or update if same predicate
                    # For simplicity, if an edge exists, we check its relation. If different, we could add another
                    # or decide on a merging strategy. Here, we'll overwrite if relation is different, or update weight.
                    # A more robust approach might involve a list of relations on an edge or typed edges.
                    if not current_edge_data or current_edge_data.get("relation") != p_relation_str:
                        cg.addEdge(s_cid, o_cid, relation=p_relation_str, weight=float(conf), 
                                   properties={"source": "uie_triple"})
                    else: # Edge with same relation exists, update weight
                        existing_weight = current_edge_data.get("weight", 0.0)
                        cg.edges[(s_cid, o_cid)]["weight"] = max(existing_weight, float(conf))
                        # Potentially add confidence scores or provenance details to edge properties
                        if "sources" not in cg.edges[(s_cid, o_cid)]: cg.edges[(s_cid, o_cid)]["sources"] = []
                        cg.edges[(s_cid, o_cid)]["sources"].append("uie_triple_update")


        elif NLP_SPACY: # Secondary choice: spaCy (if UIE didn't yield results or wasn't primary)
            # print(f"No UIE triples for chunk, or UIE not primary. Using spaCy for: {unit_text[:50]}...") # Debugging
            extract_concepts_relationships_spacy(unit_text, cg)
        elif nltk: # Tertiary choice: NLTK basic
            # print(f"No UIE/spaCy. Using NLTK for: {unit_text[:50]}...") # Debugging
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

    # Test with UIE (actual model loading will be attempted)
    print("\n--- Testing with UIE (attempting actual model load) ---")
    # UIE_MODEL_LOADED is now controlled by _load_uie_model_if_needed()
    # To force a real test, ensure transformers, torch, sentencepiece are installed
    # and internet is available for model download on first run.
    # UIE_MODEL_NAME = "t5-small" # Or "google/flan-t5-small" etc.
    populate_graph_from_text(sample_text_main, test_cg_main, mock_engine_state_main)
    
    print(f"\nNodes after UIE ({len(test_cg_main.nodes)}):")
    # for cid_m, data_m in test_cg_main.nodes.items(): print(f"  {cid_m}: {data_m}")
    print(f"Edges after UIE ({len(test_cg_main.edges)}):")
    # for (s_m, t_m), data_m in test_cg_main.edges.items(): print(f"  ({s_m}, {t_m}): {data_m}")

    # Test with spaCy (if UIE was not available or as a comparison)
    # Reset graph for a clean spaCy test run if UIE might have populated it
    test_cg_spacy = EnhancedConceptGraph() 
    print("\n--- Testing with spaCy (simulating UIE not loaded) ---")
    # Temporarily disable UIE to force spaCy path
    original_uie_loaded_state = UIE_MODEL_LOADED
    UIE_MODEL_LOADED = False 
    
    populate_graph_from_text(sample_text_main, test_cg_spacy, {"cg": test_cg_spacy})
    
    UIE_MODEL_LOADED = original_uie_loaded_state # Restore UIE state

    print(f"\nNodes after spaCy ({len(test_cg_spacy.nodes)}):")
    print(f"Edges after spaCy ({len(test_cg_spacy.edges)}):")