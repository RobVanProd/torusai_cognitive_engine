from typing import Dict, Any, List, TYPE_CHECKING, Optional, Tuple
import random

if TYPE_CHECKING:
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph

# Attempt to import spacy and load a model, similar to text_to_graph.py
try:
    import spacy
    NLP_SPACY_LG = spacy.load('en_core_web_sm')
except (ImportError, OSError):
    NLP_SPACY_LG = None
    # print("Warning: spaCy or en_core_web_sm model not found for LanguageGenerator. Query understanding will be limited.")

# Placeholder for G2T model and Tokenizer
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # Example
G2T_MODEL = None
G2T_TOKENIZER = None
G2T_MODEL_LOADED = False
# try:
#     # model_name = "tuner007/pegasus_paraphrase" # Example, not a G2T model
#     # model_name = "facebook/bart-large-cnn" # Example summarization
#     # For a real G2T model like 'graph2text-small-bdc-xt', you'd load it here.
#     # G2T_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
#     # G2T_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     # G2T_MODEL_LOADED = True
#     # print("Conceptual G2T model loaded.")
# except Exception as e:
#     print(f"Could not load conceptual G2T model: {e}")
#     G2T_MODEL_LOADED = False

# Placeholder for DATG MLP
DATG_MLP = None # Would be a small torch.nn.Module
DATG_ENABLED = False


# From User Part 1 - EnhancedLanguageGenerator
# Corresponds to L8 Language Module (Morph-Syntax) in the 15-layer specification.
# Generates simple sentences based on activated concepts in the graph.

class EnhancedLanguageGenerator:
    """
    Generates natural language sentences based on the current engine state,
    including active concepts, reflection, retrieved patterns, and the input query.
    Attempts to provide more contextually relevant responses, with placeholders for
    advanced Graph-to-Text (G2T) and DATG integration.
    """
    DEFAULT_SUBJECT_CID: str = "default_subject"
    DEFAULT_VERB_CID: str = "default_verb"
    DEFAULT_OBJECT_CID: str = "default_object"

    DEFAULT_SUBJECT_WORD: str = "thing"
    DEFAULT_VERB_WORD: str = "do"
    DEFAULT_OBJECT_WORD: str = "something"
    DEFAULT_UNKNOWN_RESPONSE: str = "I'm not sure how to respond to that yet."


    def __init__(self, cg_instance: 'EnhancedConceptGraph', narrative_log_list: List[str]):
        """
        Initializes the EnhancedLanguageGenerator.

        Args:
            cg_instance: An instance of EnhancedConceptGraph.
            narrative_log_list: A list to which generated sentences (narratives) will be appended.
                                This list is typically managed by the TorusEngine's state.
        """
        self.cg: 'EnhancedConceptGraph' = cg_instance
        self.narrative: List[str] = narrative_log_list
        self.templates: Dict[str, str] = {
            "question_what": "What does {subject} {verb_base}?", # Basic question
            "statement_svo": "{subject} {verb_present} {object}.",
            "statement_sa": "{subject} {verb_present} {attribute}.",
            "statement_s_rel_o": "{subject} {relation} {object}.",
            "simple_concept_focus": "I am thinking about {concept}.",
            "unknown_response": self.DEFAULT_UNKNOWN_RESPONSE,
            "g2t_failure_fallback": "I found some related concepts: {concept_list}."
        }
        # G2T/DATG Hyperparameters (from design brief)
        self.g2t_emax_edges = 15
        self.g2t_tmax_tokens = 48
        self.datg_lambda_penalty = 0.4
        self.datg_beam_top_k = 2


    def _conjugate_present(self, verb_base: str, is_plural_subject: bool) -> str:
        """Rudimentary present tense conjugation for a given verb base."""
        if not verb_base: return self.DEFAULT_VERB_WORD
        if verb_base in ["be"]: return "are" if is_plural_subject else "is"
        if verb_base in ["have"]: return "have" if is_plural_subject else "has"
        if is_plural_subject: return verb_base
        if verb_base.endswith("y") and len(verb_base) > 1 and verb_base[-2] not in "aeiou":
            return verb_base[:-1] + "ies"
        if verb_base.endswith(("s", "sh", "ch", "x", "z")) or verb_base in ["go", "do"]:
            return verb_base + "es"
        return verb_base + "s"

    def _get_concept_word(self, cid: Optional[str], role: str, is_plural_subject: bool = False) -> str:
        """
        Retrieves a word form for a concept ID based on its role in a sentence.
        Handles basic conjugation for present tense verbs and subject pluralization.
        """
        if not cid: # Handle cases where a CID might be None
            if role == "subject": return self.DEFAULT_SUBJECT_WORD
            if role == "verb_base" or role == "verb_present": return self.DEFAULT_VERB_WORD
            if role == "object" or role == "attribute": return self.DEFAULT_OBJECT_WORD
            return "something_unspecified"

        node_data = self.cg.nodes.get(cid)
        if not node_data:
            if role == "subject": return self.DEFAULT_SUBJECT_WORD
            if role == "verb_base" or role == "verb_present": return self.DEFAULT_VERB_WORD
            if role == "object" or role == "attribute": return self.DEFAULT_OBJECT_WORD
            return f"[{cid}_unknown]"

        word_forms = node_data.get("linguistics", {}).get("wordForms", {"base": str(cid)})
        base_form = word_forms.get("base", str(cid))
        
        if role == "subject":
            return base_form
        if role == "object" or role == "attribute":
            return base_form
        if role == "verb_base":
            return base_form
        if role == "verb_present":
            return self._conjugate_present(base_form, is_plural_subject)
        return base_form

    def _build_subgraph_for_g2t(self, root_concept_cid: Optional[str], max_edges: int) -> List[Dict[str,str]]:
        """
        Builds a subgraph around a root concept for G2T model input.
        Performs a BFS-like traversal up to max_edges.
        Returns a list of linearized triples: {'subject': cid, 'relation': str, 'object': cid}
        """
        if not root_concept_cid or root_concept_cid not in self.cg.nodes:
            return []

        subgraph_triples: List[Dict[str,str]] = []
        q: List[str] = [root_concept_cid]
        visited_nodes = {root_concept_cid}
        edges_count = 0

        head = 0
        while head < len(q) and edges_count < max_edges:
            curr_cid = q[head]
            head += 1

            # Outgoing edges
            for (s, t), edge_data in self.cg.edges.items():
                if s == curr_cid and edges_count < max_edges:
                    subgraph_triples.append({"subject": s, "relation": edge_data.get("relation", "related_to"), "object": t})
                    edges_count += 1
                    if t not in visited_nodes and len(q) < max_edges * 2: # Limit queue size
                        visited_nodes.add(t)
                        q.append(t)
                elif t == curr_cid and edges_count < max_edges: # Incoming edges
                    subgraph_triples.append({"subject": s, "relation": edge_data.get("relation", "related_to_by"), "object": t}) # Indicate directionality
                    edges_count += 1
                    if s not in visited_nodes and len(q) < max_edges * 2:
                        visited_nodes.add(s)
                        q.append(s)
        return subgraph_triples[:max_edges] # Ensure we don't exceed max_edges

    def _linearize_triples_for_g2t(self, triples: List[Dict[str,str]]) -> str:
        """Linearizes triples into the <TRIPLE> subj=X rel=Y obj=Z </TRIPLE> format."""
        # Use actual CIDs (which are already normalized strings)
        return " ".join([f"<TRIPLE> subj={t['subject']} rel={t['relation']} obj={t['object']} </TRIPLE>" for t in triples])

    def _get_datg_control_vector(self, engine_state: Dict[str, Any]) -> Optional[List[float]]:
        """
        Placeholder: Constructs the DATG control vector from engine_state['attr_vec'].
        Example: [tone_id_float, style_id_float, field1_mask, field2_mask, ...]
        """
        attr_vec_config = engine_state.get("attr_vec") # e.g., {"tone": "formal", "style": "summary", "fields_to_include": ["date", "location"]}
        if attr_vec_config:
            # This is highly dependent on how tone/style are encoded and what fields exist
            # Mock implementation:
            # tone_map = {"formal": 0.8, "informal": 0.2, "neutral": 0.5}
            # style_map = {"summary": 0.3, "detailed": 0.7, "bullet":0.5}
            # all_possible_fields = ["date", "location", "person", "event_type"] # Example
            # control_vector = [
            #     tone_map.get(attr_vec_config.get("tone", "neutral"), 0.5),
            #     style_map.get(attr_vec_config.get("style", "summary"), 0.3)
            # ]
            # field_mask = [1.0 if f in attr_vec_config.get("fields_to_include", []) else 0.0 for f in all_possible_fields]
            # control_vector.extend(field_mask)
            # return control_vector
            return [0.5, 0.5, 1.0, 0.0] # Dummy vector
        return None

    def _call_g2t_datg_model_placeholder(self, linearized_subgraph: str, datg_vector: Optional[List[float]]) -> Optional[str]:
        """
        Placeholder for calling the actual G2T model with DATG controls.
        """
        if not G2T_MODEL_LOADED or not G2T_TOKENIZER or not G2T_MODEL:
            # print("LG: G2T model/tokenizer not loaded, cannot generate with G2T.") # Debug
            return None
        
        # print(f"LG: Calling G2T with input: {linearized_subgraph[:100]}...") # Debug
        # print(f"LG: DATG vector: {datg_vector}") # Debug

        # This is where the actual model inference would happen:
        # inputs = G2T_TOKENIZER(linearized_subgraph, return_tensors="pt", max_length=self.g2t_tmax_tokens, truncation=True)
        #
        # if DATG_ENABLED and DATG_MLP and datg_vector is not None:
        #   # output_sequences = G2T_MODEL.datg_generate( # Custom generate method
        #   #    input_ids=inputs['input_ids'],
        #   #    num_beams=4, penalty_alpha=self.datg_lambda_penalty, datg_control_embedding=datg_mlp(datg_vector) ...
        #   # )
        #   pass # Placeholder for DATG generate call
        # else:
        #   # output_sequences = G2T_MODEL.generate(input_ids=inputs['input_ids'], num_beams=4, max_length=60)
        #   pass # Placeholder for standard generate call
        #
        # generated_text = G2T_TOKENIZER.decode(output_sequences[0], skip_special_tokens=True)
        # return generated_text
        
        # Mock response for placeholder:
        if "cat eats fish" in linearized_subgraph:
            return "The cat enjoys eating fish."
        if "torusai aims to build cognitive engine" in linearized_subgraph:
            return "TorusAI is focused on developing a cognitive engine."
        return f"Generated from G2T (conceptual): {linearized_subgraph[:50]}..."


    def generate(self, query_string: str, engine_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates a sentence based on the current engine state and the query.
        Prioritizes G2T+DATG, then query-driven graph search, then retrieved patterns, then general context.
        """
        if engine_state is None:
            engine_state = {}

        generated_sentence = self.DEFAULT_UNKNOWN_RESPONSE
        is_plural_subject = False
        is_question = "?" in query_string
        log_prefix = "[Gen]"

        # print(f"\nLG: Received query: '{query_string}'") # Debug

        # 0. Attempt G2T + DATG if conditions met
        #    Focus concept for subgraph could come from query, reflection, or top active.
        focus_concept_cid: Optional[str] = None
        key_query_concepts_data: List[Dict[str, str]] = []

        if NLP_SPACY_LG and query_string: # Parse query first
            query_doc = NLP_SPACY_LG(query_string.lower())
            for chunk in query_doc.noun_chunks: key_query_concepts_data.append({"text": chunk.lemma_.lower().strip(), "type": "noun_chunk"})
            for ent in query_doc.ents: key_query_concepts_data.append({"text": ent.lemma_.lower().strip(), "type": ent.label_})
            # ... (rest of spaCy query parsing from previous version) ...
            unique_concepts_text = set()
            temp_list = []
            for item in key_query_concepts_data:
                if item["text"] and len(item["text"]) > 1 and item["text"] not in unique_concepts_text:
                    temp_list.append(item)
                    unique_concepts_text.add(item["text"])
            key_query_concepts_data = temp_list
            if key_query_concepts_data: focus_concept_cid = key_query_concepts_data[0]['text']

        if not focus_concept_cid: # Fallback to reflection or active
            reflection: List[Dict[str, Any]] = engine_state.get("reflection", [])
            if reflection: focus_concept_cid = reflection[0].get("id")
            else:
                active = self.cg.getActiveConceptPath(threshold=0.6)
                if active: focus_concept_cid = active[0].get("id")
        
        if G2T_MODEL_LOADED and focus_concept_cid:
            # print(f"LG: Attempting G2T with focus: {focus_concept_cid}") # Debug
            subgraph_triples = self._build_subgraph_for_g2t(focus_concept_cid, self.g2t_emax_edges)
            if subgraph_triples and len(subgraph_triples) > 0: # Min 1 edge for G2T
                linearized_graph = self._linearize_triples_for_g2t(subgraph_triples)
                if len(linearized_graph.split()) <= self.g2t_tmax_tokens: # Check token limit
                    datg_vec = self._get_datg_control_vector(engine_state)
                    g2t_response = self._call_g2t_datg_model_placeholder(linearized_graph, datg_vec)
                    if g2t_response:
                        self.narrative.append(f"[G2T_DATG] {g2t_response}")
                        return g2t_response
                # else:
                    # print(f"LG: G2T linearized graph too long or too short. Subgraph edges: {len(subgraph_triples)}") # Debug

        # 1. Parse Query with spaCy (if not already done for G2T focus) & Identify Key Query Concepts
        # (key_query_concepts_data and query_root_verb might already be populated if G2T path wasn't taken)
        if not key_query_concepts_data and NLP_SPACY_LG and query_string: # Reparse if G2T didn't use query
            log_prefix = "[QuerySpaCy]"
            query_doc = NLP_SPACY_LG(query_string.lower())
            for chunk in query_doc.noun_chunks: key_query_concepts_data.append({"text": chunk.lemma_.lower().strip(), "type": "noun_chunk"})
            for ent in query_doc.ents: key_query_concepts_data.append({"text": ent.lemma_.lower().strip(), "type": ent.label_})
            query_root_verb = None
            for token in query_doc:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    query_root_verb = token.lemma_.lower()
                    if query_root_verb not in self.cg.nodes and query_root_verb not in ["be", "do", "have"]:
                         self.cg.addEnhancedNode(query_root_verb, linguistics={"wordForms":{"base":query_root_verb}}, properties={"pos":"VERB", "source":"query_verb"})
                    key_query_concepts_data.append({"text": query_root_verb, "type": "root_verb"})
                    break
            for token in query_doc:
                if token.is_alpha and not token.is_stop and token.lemma_.lower() not in [item['text'] for item in key_query_concepts_data]:
                    key_query_concepts_data.append({"text": token.lemma_.lower(), "type": "token"})
            unique_concepts_text = set()
            temp_list = []
            for item in key_query_concepts_data:
                if item["text"] and len(item["text"]) > 1 and item["text"] not in unique_concepts_text:
                    temp_list.append(item)
                    unique_concepts_text.add(item["text"])
            key_query_concepts_data = temp_list
            # print(f"LG: Key query concepts data (re-parsed): {key_query_concepts_data}") # Debug


        # 2. Query-Driven Graph Search & Response Formulation (if G2T failed)
        if key_query_concepts_data:
            primary_query_concept_info = key_query_concepts_data[0]
            primary_concept_text = primary_query_concept_info["text"]
            if primary_concept_text in self.cg.nodes:
                found_relations: List[Tuple[str, str, str, float]] = []
                for (s_cid, t_cid), edge_data in self.cg.edges.items():
                    relation = edge_data.get("relation", "is related to")
                    weight = edge_data.get("weight", 0.0)
                    s_word = self._get_concept_word(s_cid, "subject")
                    t_word = self._get_concept_word(t_cid, "object")
                    if s_cid == primary_concept_text and t_cid not in [item['text'] for item in key_query_concepts_data if item['text'] != primary_concept_text]:
                        found_relations.append((s_word, relation, t_word, weight))
                    elif t_cid == primary_concept_text and s_cid not in [item['text'] for item in key_query_concepts_data if item['text'] != primary_concept_text]:
                        found_relations.append((s_word, f"{relation} (to/by/of)", t_word, weight))
                
                if found_relations:
                    found_relations.sort(key=lambda x: -x[3])
                    s_w, r_w, o_w, _ = found_relations[0]
                    verb_present = self._conjugate_present(r_w.replace("verb_",""), is_plural_subject) if r_w.startswith("verb_") else r_w
                    if is_question:
                        # (Simplified question logic from before)
                        if query_root_verb and query_root_verb in ["be", "is", "are", "what", "who", "where", "when", "why", "how"]:
                             generated_sentence = f"{s_w} {verb_present} {o_w}."
                        else:
                             generated_sentence = f"Is it that {s_w} {verb_present} {o_w}?"
                    else:
                        generated_sentence = f"{s_w} {verb_present} {o_w}."
                    self.narrative.append(f"{log_prefix}[GraphDirect] {generated_sentence}")
                    return generated_sentence

        # 3. Try to use highly relevant retrieved patterns
        retrieved_patterns: List[Tuple[str, str, Dict[str, Any], float]] = engine_state.get("retrieved_patterns", [])
        if retrieved_patterns and retrieved_patterns[0][3] > 0.75:
            pattern_response = retrieved_patterns[0][1]
            generated_sentence = pattern_response
            self.narrative.append(f"{log_prefix}[PatternBased] {generated_sentence}")
            return generated_sentence

        # 4. Fallback to Reflection / Active Concepts with basic SVO template
        source_concepts: List[Dict[str, Any]] = engine_state.get("reflection", [])
        if not source_concepts:
            source_concepts = self.cg.getActiveConceptPath(threshold=0.5)

        if source_concepts:
            subj_cid = source_concepts[0].get("id") if len(source_concepts) > 0 else None
            verb_cid = source_concepts[1].get("id") if len(source_concepts) > 1 else None
            obj_cid = source_concepts[2].get("id") if len(source_concepts) > 2 else None
            subj_cid = subj_cid or self.DEFAULT_SUBJECT_CID
            verb_cid = verb_cid or self.DEFAULT_VERB_CID
            obj_cid = obj_cid or self.DEFAULT_OBJECT_CID
            
            default_nodes_to_add = {
                self.DEFAULT_SUBJECT_CID: self.DEFAULT_SUBJECT_WORD,
                self.DEFAULT_VERB_CID: self.DEFAULT_VERB_WORD,
                self.DEFAULT_OBJECT_CID: self.DEFAULT_OBJECT_WORD
            }
            for cid_default, word_default in default_nodes_to_add.items():
                if cid_default in [subj_cid, verb_cid, obj_cid] and cid_default not in self.cg.nodes:
                    self.cg.addEnhancedNode(cid_default, linguistics={"wordForms": {"base": word_default}})

            template_key = "question_what" if is_question else "statement_svo"
            chosen_template = self.templates.get(template_key, self.templates["statement_svo"])
            
            try:
                generated_sentence = chosen_template.format(
                    subject=self._get_concept_word(subj_cid, "subject", is_plural_subject),
                    verb_base=self._get_concept_word(verb_cid, "verb_base"),
                    verb_present=self._get_concept_word(verb_cid, "verb_present", is_plural_subject),
                    object=self._get_concept_word(obj_cid, "object"),
                    attribute=self._get_concept_word(obj_cid, "attribute")
                ).strip()
                log_prefix = f"{log_prefix}[FallbackSVO]"
            except KeyError:
                generated_sentence = f"{self._get_concept_word(subj_cid, 'subject')} is related to {self._get_concept_word(obj_cid, 'object')}."
                log_prefix = f"{log_prefix}[FallbackGeneric]"
        
        if not generated_sentence.strip() or generated_sentence == self.DEFAULT_UNKNOWN_RESPONSE :
             if source_concepts and source_concepts[0].get("id"):
                 focused_concept = self._get_concept_word(source_concepts[0]["id"], "subject")
                 generated_sentence = self.templates["simple_concept_focus"].format(concept=focused_concept)
                 log_prefix = f"{log_prefix}[Focus]"
             else:
                 generated_sentence = self.DEFAULT_UNKNOWN_RESPONSE
                 log_prefix = f"{log_prefix}[DefaultUnknown]"

        self.narrative.append(f"{log_prefix} {generated_sentence}")
        return generated_sentence
