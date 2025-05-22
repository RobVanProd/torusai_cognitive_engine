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
G2T_MODEL_LOADED = False # Set to True when a real G2T model is chosen and loaded
# try:
#     # model_name = "your-chosen-g2t-model-on-huggingface" 
#     # G2T_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
#     # G2T_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     # G2T_MODEL_LOADED = True
#     # print("Conceptual G2T model loaded.")
# except Exception as e:
#     print(f"Could not load conceptual G2T model: {e}")
#     G2T_MODEL_LOADED = False

# Placeholder for DATG MLP
DATG_MLP = None # Would be a small torch.nn.Module, e.g., torch.nn.Linear(hidden_size, datg_vector_size)
DATG_ENABLED = False # Set to True when DATG logic and MLP are implemented


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

    DEFAULT_SUBJECT_WORD: str = "something" # Changed from "thing" for variety
    DEFAULT_VERB_WORD: str = "is" # Changed from "do"
    DEFAULT_OBJECT_WORD: str = "related" # Changed from "something"
    DEFAULT_UNKNOWN_RESPONSE: str = "I'm not sure how to respond to that at the moment."


    def __init__(self, cg_instance: 'EnhancedConceptGraph', narrative_log_list: List[str]):
        self.cg: 'EnhancedConceptGraph' = cg_instance
        self.narrative: List[str] = narrative_log_list
        self.templates: Dict[str, str] = {
            "question_what": "What is known about {subject} regarding {verb_base}?",
            "statement_svo": "{subject} {verb_present} {object}.",
            "statement_sa": "{subject} {verb_present} {attribute}.",
            "statement_s_rel_o": "{subject} {relation} {object}.",
            "simple_concept_focus": "My current focus is on {concept}.",
            "unknown_response": self.DEFAULT_UNKNOWN_RESPONSE,
            "g2t_failure_fallback": "I found some related concepts: {concept_list}.",
            "kgqa_result": "Based on my knowledge graph: {result_sentence}.",
            "kgqa_no_result": "I couldn't find a direct answer in my knowledge graph for that specific KG query."
        }
        # G2T/DATG Hyperparameters (from design brief)
        self.g2t_emax_edges: int = 15
        self.g2t_tmax_tokens: int = 48 # Max tokens for G2T encoder input
        self.datg_lambda_penalty: float = 0.4
        self.datg_beam_top_k: int = 2
        # KGQA related
        self.kgqa_active: bool = True # Control flag for KGQA path

    def _conjugate_present(self, verb_base: str, is_plural_subject: bool) -> str:
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
        if not cid:
            if role == "subject": return self.DEFAULT_SUBJECT_WORD
            if role == "verb_base" or role == "verb_present": return self.DEFAULT_VERB_WORD
            if role == "object" or role == "attribute": return self.DEFAULT_OBJECT_WORD
            return "unspecified_item"

        node_data = self.cg.nodes.get(cid)
        if not node_data:
            return f"concept_{cid}" # Return CID if node not found but CID given

        word_forms = node_data.get("linguistics", {}).get("wordForms", {"base": str(cid)})
        base_form = word_forms.get("base", str(cid))
        
        if role == "subject" or role == "object" or role == "attribute" or role == "concept":
            return base_form 
        if role == "verb_base":
            return base_form
        if role == "verb_present":
            return self._conjugate_present(base_form, is_plural_subject)
        return base_form

    def _build_subgraph_for_g2t(self, root_concept_cid: Optional[str], max_edges: int) -> List[Dict[str,str]]:
        if not root_concept_cid or root_concept_cid not in self.cg.nodes:
            return []
        subgraph_triples: List[Dict[str,str]] = []
        q: List[str] = [root_concept_cid]
        visited_nodes = {root_concept_cid}
        edges_count = 0; head = 0
        while head < len(q) and edges_count < max_edges:
            curr_cid = q[head]; head += 1
            for (s, t), edge_data in self.cg.edges.items():
                if s == curr_cid and edges_count < max_edges:
                    subgraph_triples.append({"subject": s, "relation": edge_data.get("relation", "related_to"), "object": t})
                    edges_count += 1
                    if t not in visited_nodes and len(q) < max_edges * 3: # Wider BFS queue
                        visited_nodes.add(t); q.append(t)
                elif t == curr_cid and edges_count < max_edges:
                    subgraph_triples.append({"subject": s, "relation": edge_data.get("relation", "related_to_by"), "object": t})
                    edges_count += 1
                    if s not in visited_nodes and len(q) < max_edges * 3:
                        visited_nodes.add(s); q.append(s)
        return subgraph_triples[:max_edges]

    def _linearize_triples_for_g2t(self, triples: List[Dict[str,str]]) -> str:
        return " ".join([f"<TRIPLE> subj={self._get_concept_word(t['subject'],'concept')} rel={t['relation']} obj={self._get_concept_word(t['object'],'concept')} </TRIPLE>" for t in triples])

    def _get_datg_control_vector(self, engine_state: Dict[str, Any]) -> Optional[List[float]]:
        attr_vec_config = engine_state.get("attr_vec") 
        if attr_vec_config:
            # Placeholder: actual encoding of tone, style, fields needed
            return [0.5, 0.5, 1.0, 0.0, 0.0] # Example: [tone, style, field1_mask, field2_mask, field3_mask]
        return None

    def _call_g2t_datg_model_placeholder(self, linearized_subgraph: str, datg_vector: Optional[List[float]]) -> Optional[str]:
        if not G2T_MODEL_LOADED or not G2T_TOKENIZER or not G2T_MODEL: return None
        # print(f"LG: Conceptual G2T call with input: {linearized_subgraph[:100]}..., DATG: {datg_vector}")
        # Actual model call would be here. For now, mock response.
        if "cat related_to fish" in linearized_subgraph or "cat eats fish" in linearized_subgraph : # Adjusted for linearized output
            return "The cat seems interested in fish."
        if "torusai related_to cognitive_engine" in linearized_subgraph:
            return "TorusAI is developing a cognitive engine."
        return f"Concepts processed by G2T: {linearized_subgraph[:70]}..."

    def _parse_kg_query_placeholder(self, query_text: str) -> Optional[Any]:
        # Placeholder for text-to-SPARQL or other structured query format
        # print(f"LG: Conceptual KGQA Parse: '{query_text}'")
        if "who created hamlet" in query_text.lower(): return {"type": "SPARQL_placeholder", "query": "SELECT ?creator WHERE { <hamlet> <created_by> ?creator . }"}
        if "what is hamlet" in query_text.lower(): return {"type": "SPARQL_placeholder", "query": "SELECT ?description WHERE { <hamlet> <is_a> ?description . }"}
        return None

    def _execute_graph_query_placeholder(self, structured_query: Any) -> List[Dict[str, Any]]:
        # Placeholder for executing the structured query against self.cg
        # print(f"LG: Conceptual KGQA Execute: {structured_query}")
        results = []
        if isinstance(structured_query, dict) and structured_query.get("type") == "SPARQL_placeholder":
            # This would involve a real SPARQL engine or graph traversal based on the parsed query
            if "hamlet" in structured_query.get("query","") and "created_by" in structured_query.get("query",""):
                # Simulate finding Shakespeare
                for (s,t), data in self.cg.edges.items():
                    if t == "hamlet" and data.get("relation") == "wrote": # Assuming 'wrote' implies 'created_by'
                        results.append({"creator": self._get_concept_word(s, "concept")})
            elif "hamlet" in structured_query.get("query","") and "is_a" in structured_query.get("query",""):
                 for (s,t), data in self.cg.edges.items():
                    if s == "hamlet" and data.get("relation") == "is_a_type_of": # or just 'is_a'
                        results.append({"description": self._get_concept_word(t, "concept")})
        return results

    def _format_kgqa_results(self, results: List[Dict[str, Any]], original_query: str) -> str:
        if not results: return self.templates["kgqa_no_result"]
        # Simple formatting, assumes single binding
        first_result = results[0]
        answer_parts = [f"{k}: {v}" for k,v in first_result.items()]
        return self.templates["kgqa_result"].format(result_sentence=", ".join(answer_parts))


    def generate(self, query_string: str, engine_state: Optional[Dict[str, Any]] = None) -> str:
        if engine_state is None: engine_state = {}
        generated_sentence = self.DEFAULT_UNKNOWN_RESPONSE
        is_plural_subject = False 
        is_question = "?" in query_string
        log_prefix = "[Gen]"

        # 0. KGQA Path (if query is prefixed)
        if self.kgqa_active and query_string.upper().startswith("KG:"):
            actual_kg_query = query_string[3:].strip()
            log_prefix = "[KGQA]"
            structured_query = self._parse_kg_query_placeholder(actual_kg_query)
            if structured_query:
                query_results = self._execute_graph_query_placeholder(structured_query)
                generated_sentence = self._format_kgqa_results(query_results, actual_kg_query)
                self.narrative.append(f"{log_prefix} {generated_sentence}")
                return generated_sentence
            else:
                self.narrative.append(f"{log_prefix} Could not parse KG query: {actual_kg_query}")
                # Fall through to other methods if KG parsing fails

        # 1. Attempt G2T + DATG 
        focus_concept_cid: Optional[str] = None
        key_query_concepts_data: List[Dict[str, str]] = [] 
        if NLP_SPACY_LG and query_string: 
            query_doc = NLP_SPACY_LG(query_string.lower())
            for chunk in query_doc.noun_chunks: key_query_concepts_data.append({"text": chunk.lemma_.lower().strip(), "type": "noun_chunk"})
            # ... (more spaCy parsing as before to populate key_query_concepts_data) ...
            if key_query_concepts_data: focus_concept_cid = key_query_concepts_data[0]['text']

        if not focus_concept_cid: 
            reflection: List[Dict[str, Any]] = engine_state.get("reflection", [])
            if reflection: focus_concept_cid = reflection[0].get("id")
            else:
                active = self.cg.getActiveConceptPath(threshold=0.55) # Higher threshold for G2T focus
                if active: focus_concept_cid = active[0].get("id")
        
        if G2T_MODEL_LOADED and focus_concept_cid:
            log_prefix = "[G2T]"
            subgraph_triples = self._build_subgraph_for_g2t(focus_concept_cid, self.g2t_emax_edges)
            if subgraph_triples:
                linearized_graph = self._linearize_triples_for_g2t(subgraph_triples)
                if len(linearized_graph.split()) <= self.g2t_tmax_tokens:
                    datg_vec = self._get_datg_control_vector(engine_state)
                    g2t_response = self._call_g2t_datg_model_placeholder(linearized_graph, datg_vec)
                    if g2t_response:
                        self.narrative.append(f"{log_prefix} {g2t_response}")
                        return g2t_response
                    else: # G2T model returned None/empty
                        generated_sentence = self.templates["g2t_failure_fallback"].format(concept_list=", ".join(list(set(t['subject'] for t in subgraph_triples))[:3]))
                        self.narrative.append(f"{log_prefix}[Fail] {generated_sentence}")
                        return generated_sentence


        # 2. Query-Driven Graph Search (if G2T failed or not applicable)
        # (Repopulate key_query_concepts_data if G2T path wasn't fully taken or didn't parse query)
        if not key_query_concepts_data and NLP_SPACY_LG and query_string:
            query_doc = NLP_SPACY_LG(query_string.lower()) 
            for chunk in query_doc.noun_chunks: key_query_concepts_data.append({"text": chunk.lemma_.lower().strip(), "type": "noun_chunk"})
            for ent in query_doc.ents: key_query_concepts_data.append({"text": ent.lemma_.lower().strip(), "type": ent.label_})
            query_root_verb = None
            for token in query_doc:
                if token.dep_ == "ROOT" and token.pos_ == "VERB": query_root_verb = token.lemma_.lower(); key_query_concepts_data.append({"text":query_root_verb, "type":"root_verb"}); break
            unique_concepts_text = set(); temp_list = []
            for item in key_query_concepts_data:
                if item["text"] and len(item["text"]) > 1 and item["text"] not in unique_concepts_text: temp_list.append(item); unique_concepts_text.add(item["text"])
            key_query_concepts_data = temp_list
        
        if key_query_concepts_data:
            log_prefix = "[QueryGraph]"
            primary_concept_text = key_query_concepts_data[0]["text"]
            if primary_concept_text in self.cg.nodes:
                found_relations: List[Tuple[str, str, str, float]] = []
                # ... (Simplified graph search logic from previous version) ...
                for (s_cid, t_cid), edge_data in self.cg.edges.items():
                    if s_cid == primary_concept_text: found_relations.append((self._get_concept_word(s_cid,"subject"), edge_data.get("relation","related_to"), self._get_concept_word(t_cid,"object"), edge_data.get("weight",0.0)))
                if found_relations:
                    found_relations.sort(key=lambda x: -x[3])
                    s_w, r_w, o_w, _ = found_relations[0]
                    verb_present = self._conjugate_present(r_w.replace("verb_",""), is_plural_subject) if r_w.startswith("verb_") else r_w
                    generated_sentence = f"{s_w} {verb_present} {o_w}."
                    self.narrative.append(f"{log_prefix} {generated_sentence}")
                    return generated_sentence
        
        # 3. Pattern-Based Fallback
        retrieved_patterns: List[Tuple[str, str, Dict[str, Any], float]] = engine_state.get("retrieved_patterns", [])
        if retrieved_patterns and retrieved_patterns[0][3] > 0.70: # Slightly lower threshold for fallback
            generated_sentence = retrieved_patterns[0][1]
            self.narrative.append(f"[PatternFallback] {generated_sentence}")
            return generated_sentence

        # 4. Reflection/Active Concepts SVO Fallback
        source_concepts: List[Dict[str, Any]] = engine_state.get("reflection", [])
        if not source_concepts: source_concepts = self.cg.getActiveConceptPath(threshold=0.5)
        if source_concepts:
            log_prefix = "[FallbackSVO]"
            # ... (SVO template filling logic from previous version) ...
            subj_cid = source_concepts[0].get("id") if len(source_concepts) > 0 else self.DEFAULT_SUBJECT_CID
            # ... (rest of SVO logic) ...
            template_key = "question_what" if is_question else "statement_svo"
            chosen_template = self.templates.get(template_key, self.templates["statement_svo"])
            try:
                generated_sentence = chosen_template.format(
                    subject=self._get_concept_word(subj_cid, "subject",is_plural_subject),
                    verb_base=self._get_concept_word(source_concepts[1].get("id", self.DEFAULT_VERB_CID) if len(source_concepts)>1 else self.DEFAULT_VERB_CID, "verb_base"),
                    verb_present=self._get_concept_word(source_concepts[1].get("id", self.DEFAULT_VERB_CID) if len(source_concepts)>1 else self.DEFAULT_VERB_CID, "verb_present",is_plural_subject),
                    object=self._get_concept_word(source_concepts[2].get("id", self.DEFAULT_OBJECT_CID) if len(source_concepts)>2 else self.DEFAULT_OBJECT_CID, "object")
                ).strip()
            except Exception: # Catch broad exception if formatting fails
                 generated_sentence = self.templates["simple_concept_focus"].format(concept=self._get_concept_word(subj_cid, "concept"))


        if not generated_sentence.strip() or generated_sentence == self.DEFAULT_UNKNOWN_RESPONSE:
             if source_concepts and source_concepts[0].get("id"):
                 focused_concept = self._get_concept_word(source_concepts[0]["id"], "subject")
                 generated_sentence = self.templates["simple_concept_focus"].format(concept=focused_concept)
                 log_prefix = "[Focus]"
        
        self.narrative.append(f"{log_prefix} {generated_sentence}")
        return generated_sentence
