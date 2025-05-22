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


# From User Part 1 - EnhancedLanguageGenerator
# Corresponds to L8 Language Module (Morph-Syntax) in the 15-layer specification.
# Generates simple sentences based on activated concepts in the graph.

class EnhancedLanguageGenerator:
    """
    Generates natural language sentences based on the current engine state,
    including active concepts, reflection, retrieved patterns, and the input query.
    Attempts to provide more contextually relevant responses.
    """
    DEFAULT_SUBJECT_CID: str = "default_subject"
    DEFAULT_VERB_CID: str = "default_verb"
    DEFAULT_OBJECT_CID: str = "default_object"

    DEFAULT_SUBJECT_WORD: str = "thing"
    DEFAULT_VERB_WORD: str = "do"
    DEFAULT_OBJECT_WORD: str = "something"
    DEFAULT_UNKNOWN_RESPONSE: str = "I'm not sure how to respond to that."


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
            "statement_svo": "{subject} {verb_present} {object}.", # Basic SVO
            "statement_sa": "{subject} {verb_present} {attribute}.", # Subject-Attribute (e.g. X is Y)
            "statement_s_rel_o": "{subject} {relation} {object}.", # More generic relation
            "simple_concept_focus": "I am thinking about {concept}.",
            "unknown_response": self.DEFAULT_UNKNOWN_RESPONSE
        }

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
            # Fallback for missing node ID based on role
            if role == "subject": return self.DEFAULT_SUBJECT_WORD
            if role == "verb_base" or role == "verb_present": return self.DEFAULT_VERB_WORD
            if role == "object" or role == "attribute": return self.DEFAULT_OBJECT_WORD
            return f"[{cid}_unknown]"

        word_forms = node_data.get("linguistics", {}).get("wordForms", {"base": str(cid)})
        base_form = word_forms.get("base", str(cid))
        
        if role == "subject":
            # TODO: Use linguistic features from node_data if available (e.g., number: singular/plural)
            # For now, assume singular unless is_plural_subject is True (which is not yet dynamically set)
            return base_form
        if role == "object" or role == "attribute":
            return base_form
        if role == "verb_base":
            return base_form
        if role == "verb_present":
            return self._conjugate_present(base_form, is_plural_subject)
        return base_form

    def generate(self, query_string: str, engine_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates a sentence based on the current engine state and the query.
        Prioritizes query-driven graph search, then retrieved patterns, then general context.
        """
        if engine_state is None:
            engine_state = {}

        generated_sentence = self.DEFAULT_UNKNOWN_RESPONSE
        is_plural_subject = False # TODO: Determine from context or node data
        is_question = "?" in query_string
        log_prefix = "[Gen]" # For debugging generation path

        # 1. Parse Query with spaCy (if available) & Identify Key Query Concepts
        key_query_concepts_data: List[Dict[str, str]] = [] # Store as {'text': str, 'type': 'noun_chunk'/'entity'/'verb'}
        if NLP_SPACY_LG and query_string:
            query_doc = NLP_SPACY_LG(query_string.lower())
            for chunk in query_doc.noun_chunks:
                key_query_concepts_data.append({"text": chunk.lemma_.lower().strip(), "type": "noun_chunk"})
            for ent in query_doc.ents:
                key_query_concepts_data.append({"text": ent.lemma_.lower().strip(), "type": ent.label_}) # Store entity type
            for token in query_doc:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    key_query_concepts_data.append({"text": token.lemma_.lower(), "type": "root_verb"})
            
            # Deduplicate based on text
            unique_concepts_text = set()
            temp_list = []
            for item in key_query_concepts_data:
                if item["text"] and len(item["text"]) > 1 and item["text"] not in unique_concepts_text:
                    temp_list.append(item)
                    unique_concepts_text.add(item["text"])
            key_query_concepts_data = temp_list
            log_prefix = "[QuerySpaCy]"


        # 2. Query-Driven Graph Search & Response Formulation
        if key_query_concepts_data:
            primary_query_concept_info = key_query_concepts_data[0]
            primary_concept_text = primary_query_concept_info["text"]

            if primary_concept_text in self.cg.nodes:
                found_relations: List[Tuple[str, str, str, float]] = [] # s_word, relation_word, t_word, weight
                for (s_cid, t_cid), edge_data in self.cg.edges.items():
                    relation = edge_data.get("relation", "is related to")
                    weight = edge_data.get("weight", 0.0)
                    
                    if s_cid == primary_concept_text: # Outgoing relation
                        s_word = self._get_concept_word(s_cid, "subject")
                        t_word = self._get_concept_word(t_cid, "object")
                        found_relations.append((s_word, relation, t_word, weight))
                    elif t_cid == primary_concept_text: # Incoming relation
                        s_word = self._get_concept_word(s_cid, "subject")
                        t_word = self._get_concept_word(t_cid, "object") 
                        found_relations.append((s_word, f"{relation} (of/by/to)", t_word, weight)) 
                
                if found_relations:
                    found_relations.sort(key=lambda x: -x[3]) # Sort by edge weight
                    s_w, r_w, o_w, _ = found_relations[0]
                    
                    # Basic conjugation/phrasing for the relation
                    verb_present = self._conjugate_present(r_w.replace("verb_",""), is_plural_subject) if r_w.startswith("verb_") else r_w

                    if is_question:
                         # Very basic question answering attempt
                        if any(q_word in query_string.lower() for q_word in ["what is", "define", "tell me about"]) and \
                           primary_concept_text.lower() in s_w.lower():
                            generated_sentence = f"{s_w} {verb_present} {o_w}."
                        else:
                            generated_sentence = f"Regarding {primary_concept_text}, one known relation is: {s_w} {verb_present} {o_w}."
                    else:
                        generated_sentence = f"{s_w} {verb_present} {o_w}."
                    
                    self.narrative.append(f"{log_prefix}[GraphDirect] {generated_sentence}")
                    return generated_sentence

        # 3. Try to use highly relevant retrieved patterns (if query-driven failed or no query concepts)
        retrieved_patterns: List[Tuple[str, str, Dict[str, Any], float]] = engine_state.get("retrieved_patterns", [])
        if retrieved_patterns and retrieved_patterns[0][3] > 0.75: # Threshold for using pattern
            pattern_response = retrieved_patterns[0][1]
            # TODO: Adapt response based on current query vs. pattern's trigger
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

            # Ensure default CIDs are used if any part is None
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

            template_key = "question_what" if is_question else "statement"
            chosen_template = self.templates.get(template_key, self.templates["statement_svo"]) # Default to SVO statement
            
            try:
                generated_sentence = chosen_template.format(
                    subject=self._get_concept_word(subj_cid, "subject", is_plural_subject),
                    verb_base=self._get_concept_word(verb_cid, "verb_base"),
                    verb_present=self._get_concept_word(verb_cid, "verb_present", is_plural_subject),
                    object=self._get_concept_word(obj_cid, "object"),
                    attribute=self._get_concept_word(obj_cid, "attribute") # For statement_sa if obj_cid is used as attribute
                ).strip()
                log_prefix = f"{log_prefix}[FallbackSVO]"
            except KeyError: # If template is missing a key used by format
                generated_sentence = f"{self._get_concept_word(subj_cid, 'subject')} is related to {self._get_concept_word(obj_cid, 'object')}."
                log_prefix = f"{log_prefix}[FallbackGeneric]"

        # Final fallback if all else fails
        if not generated_sentence.strip() or generated_sentence == self.DEFAULT_UNKNOWN_RESPONSE :
             if source_concepts and source_concepts[0].get("id"):
                 focused_concept = self._get_concept_word(source_concepts[0]["id"], "subject")
                 generated_sentence = self.templates["simple_concept_focus"].format(concept=focused_concept)
                 log_prefix = f"{log_prefix}[Focus]"
             else: # Absolute fallback
                 generated_sentence = self.DEFAULT_UNKNOWN_RESPONSE
                 log_prefix = f"{log_prefix}[DefaultUnknown]"


        self.narrative.append(f"{log_prefix} {generated_sentence}")
        return generated_sentence
