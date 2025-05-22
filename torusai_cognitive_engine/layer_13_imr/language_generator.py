from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..layer_04_dcg.concept_graph import EnhancedConceptGraph

# From User Part 1 - EnhancedLanguageGenerator
# Corresponds to L8 Language Module (Morph-Syntax) in the 15-layer specification.
# Generates simple sentences based on activated concepts in the graph.

class EnhancedLanguageGenerator:
    """
    Generates basic natural language sentences based on the current state of
    active concepts in the concept graph. It uses simple templates and
    rudimentary grammar rules.
    """
    DEFAULT_SUBJECT_CID: str = "default_subject"
    DEFAULT_VERB_CID: str = "default_verb"
    DEFAULT_OBJECT_CID: str = "default_object"

    DEFAULT_SUBJECT_WORD: str = "thing"
    DEFAULT_VERB_WORD: str = "do"
    DEFAULT_OBJECT_WORD: str = "something"

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
            "question_what": "What does {subject} {verb_base}?",
            "statement": "{subject} {verb_present} {object}."
            # Future: Add more templates for variety, e.g., based on query type or reflection.
        }

    def _conjugate_present(self, verb_base: str, is_plural_subject: bool) -> str:
        """Rudimentary present tense conjugation for a given verb base."""
        if verb_base in ["be"]: return "are" if is_plural_subject else "is"
        if verb_base in ["have"]: return "have" if is_plural_subject else "has"
        if is_plural_subject: return verb_base
        # Simple rule, might need expansion for irregular verbs or different tenses
        return verb_base + "s" if not verb_base.endswith(("s", "sh", "ch", "x", "z")) else verb_base + "es"

    def _get_concept_word(self, cid: str, role: str, is_plural_subject: bool = False) -> str:
        """
        Retrieves a word form for a concept ID based on its role in a sentence.
        Handles basic conjugation for present tense verbs and subject pluralization.
        """
        node_data = self.cg.nodes.get(cid)
        if not node_data:
            # print(f"Warning: Node {cid} not found in CG for language generation.")
            # Fallback for missing node ID based on role
            if role == "subject": return self.DEFAULT_SUBJECT_WORD
            if role == "verb_base" or role == "verb_present": return self.DEFAULT_VERB_WORD
            if role == "object": return self.DEFAULT_OBJECT_WORD
            return f"[{cid}_unknown]"

        word_forms = node_data.get("wordForms", {"base": str(cid)})
        base_form = word_forms.get("base", str(cid))
        
        if role == "subject":
            # Basic pluralization for subjects, may need enhancement
            # TODO: Use linguistic features from node_data if available (e.g., number: singular/plural)
            return base_form + "s" if is_plural_subject and not base_form.endswith("s") else base_form
        if role == "object":
            return base_form
        if role == "verb_base":
            return base_form
        if role == "verb_present":
            return self._conjugate_present(base_form, is_plural_subject)
        return base_form # Default fallback

    def generate(self, query_string: str, engine_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates a sentence based on the current engine state (active concepts,
        reflection, retrieved patterns) and the query type.

        Args:
            query_string: The input query string, used to determine if a question
                          or statement template should be used and potentially for context.
            engine_state: The shared engine state dictionary, providing context like
                          'reflection' and 'retrieved_patterns'.

        Returns:
            The generated sentence as a string.
        """
        if engine_state is None:
            engine_state = {}

        generated_sentence = ""
        is_plural_subject = False # TODO: Determine from context or node data

        # 1. Try to use retrieved patterns if available and relevant
        retrieved_patterns: List[Tuple[str, str, Dict[str, Any], float]] = engine_state.get("retrieved_patterns", [])
        if retrieved_patterns and retrieved_patterns[0][3] > 0.7: # High similarity threshold for using pattern
            # Use the response from the best matching pattern
            # TODO: Adapt this response using current context if possible. For now, direct use.
            generated_sentence = retrieved_patterns[0][1] # response part of the tuple
            self.narrative.append(f"[PatternBased] {generated_sentence}")
            return generated_sentence

        # 2. If no strong pattern, use reflection or active concepts
        source_concepts: List[Dict[str, Any]] = engine_state.get("reflection", [])
        if not source_concepts: # Fallback to general active concepts
            source_concepts = self.cg.getActiveConceptPath(threshold=0.5) # Slightly lower threshold for generation

        if not source_concepts: # Still no concepts to work with
            generated_sentence = "I'm not sure what to say about that."
            self.narrative.append(generated_sentence)
            return generated_sentence
            
        # Attempt to build a sentence from source_concepts (simplified graph traversal idea)
        # This is a placeholder for more sophisticated graph-to-text logic.
        
        # Try to find SVO from reflection/active concepts
        # This logic is still very basic and relies on the order from getActiveConceptPath
        # or the structure of reflection.
        
        subj_cid = source_concepts[0].get("id", self.DEFAULT_SUBJECT_CID) if len(source_concepts) > 0 else self.DEFAULT_SUBJECT_CID
        
        # Try to find a verb related to the subject
        verb_cid = self.DEFAULT_VERB_CID
        obj_cid = self.DEFAULT_OBJECT_CID

        # Simple graph traversal: look for an outgoing edge from subject that might be a verb or lead to an object
        # This is highly experimental and needs refinement.
        if subj_cid in self.cg.nodes:
            potential_verbs_objects = []
            for (s, t), edge_data in self.cg.edges.items():
                if s == subj_cid:
                    # If edge relation itself is verb-like or t is a verb concept
                    relation_is_verb = edge_data.get("relation", "").startswith("verb_") or \
                                       self.cg.nodes.get(t, {}).get("properties", {}).get("pos") == "VERB"
                    if relation_is_verb and t not in [subj_cid]: # Avoid self-loops as main verb
                        verb_cid = t
                        # Now look for an object connected to this verb
                        for (s2, t2), edge_data2 in self.cg.edges.items():
                            if s2 == verb_cid and t2 not in [subj_cid, verb_cid]:
                                obj_cid = t2
                                potential_verbs_objects.append((verb_cid, obj_cid, edge_data.get("weight",0) + edge_data2.get("weight",0)))
                                break # Found one SVO path
                        if not potential_verbs_objects or obj_cid == self.DEFAULT_OBJECT_CID : # if verb found but no object from it
                             potential_verbs_objects.append((verb_cid, self.DEFAULT_OBJECT_CID, edge_data.get("weight",0)))
                        break # Found a primary verb for the subject
            
            if potential_verbs_objects: # Pick the one with highest combined weight if multiple
                potential_verbs_objects.sort(key=lambda x: -x[2])
                verb_cid = potential_verbs_objects[0][0]
                obj_cid = potential_verbs_objects[0][1]
            elif len(source_concepts) > 1 : # Fallback to second active concept as verb if no relation found
                 verb_cid = source_concepts[1].get("id", self.DEFAULT_VERB_CID)
                 if len(source_concepts) > 2:
                     obj_cid = source_concepts[2].get("id", self.DEFAULT_OBJECT_CID)


        # Ensure default concepts exist if they are to be used
        default_nodes_to_add = {
            self.DEFAULT_SUBJECT_CID: self.DEFAULT_SUBJECT_WORD,
            self.DEFAULT_VERB_CID: self.DEFAULT_VERB_WORD,
            self.DEFAULT_OBJECT_CID: self.DEFAULT_OBJECT_WORD
        }
        for cid_default, word_default in default_nodes_to_add.items():
            if cid_default in [subj_cid, verb_cid, obj_cid] and cid_default not in self.cg.nodes:
                self.cg.addEnhancedNode(cid_default, linguistics={"wordForms": {"base": word_default}})
        
        template_key = "question_what" if "?" in query_string else "statement"
        chosen_template = self.templates.get(template_key, self.templates["statement"])
        
        generated_sentence = chosen_template.format(
            subject=self._get_concept_word(subj_cid, "subject", is_plural_subject),
            verb_base=self._get_concept_word(verb_cid, "verb_base"),
            verb_present=self._get_concept_word(verb_cid, "verb_present", is_plural_subject),
            object=self._get_concept_word(obj_cid, "object")
        ).strip()

        self.narrative.append(generated_sentence)
        return generated_sentence
