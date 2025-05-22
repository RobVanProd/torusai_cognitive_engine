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

    def generate(self, query_string: str) -> str:
        """
        Generates a sentence based on the current active concepts and the query type.

        Args:
            query_string: The input query string, used to determine if a question
                          or statement template should be used.

        Returns:
            The generated sentence as a string.
        """
        active_concepts: List[Dict[str, Any]] = self.cg.getActiveConceptPath()
        
        # Determine subject, verb, object CIDs from active concepts, with defaults
        subj_cid = active_concepts[0]["id"] if len(active_concepts) > 0 else self.DEFAULT_SUBJECT_CID
        verb_cid = active_concepts[1]["id"] if len(active_concepts) > 1 else self.DEFAULT_VERB_CID
        obj_cid = active_concepts[2]["id"] if len(active_concepts) > 2 else self.DEFAULT_OBJECT_CID

        # Ensure default concepts exist in the graph for generation if they are to be used.
        # This side-effect of adding nodes during generation might be better handled
        # by pre-populating essential default nodes or having robust fallback in _get_concept_word.
        if subj_cid == self.DEFAULT_SUBJECT_CID and subj_cid not in self.cg.nodes:
            self.cg.addEnhancedNode(subj_cid, linguistics={"wordForms": {"base": self.DEFAULT_SUBJECT_WORD}})
        if verb_cid == self.DEFAULT_VERB_CID and verb_cid not in self.cg.nodes:
            self.cg.addEnhancedNode(verb_cid, linguistics={"wordForms": {"base": self.DEFAULT_VERB_WORD}})
        if obj_cid == self.DEFAULT_OBJECT_CID and obj_cid not in self.cg.nodes:
            self.cg.addEnhancedNode(obj_cid, linguistics={"wordForms": {"base": self.DEFAULT_OBJECT_WORD}})
        
        # For this basic generator, assume subject is singular unless more sophisticated logic is added.
        # TODO: Determine plurality from node data or query context.
        is_plural_subject = False

        template_key = "question_what" if "?" in query_string else "statement"
        chosen_template = self.templates.get(template_key, self.templates["statement"]) # Fallback to statement
        
        generated_sentence = chosen_template.format(
            subject=self._get_concept_word(subj_cid, "subject", is_plural_subject),
            verb_base=self._get_concept_word(verb_cid, "verb_base"),
            verb_present=self._get_concept_word(verb_cid, "verb_present", is_plural_subject),
            object=self._get_concept_word(obj_cid, "object")
        ).strip()

        self.narrative.append(generated_sentence)
        return generated_sentence
