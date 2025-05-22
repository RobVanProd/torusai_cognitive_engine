import unittest
from unittest.mock import patch, MagicMock
from torusai_cognitive_engine.document_processor.text_to_graph import (
    _parse_entity_part, 
    populate_graph_from_text, 
    extract_triples_via_uie, # Imported for direct patching
    UIE_MODEL_LOADED, # To manage its state
    UIE_MODEL, # To potentially mock its behavior if _load_uie_model_if_needed is not fully bypassed
    UIE_TOKENIZER, # Same as UIE_MODEL
    NLP_SPACY, # To disable spaCy for focused UIE tests
    nltk # To disable nltk for focused UIE tests
)
from torusai_cognitive_engine.layer_04_dcg.concept_graph import EnhancedConceptGraph

# Store original NLP tool states
ORIGINAL_NLP_SPACY = NLP_SPACY
ORIGINAL_NLTK = nltk

class TestParseEntityPart(unittest.TestCase):
    def test_with_type(self):
        self.assertEqual(_parse_entity_part("entity [TYPE]"), ("entity", "TYPE"))

    def test_no_type(self):
        self.assertEqual(_parse_entity_part("entity"), ("entity", None))

    def test_extra_spaces(self):
        self.assertEqual(_parse_entity_part("  entity  [  TYPE  ]  "), ("entity", "TYPE"))
        self.assertEqual(_parse_entity_part("  entity  [TYPE]"), ("entity", "TYPE"))
        self.assertEqual(_parse_entity_part("entity[TYPE  ]"), ("entity", "TYPE"))

    def test_empty_input(self):
        self.assertEqual(_parse_entity_part(""), ("", None))
        self.assertEqual(_parse_entity_part("  "), ("", None))

    def test_type_only(self):
        # This case might depend on desired behavior, current regex expects text before type
        self.assertEqual(_parse_entity_part("[TYPE]"), ("[TYPE]", None)) # Current behavior

    def test_malformed_type(self):
        self.assertEqual(_parse_entity_part("entity [TYPE"), ("entity [TYPE", None))
        self.assertEqual(_parse_entity_part("entity TYPE]"), ("entity TYPE]", None))
        self.assertEqual(_parse_entity_part("entity []"), ("entity []", None)) # Current behavior, type is empty string

class TestUIEGraphPopulation(unittest.TestCase):
    def setUp(self):
        self.cg = EnhancedConceptGraph()
        self.original_uie_model_loaded = UIE_MODEL_LOADED
        # Ensure UIE is conceptually 'on' for these tests, mock will bypass actual loading
        global UIE_MODEL_LOADED
        UIE_MODEL_LOADED = True 

        # Disable spaCy and NLTK for these tests to isolate UIE behavior
        global NLP_SPACY, nltk
        NLP_SPACY = None
        nltk = None


    def tearDown(self):
        global UIE_MODEL_LOADED
        UIE_MODEL_LOADED = self.original_uie_model_loaded
        
        # Restore NLP tools
        global NLP_SPACY, nltk
        NLP_SPACY = ORIGINAL_NLP_SPACY
        nltk = ORIGINAL_NLTK


    @patch('torusai_cognitive_engine.document_processor.text_to_graph.extract_triples_via_uie')
    def test_basic_triple_population(self, mock_extract_triples):
        mock_triples = [
            {"subject": "cat", "subject_type": "ANIMAL", "predicate": "chases", "object": "mouse", "object_type": "ANIMAL", "confidence": 0.9}
        ]
        mock_extract_triples.return_value = mock_triples

        populate_graph_from_text("Some text about a cat and a mouse", self.cg, {})

        self.assertIn("cat", self.cg.nodes)
        self.assertEqual(self.cg.nodes["cat"]["properties"]["source"], "uie_import")
        self.assertEqual(self.cg.nodes["cat"]["properties"]["entity_type"], "ANIMAL")

        self.assertIn("mouse", self.cg.nodes)
        self.assertEqual(self.cg.nodes["mouse"]["properties"]["source"], "uie_import")
        self.assertEqual(self.cg.nodes["mouse"]["properties"]["entity_type"], "ANIMAL")

        self.assertIn(("cat", "mouse"), self.cg.edges)
        edge_data = self.cg.edges[("cat", "mouse")]
        self.assertEqual(edge_data["relation"], "chases")
        self.assertEqual(edge_data["weight"], 0.9)
        self.assertEqual(edge_data["properties"]["source"], "uie_triple")

    @patch('torusai_cognitive_engine.document_processor.text_to_graph.extract_triples_via_uie')
    def test_no_triples_returned(self, mock_extract_triples):
        mock_extract_triples.return_value = []
        
        initial_node_count = len(self.cg.nodes)
        initial_edge_count = len(self.cg.edges)

        populate_graph_from_text("Some other text", self.cg, {})
        
        # Check that no UIE-specific nodes were added. 
        # If spaCy/NLTK were active, they might add nodes.
        # For this test, we rely on them being disabled in setUp.
        self.assertEqual(len(self.cg.nodes), initial_node_count, "Node count should not change if UIE returns no triples and other NLP is off")
        self.assertEqual(len(self.cg.edges), initial_edge_count, "Edge count should not change if UIE returns no triples and other NLP is off")
        self.assertNotIn("hypothetical_uie_node", self.cg.nodes) # Double check

    @patch('torusai_cognitive_engine.document_processor.text_to_graph.extract_triples_via_uie')
    def test_predicate_normalization(self, mock_extract_triples):
        mock_triples = [
            {"subject": "lion", "subject_type": "ANIMAL", "predicate": "Is King Of", "object": "jungle", "object_type": "LOCATION", "confidence": 0.8}
        ]
        mock_extract_triples.return_value = mock_triples

        populate_graph_from_text("The lion is king of the jungle", self.cg, {})

        self.assertIn(("lion", "jungle"), self.cg.edges)
        edge_data = self.cg.edges[("lion", "jungle")]
        self.assertEqual(edge_data["relation"], "is_king_of") # Normalized

    @patch('torusai_cognitive_engine.document_processor.text_to_graph.extract_triples_via_uie')
    def test_node_property_update_from_uie(self, mock_extract_triples):
        # Pre-populate graph with a node that UIE will also identify
        self.cg.addEnhancedNode("dog", properties={"source": "initial_source", "initial_prop": "value"}, 
                                linguistics={"wordForms": {"base": "dog"}})
        self.assertEqual(self.cg.nodes["dog"]["properties"]["source"], "initial_source")
        self.assertNotIn("entity_type", self.cg.nodes["dog"]["properties"])

        mock_triples = [
            {"subject": "dog", "subject_type": "CANINE", "predicate": "barks_at", "object": "mailman", "object_type": "PERSON", "confidence": 0.7}
        ]
        mock_extract_triples.return_value = mock_triples

        populate_graph_from_text("The dog barks at the mailman", self.cg, {})

        self.assertIn("dog", self.cg.nodes)
        # Source should remain the original one, as per current implementation (doc_freq is updated, new props added)
        self.assertEqual(self.cg.nodes["dog"]["properties"]["source"], "initial_source")
        self.assertEqual(self.cg.nodes["dog"]["properties"]["initial_prop"], "value") # Existing props should persist
        self.assertEqual(self.cg.nodes["dog"]["properties"]["entity_type"], "CANINE") # New property added
        self.assertEqual(self.cg.nodes["dog"]["doc_freq"], 1) # doc_freq should be incremented

        self.assertIn("mailman", self.cg.nodes)
        self.assertEqual(self.cg.nodes["mailman"]["properties"]["source"], "uie_import")
        self.assertEqual(self.cg.nodes["mailman"]["properties"]["entity_type"], "PERSON")

    @patch('torusai_cognitive_engine.document_processor.text_to_graph._load_uie_model_if_needed')
    def test_uie_model_not_loaded_graceful_fallback_mocked(self, mock_load_model):
        # This test ensures that if _load_uie_model_if_needed sets UIE_MODEL_LOADED to False,
        # then extract_triples_via_uie (and thus UIE processing) is skipped.
        # We mock _load_uie_model_if_needed to simulate this.
        
        # Simulate _load_uie_model_if_needed failing to load the model
        def side_effect_load_model():
            global UIE_MODEL_LOADED
            UIE_MODEL_LOADED = False
        mock_load_model.side_effect = side_effect_load_model
        
        # Explicitly set UIE_MODEL_LOADED to True before call, so load_model attempts to run
        global UIE_MODEL_LOADED
        UIE_MODEL_LOADED = True

        # Since extract_triples_via_uie will not be called if UIE_MODEL_LOADED is False after _load_uie_model_if_needed,
        # we don't need to mock extract_triples_via_uie itself.
        # We are testing the logic within populate_graph_from_text that checks UIE_MODEL_LOADED.

        initial_node_count = len(self.cg.nodes)
        populate_graph_from_text("Text when UIE model fails to load", self.cg, {})
        
        # Assert that UIE_MODEL_LOADED became False
        self.assertFalse(UIE_MODEL_LOADED)
        # Assert that no UIE-specific nodes were added (assuming NLTK/spaCy are off)
        self.assertEqual(len(self.cg.nodes), initial_node_count)
        # Reset UIE_MODEL_LOADED to True for other tests if side_effect changed it globally for the class
        UIE_MODEL_LOADED = True


if __name__ == '__main__':
    unittest.main()
