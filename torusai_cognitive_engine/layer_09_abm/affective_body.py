from typing import Dict, Any, Optional # Added Optional

# From User Part 3 - Labelled "Layer 9: Affective Body Layer"
# Corresponds to L9 Affective Body in the 15-layer specification.
# Determines a global affective state (valence) based on recent interactions.

class AffectiveBodyLayer:
    """
    Simulates an affective state for the engine by calculating a 'valence' score.
    This score is based on a simple heuristic analysis of the engine's last response.
    The valence is stored in the shared engine state and can influence other layers.
    
    Note: The current sentiment detection heuristic is very basic and would need
    significant improvement for more nuanced affective modeling.
    """
    DEFAULT_POSITIVE_VALENCE: float = 0.1
    DEFAULT_NEGATIVE_VALENCE: float = -0.05
    POSITIVE_KEYWORD: str = "good" # Example keyword

    def __init__(self,
                 positive_valence: Optional[float] = None,
                 negative_valence: Optional[float] = None,
                 positive_keyword: Optional[str] = None):
        """
        Initializes the AffectiveBodyLayer.

        Args:
            positive_valence: The valence value to set for positive sentiment.
            negative_valence: The valence value to set for negative sentiment.
            positive_keyword: The keyword to look for to indicate positive sentiment.
        """
        self.positive_valence: float = positive_valence if positive_valence is not None else self.DEFAULT_POSITIVE_VALENCE
        self.negative_valence: float = negative_valence if negative_valence is not None else self.DEFAULT_NEGATIVE_VALENCE
        self.positive_keyword: str = positive_keyword if positive_keyword is not None else self.POSITIVE_KEYWORD

    def run(self, state_dict: Dict[str, Any]) -> None:
        """
        Calculates and sets the 'valence' in the shared state dictionary.
        It checks if the `positive_keyword` is present in the `last_response`.

        Args:
            state_dict: The shared engine state dictionary, expected to contain
                        'last_response' and will be updated with 'valence'.
        """
        last_response = state_dict.get("last_response", "").lower()
        
        # Basic heuristic: check for a positive keyword.
        # This could be expanded with more sophisticated sentiment analysis.
        if self.positive_keyword in last_response:
            valence_value = self.positive_valence
        else:
            valence_value = self.negative_valence
            
        state_dict['valence'] = valence_value
