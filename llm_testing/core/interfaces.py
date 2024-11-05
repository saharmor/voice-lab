from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from core.data_types import ConversationContext, EntitySpeaking, LLMResponse

class LLMInterface(ABC):
    """Abstract interface for LLM interactions"""
    @abstractmethod
    def generate_response_with_conversation_history(self, context: Optional[ConversationContext],
                                                     entity_speaking: EntitySpeaking,
                                                     tools: Optional[List[Dict[str, Any]]] = None,
                                                     user_input: str = None) -> List[Dict[str, Any]]:
                                                     
        pass

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = []) -> LLMResponse:
        """
        Generate a response given the context and user input
        
        Args:
            context: The conversation context
            user_input: The user's input text
            
        Returns:
            LLMResponse containing the response content and conversation end status
        """
        pass
