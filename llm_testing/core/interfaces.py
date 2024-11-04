from abc import ABC, abstractmethod
from core.data_types import ConversationContext, LLMResponse

class LLMInterface(ABC):
    """Abstract interface for LLM interactions"""
    @abstractmethod
    def generate_response(self, context: ConversationContext, user_input: str) -> LLMResponse:
        """
        Generate a response given the context and user input
        
        Args:
            context: The conversation context
            user_input: The user's input text
            
        Returns:
            LLMResponse containing the response content and conversation end status
        """
        pass
