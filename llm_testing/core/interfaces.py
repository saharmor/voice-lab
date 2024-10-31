from abc import ABC, abstractmethod
from llm_testing.core.data_types import ConversationContext

class LLMInterface(ABC):
    """Abstract interface for LLM interactions"""
    @abstractmethod
    def generate_response(self, context: ConversationContext, user_input: str) -> str:
        """Generate a response given the context and user input"""
        pass
