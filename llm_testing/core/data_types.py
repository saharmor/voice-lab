from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime
from enum import Enum

from core.agent_config import AgentTaskConfig
from core.evaluator import EvaluationResponse

@dataclass
class TestResult:
    evaluation_result: EvaluationResponse
    conversation_history: List[Dict[str, str]]

@dataclass
class ConversationContext:
    """Represents the context of the conversation"""
    system_prompt: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class ConversationEndStatus:
    reason: Optional[str] = None
    who_ended: Optional[str] = None
    termination_evidence: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.who_ended and self.who_ended not in ['callee', 'agent']:
            raise ValueError("who_ended must be either 'callee' or 'agent'")

class LLMResponse:
    def __init__(self, response_content: str, tools_called):
        """
        Args:
            response_content: The actual response text from the LLM
            end_status: The conversation end status
        """
        self.response_content = response_content
        self.tools_called = tools_called

# TODO get rid of all 'callee' and 'agent' literal strings
class EntitySpeaking(Enum):
    """Represents the entity speaking in the conversation"""
    VOICE_AGENT = "voice_agent"
    CALLEE = "callee"

class TestedComponentType(Enum):
    """Represents the component being tested"""
    UNDERLYING_LLM = "underlying_llm"
    AGENT = "agent"

@dataclass
class TestedComponent:
    """Represents a component being tested"""
    type: TestedComponentType
    variations: List[str]

class TestScenario:
    """Represents a scenario for a test"""
    tested_components: List[TestedComponent]
    agent_config: AgentTaskConfig
