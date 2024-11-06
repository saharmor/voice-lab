from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field

from core.agent_config import AgentTaskConfig

@dataclass
class EvaluationMetadata:
    """Metadata for conversation evaluation"""
    key_observations: List[str] = field(default_factory=list)
    missed_objectives: List[str] = field(default_factory=list)
    
@dataclass
class ConversationEvaluation:
    """Result of evaluating a conversation against its goals"""
    goal_achieved: bool
    reasoning: str
    metadata: EvaluationMetadata


class MetricResult(BaseModel):
    name: str = Field(description="name of the metric")
    eval_output_type: str = Field(description="either 'success_flag' or 'range_score'")
    eval_output: str = Field(description="boolean if success flag or numeric score if range_score")
    eval_output_success_threshold: int = Field(description="threshold for success if eval_output_type is range_score")
    reasoning: str = Field(description="explanation of how the output was determined") # TODO consider using CoT for better reasoning https://platform.openai.com/docs/guides/structured-outputs#chain-of-thought
    evidence: str = Field(description="evidence/quotes from the conversation history supporting your output score. Can be empty if no evidence is needed.")


class EvaluationResponse(BaseModel):
    summary: str = Field(description="summary of the overall conversation")
    evaluation_results: List[MetricResult]

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
