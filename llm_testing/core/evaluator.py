import json
from typing import List, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from core.goals import AgentTaskConfig
from core.interfaces import LLMInterface
from core.data_types import ConversationContext

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

class ConversationEvaluator(ABC):
    """Abstract base class for conversation evaluators"""
    @abstractmethod
    def evaluate(self, 
                conversation_history: List[Dict[str, str]],
                task_config: AgentTaskConfig) -> ConversationEvaluation:
        pass

class LLMConversationEvaluator(ConversationEvaluator):
    """Evaluates conversation outcome using an LLM"""
    def __init__(self, evaluation_llm: LLMInterface):
        self.llm = evaluation_llm
    
    def evaluate(self, 
                conversation_history: List[Dict[str, str]],
                task_config: AgentTaskConfig) -> ConversationEvaluation:
        prompt = self._create_evaluation_prompt(conversation_history, task_config)
        evaluation_response = self.llm.generate_response(
            ConversationContext(system_prompt=self._get_evaluator_system_prompt()),
            prompt
        )
        
        return self._parse_evaluation_response(evaluation_response)
    
    def _get_evaluator_system_prompt(self) -> str:
        # TODO: use OpenAI's structured output mode for if typeof(self.llm) == OpenAIProvider
        return """You are an objective phone agent conversation evaluator. Your task is to determine if the agent achieved its stated goal and success criteria.
Provide your evaluation in JSON format with the following structure:
{
    "goal_achieved": boolean,
    "reasoning": "detailed explanation of your evaluation",
    "metadata": {
        "key_observations": [],
        "missed_objectives": []
    }
}"""
    
    def _create_evaluation_prompt(self, 
                                conversation_history: List[Dict[str, str]],
                                task_config: AgentTaskConfig) -> str:
        return f"""Please evaluate the following conversation according to the provided goal and success criteria:

Goal: {task_config.system_prompt}
Success Criteria: {json.dumps(task_config.success_criteria, indent=2)}

Conversation:
{self._format_conversation(conversation_history)}
"""

    def _format_conversation(self, history: List[Dict[str, str]]) -> str:
        return "\n".join([
            f"{turn['speaker']}: {turn['text']}" for turn in history
        ])
    
    def _parse_evaluation_response(self, response: str) -> ConversationEvaluation:
        try:
            evaluation_data = json.loads(response)
            return ConversationEvaluation(
                goal_achieved=evaluation_data["goal_achieved"],
                reasoning=evaluation_data["reasoning"],
                metadata=evaluation_data.get("metadata", {})
            )
        except json.JSONDecodeError:
            return ConversationEvaluation(
                success=False,
                goal_achieved=False,
                reasoning="Failed to parse evaluation response",
                metadata={"error": "Invalid evaluation format"}
            )


