import json
from typing import List, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from core.goals import AgentTaskConfig
from core.interfaces import LLMInterface

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

class LLMConversationEvaluator(ConversationEvaluator):
    """Evaluates conversation outcome using an LLM"""
    def __init__(self, evaluation_llm: LLMInterface):
        self.llm = evaluation_llm
    
    def evaluate(self, conversation_history: List[Dict[str, str]], task_config: AgentTaskConfig) -> EvaluationResponse:
        prompt = self._create_evaluation_prompt(conversation_history, task_config)
        system_prompt = self._get_evaluator_system_prompt()
        evaluation_response = self.llm.generate_response_with_structured_output(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": prompt}],
            response_format=EvaluationResponse
        )
         
        return evaluation_response
    
    def _generate_metrics_prompt(self) -> str:
        metrics_str = ""
        with open("llm_testing/core/eval_metrics.json", "r") as file:
            metrics = json.load(file)
        
        for metric in metrics:
            curr_metric = metrics[metric]
            metrics_str += f"Metric: {metric}\nEvaluation Prompt: {curr_metric['eval_prompt']}\n"

        return metrics_str

    def _get_evaluator_system_prompt(self) -> str:
        # TODO: use OpenAI's structured output mode for if typeof(self.llm) == OpenAIProvider
        return f"""You are an objective phone agent conversation evaluator who evalutes AI agents calling to businesses. You will be provided a call transcript and score it across the different provided metrics.
For each metric, provide a score according to the scoring format and an explanation of your evaluation.
success_flag is a boolean value that indicates whether the metric was achieved. range_score is a number between 0 and 10 that indicates the degree to which the metric was achieved.

# Metrics
{self._generate_metrics_prompt()}
"""
    
    def _create_evaluation_prompt(self, 
                                conversation_history: List[Dict[str, str]],
                                task_config: AgentTaskConfig) -> str:
        return f"""Please evaluate the following conversation according to the provided metrics:

Task and additional context: {task_config.generate_system_prompt()}
Success Criteria: {json.dumps(task_config.success_criteria, indent=2)}

Conversation:
{self._format_conversation(conversation_history)}
"""

    def _format_conversation(self, history: List[Dict[str, str]]) -> str:
        return "\n".join([
            f"{turn['speaker']}: {turn['text']}" for turn in history
        ])

