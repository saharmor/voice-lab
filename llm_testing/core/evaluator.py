import json
from typing import List, Dict
from abc import ABC, abstractmethod


from core.agent_config import AgentTaskConfig
from core.interfaces import LLMInterface
from core.data_types import EntitySpeaking, EvaluationResponse
from core.personas import CalleePersona


class ConversationEvaluator(ABC):
    """Abstract base class for conversation evaluators"""
    @abstractmethod
    def evaluate(self, 
                conversation_history: List[Dict[str, str]],
                task_config: AgentTaskConfig,
                persona: CalleePersona) -> EvaluationResponse:
        pass

class LLMConversationEvaluator(ConversationEvaluator):
    """Evaluates conversation outcome using an LLM"""
    def __init__(self, evaluation_llm: LLMInterface):
        self.llm = evaluation_llm
    
    def evaluate(self, conversation_history: List[Dict[str, str]], task_config: AgentTaskConfig, persona: CalleePersona) -> EvaluationResponse:
        prompt = self._create_evaluation_prompt(conversation_history, task_config, persona)
        system_prompt = self._get_evaluator_system_prompt()
        evaluation_response = self.llm.generate_response_with_structured_output(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": prompt}],
            response_format=EvaluationResponse
        )
         
        return evaluation_response
    
    def _generate_metrics_prompt(self) -> str:
        metrics_str = ""
        with open("llm_testing/config/eval_metrics.json", "r") as file:
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
                                task_config: AgentTaskConfig,
                                persona: CalleePersona) -> str:
        return f"""Please evaluate the following conversation according to the provided metrics:

Task and additional context: {task_config.generate_system_prompt()}
Success Criteria: {json.dumps(task_config.success_criteria, indent=2)}

Conversation:
{self._format_conversation(conversation_history, persona)}
"""

    def _format_conversation(self, history: List[Dict[str, str]], persona: CalleePersona) -> str:
        formatted_history = ""
        for message in history:
            if message["speaker"] == EntitySpeaking.CALLEE.value:
                formatted_history += f"{' '.join(persona.role.split('_')).title() if persona else 'Callee'}: {message['text']}\n"
            else:
                formatted_history += f"Voice Agent: {message['text']}\n"
        
        return formatted_history


