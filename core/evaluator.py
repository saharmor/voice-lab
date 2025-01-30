import json
from typing import List, Dict, Optional
from abc import ABC, abstractmethod


from .agent_config import AgentTaskConfig
from .interfaces import LLMInterface
from .data_types import EvaluationResponse
from .personas import CalleePersona


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
    def __init__(self, evaluation_llm: LLMInterface, eval_metrics_path: str, eval_system_prompt: str):
        self.llm = evaluation_llm
        self.eval_metrics_config_path = eval_metrics_path
        self.eval_system_prompt = eval_system_prompt
    
    def evaluate(self, conversation_history: List[Dict[str, str]], task_config: AgentTaskConfig,
                  persona: CalleePersona, success_criteria: Optional[str] = None,
                  scenario_guidelines: Optional[str] = None) -> EvaluationResponse:
        prompt = self._create_evaluation_prompt(conversation_history, task_config, persona, success_criteria, scenario_guidelines)
        system_prompt = self._get_evaluator_system_prompt()
        evaluation_response = self.llm.generate_response_with_structured_output(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": prompt}],
            response_format=EvaluationResponse
        )
         
        return evaluation_response
    
    def _generate_metrics_prompt(self, eval_metrics_path: str) -> str:
        metrics_str = ""
        with open(eval_metrics_path, "r") as file:
            metrics = json.load(file)
        
        for metric in metrics:
            curr_metric = metrics[metric]
            metrics_str += f"Metric: {metric}\nEvaluation Prompt: {curr_metric['eval_prompt']}\n"

        return metrics_str

    def _get_evaluator_system_prompt(self) -> str:
        # TODO: use OpenAI's structured output mode for if typeof(self.llm) == OpenAIProvider
        return f"""{self.eval_system_prompt}
For each metric, provide a score according to the scoring format and an explanation of your evaluation.
success_flag is a boolean value that indicates whether the metric was achieved. range_score is a number between 0 and 10 that indicates the degree to which the metric was achieved.

# Metrics
{self._generate_metrics_prompt(self.eval_metrics_config_path)}
"""
    
    def _create_evaluation_prompt(self, 
                                conversation_history: List[Dict[str, str]],
                                task_config: Optional[AgentTaskConfig] = None,
                                persona: Optional[CalleePersona] = None,
                                success_criteria: Optional[str] = None,
                                scenario_guidelines: Optional[str] = None) -> str:
        return f"""Please evaluate the following conversation according to the provided metrics, success criteria, and guidelines, if exists:

{f"# Task and additional context\n{task_config.generate_system_prompt()}" if task_config else ""}

{f"# Success criteria\n{json.dumps(task_config.success_criteria, indent=2) if task_config else success_criteria}"}

{f"# Guidelines\n{scenario_guidelines}" if scenario_guidelines else ""}

{f"# Conversation\n{self._format_conversation(conversation_history, persona)}"}
"""

    def _format_conversation(self, history: List[Dict[str, str]], persona: CalleePersona) -> str:
        formatted_history = ""
        # TODO check if works for llm_testing
        for message in history:
            formatted_history += f"{message['role']}: {message['content']}\n"

        return formatted_history


