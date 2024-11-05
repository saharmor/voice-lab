from dataclasses import dataclass, field
import json
from typing import Dict, Any, List

@dataclass
class AgentTaskConfig:
    """Represents the desired outcome of a conversation"""
    system_prompt: str
    initial_message: str
    tool_calls: List[Dict[str, Any]]
    success_criteria: Dict[str, Any]
    additional_context: Dict[str, Any] = field(default_factory=dict)

    def generate_system_prompt(self) -> str:
        return f'''System prompt: {self.system_prompt}
        Additional context for you to use when generating your response: {json.dumps(self.additional_context, indent=2)}
        '''

