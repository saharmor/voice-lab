from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class AgentTaskConfig:
    """Represents the desired outcome of a conversation"""
    system_prompt: str
    initial_message: str
    tool_calls: List[Dict[str, Any]]
    success_criteria: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Example success_criteria for hotel booking:
    # {
    #     "booking_dates": {"start": "2024-12-12", "end": "2024-12-24"},
    #     "required_confirmations": ["booking_reference", "price"],
    #     "max_turns": 10
    # }

