from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class ConversationGoal:
    """Represents the desired outcome of a conversation"""
    description: str
    success_criteria: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Example success_criteria for hotel booking:
    # {
    #     "booking_dates": {"start": "2024-12-12", "end": "2024-12-24"},
    #     "required_confirmations": ["booking_reference", "price"],
    #     "max_turns": 10
    # }

@dataclass
class Persona:
    """Represents the character/role being played by the conversation participant"""
    name: str
    description: str
    traits: List[str]
    constraints: Dict[str, Any]
    
    # Example for hotel receptionist:
    # {
    #     "available_dates": {"start": "2024-12-12", "end": "2024-12-14"},
    #     "mood": "angry",
    #     "response_style": "curt"
    # }
