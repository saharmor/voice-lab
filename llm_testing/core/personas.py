from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class Mood(Enum):
    """Enumeration of possible persona moods"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    HELPFUL = "helpful"
    CONFUSED = "confused"
    PROFESSIONAL = "professional"
    IMPATIENT = "impatient"

class ResponseStyle(Enum):
    """Enumeration of possible response styles"""
    FORMAL = "formal"
    CASUAL = "casual"
    CURT = "curt"
    VERBOSE = "verbose"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"


@dataclass
class CalleePersona:
    """Represents a character/role in the conversation"""
    name: str
    description: str
    role: str
    traits: List[str]
    initial_message: str
    mood: Mood
    response_style: ResponseStyle
    additional_context: Dict[str, Any]
    background_info: Optional[str] = None
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary format"""
        return {
            "name": self.name,
            "description": self.description,
            "role": self.role,
            "traits": self.traits,  
            "initial_message": self.initial_message,
            "mood": self.mood.value,
            "response_style": self.response_style.value,
            "additional_context": self.additional_context,
            "background_info": self.background_info,
            "knowledge_base": self.knowledge_base
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalleePersona':
        return cls(
            name=data["name"],
            description=data["description"],
            role=data["role"],
            traits=data["traits"],
            initial_message=data["initial_message"],
            mood=Mood(data["mood"]),
            response_style=ResponseStyle(data["response_style"]),
            additional_context=data["additional_context"],
            background_info=data.get("background_info"),
            knowledge_base=data.get("knowledge_base", {})
        )

