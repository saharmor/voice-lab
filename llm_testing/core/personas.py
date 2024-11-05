from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
from pathlib import Path


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

class PersonaTemplate:
    """Pre-defined templates for common persona types"""
    
    @staticmethod
    def hotel_receptionist(
        name: str = "Hotel Receptionist",
        mood: Mood = Mood.PROFESSIONAL,
        response_style: ResponseStyle = ResponseStyle.FORMAL,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> CalleePersona:
        """Create a hotel receptionist persona"""
        return CalleePersona(
            name=name,
            description="Hotel receptionist responsible for managing bookings",
            initial_message=f"Hi, {name} here, how can I help you today?",
            role="hotel_receptionist",
            traits=[
                "organized",
                "detail-oriented",
                "customer-focused"
            ],
            mood=mood,
            response_style=response_style,
            additional_context=additional_context,
            background_info="Experienced in hotel management and customer service",
            knowledge_base={
                "booking_policies": [
                    "24-hour cancellation policy",
                    "Credit card required for reservation",
                    "Check-in time: 3 PM",
                    "Check-out time: 11 AM"
                ]
            }
        )
    
    @staticmethod
    def pharmacy_clerk(
        name: str = "Pharmacy Clerk",
        mood: Mood = Mood.HELPFUL,
        response_style: ResponseStyle = ResponseStyle.PROFESSIONAL,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> CalleePersona:
        """Create a pharmacy clerk persona"""
        
        return CalleePersona(
            name=name,
            description="Pharmacy clerk assisting with medication inquiries",
            initial_message=f"Hi, {name} here, how can I help you today?",
            role="pharmacy_clerk",
            traits=[
                "knowledgeable",
                "patient",
                "detail-oriented",
                "professional"
            ],
            mood=mood,
            response_style=response_style,
            additional_context=additional_context,
            background_info="Certified pharmacy technician with healthcare background",
            knowledge_base={
                "policies": [
                    "Prescription required for controlled substances",
                    "Insurance information needed for claims",
                    "Generic substitution when available"
                ]
            }
        )

class PersonaLibrary:
    """Manages a collection of personas"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("personas")
        self.personas: Dict[str, CalleePersona] = {}
        
        if self.storage_path.exists():
            self._load_personas()
    
    def add_persona(self, persona: CalleePersona) -> None:
        """Add a persona to the library"""
        self.personas[persona.name] = persona
        self._save_personas()
    
    def get_persona(self, name: str) -> Optional[CalleePersona]:
        """Retrieve a persona by name"""
        return self.personas.get(name)
    
    def _load_personas(self) -> None:
        """Load personas from storage"""
        for file_path in self.storage_path.glob("*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
                persona = CalleePersona.from_dict(data)
                self.personas[persona.name] = persona
    
    def _save_personas(self) -> None:
        """Save personas to storage"""
        self.storage_path.mkdir(exist_ok=True)
        for name, persona in self.personas.items():
            file_path = self.storage_path / f"{name.lower().replace(' ', '_')}.json"
            with open(file_path, "w") as f:
                json.dump(persona.to_dict(), f, indent=2)