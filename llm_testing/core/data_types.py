from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime

@dataclass
class ConversationContext:
    """Represents the context of the conversation"""
    system_prompt: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestExpectation:
    """Defines what we expect from the LLM response"""
    contains_phrases: List[str] = field(default_factory=list)
    excludes_phrases: List[str] = field(default_factory=list)
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    custom_validators: List[Callable[[str], bool]] = field(default_factory=list)
    
    def validate(self, response: str) -> List[str]:
        """Validates a response against all expectations"""
        errors = []
        
        for phrase in self.contains_phrases:
            if phrase.lower() not in response.lower():
                errors.append(f"Expected phrase '{phrase}' not found in response")
        
        for phrase in self.excludes_phrases:
            if phrase.lower() in response.lower():
                errors.append(f"Excluded phrase '{phrase}' found in response")
        
        if self.max_length and len(response) > self.max_length:
            errors.append(f"Response length {len(response)} exceeds maximum {self.max_length}")
        if self.min_length and len(response) < self.min_length:
            errors.append(f"Response length {len(response)} below minimum {self.min_length}")
        
        for validator in self.custom_validators:
            if not validator(response):
                errors.append(f"Custom validation failed: {validator.__name__}")
        
        return errors

@dataclass
class TestCase:
    """Represents a single test case for the LLM"""
    name: str
    description: str
    context: ConversationContext
    user_input: str
    expectations: TestExpectation
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestResult:
    """Stores the result of a test execution"""
    test_case: TestCase
    response: str
    passed: bool
    errors: List[str]
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ConversationEndStatus:
    should_end: bool
    reason: Optional[str] = None
    who_ended: Optional[str] = None

    def __post_init__(self):
        if self.who_ended and self.who_ended not in ['callee', 'agent']:
            raise ValueError("who_ended must be either 'callee' or 'agent'")

class LLMResponse:
    def __init__(self, response_content: str, end_status: ConversationEndStatus):
        """
        Args:
            response_content: The actual response text from the LLM
            end_status: The conversation end status
        """
        self.response_content = response_content
        self.end_status = end_status
