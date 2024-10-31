from typing import List, Dict, Optional, Callable
from llm_testing.core.data_types import (
    TestCase,
    ConversationContext,
    TestExpectation
)

class TestCaseBuilder:
    def __init__(self):
        self.test_case = TestCase(
            name="",
            description="",
            context=ConversationContext(system_prompt=""),
            user_input="",
            expectations=TestExpectation()
        )
    
    def with_name(self, name: str) -> 'TestCaseBuilder':
        self.test_case.name = name
        return self

    def with_description(self, description: str) -> 'TestCaseBuilder':
        self.test_case.description = description
        return self
    
    def with_system_prompt(self, prompt: str) -> 'TestCaseBuilder':
        self.test_case.context.system_prompt = prompt
        return self
    
    def with_conversation_history(self, history: List[Dict[str, str]]) -> 'TestCaseBuilder':
        self.test_case.context.conversation_history = history
        return self
    
    def with_user_input(self, user_input: str) -> 'TestCaseBuilder':
        self.test_case.user_input = user_input
        return self
    
    def expect_phrases(self, phrases: List[str]) -> 'TestCaseBuilder':
        self.test_case.expectations.contains_phrases.extend(phrases)
        return self
    
    def exclude_phrases(self, phrases: List[str]) -> 'TestCaseBuilder':
        self.test_case.expectations.excludes_phrases.extend(phrases)
        return self
    
    def with_length_constraints(self, min_length: Optional[int] = None, 
                              max_length: Optional[int] = None) -> 'TestCaseBuilder':
        self.test_case.expectations.min_length = min_length
        self.test_case.expectations.max_length = max_length
        return self
    
    def with_custom_validator(self, validator: Callable[[str], bool]) -> 'TestCaseBuilder':
        self.test_case.expectations.custom_validators.append(validator)
        return self
    
    def build(self) -> TestCase:
        return self.test_case
