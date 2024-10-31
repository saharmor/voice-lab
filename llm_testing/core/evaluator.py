import json
from typing import List, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from core.goals import ConversationGoal, Persona
from core.interfaces import LLMInterface
from core.data_types import ConversationContext

@dataclass
class ConversationEvaluation:
    """Result of evaluating a conversation against its goals"""
    success: bool
    goal_achieved: bool
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConversationEvaluator(ABC):
    """Abstract base class for conversation evaluators"""
    @abstractmethod
    def evaluate(self, 
                conversation_history: List[Dict[str, str]],
                goal: ConversationGoal,
                persona: Persona) -> ConversationEvaluation:
        pass

class LLMConversationEvaluator(ConversationEvaluator):
    """Evaluates conversation success using an LLM"""
    def __init__(self, evaluation_llm: LLMInterface):
        self.llm = evaluation_llm
    
    def evaluate(self, 
                conversation_history: List[Dict[str, str]],
                goal: ConversationGoal,
                persona: Persona) -> ConversationEvaluation:
        # Create the evaluation prompt
        prompt = self._create_evaluation_prompt(
            conversation_history, goal, persona)
        
        # Get LLM evaluation
        evaluation_response = self.llm.generate_response(
            ConversationContext(system_prompt=self._get_evaluator_system_prompt()),
            prompt
        )
        
        # Parse evaluation response
        return self._parse_evaluation_response(evaluation_response)
    
    def _get_evaluator_system_prompt(self) -> str:
        return """You are an objective conversation evaluator. 
        Your task is to determine if a conversation achieved its stated goal 
        while respecting the given constraints and persona characteristics.
        Provide your evaluation in JSON format with the following structure:
        {
            "success": boolean,
            "goal_achieved": boolean,
            "reasoning": "detailed explanation of your evaluation",
            "metadata": {
                "key_observations": [],
                "missed_objectives": [],
                "persona_adherence": float
            }
        }"""
    
    def _create_evaluation_prompt(self, 
                                conversation_history: List[Dict[str, str]],
                                goal: ConversationGoal,
                                persona: Persona) -> str:
        return f"""Please evaluate the following conversation:

Goal: {goal.description}
Success Criteria: {json.dumps(goal.success_criteria, indent=2)}

Persona: {persona.name}
Persona Description: {persona.description}
Persona Constraints: {json.dumps(persona.constraints, indent=2)}

Conversation:
{self._format_conversation(conversation_history)}

Evaluate if the conversation achieved its goal while maintaining persona consistency.
"""

    def _format_conversation(self, history: List[Dict[str, str]]) -> str:
        return "\n".join([
            f"{turn['speaker']}: {turn['text']}" for turn in history
        ])
    
    def _parse_evaluation_response(self, response: str) -> ConversationEvaluation:
        try:
            evaluation_data = json.loads(response)
            return ConversationEvaluation(
                success=evaluation_data["success"],
                goal_achieved=evaluation_data["goal_achieved"],
                reasoning=evaluation_data["reasoning"],
                metadata=evaluation_data.get("metadata", {})
            )
        except json.JSONDecodeError:
            return ConversationEvaluation(
                success=False,
                goal_achieved=False,
                reasoning="Failed to parse evaluation response",
                metadata={"error": "Invalid evaluation format"}
            )

# Update test_runner.py to include goal-based testing
class GoalBasedTestRunner:
    def __init__(self, 
                 llm: LLMInterface,
                 evaluator: ConversationEvaluator):
        self.llm = llm
        self.evaluator = evaluator
        self.conversation_history: List[Dict[str, str]] = []
    
    def run_conversation_test(self,
                            goal: ConversationGoal,
                            persona: Persona,
                            max_turns: int = 10) -> ConversationEvaluation:
        self.conversation_history = []
        
        for _ in range(max_turns):
            # Get last user message if exists
            last_message = (self.conversation_history[-1]["text"] 
                          if self.conversation_history else "")
            
            # Generate agent response
            context = ConversationContext(
                system_prompt=self._create_system_prompt(goal, persona),
                conversation_history=self.conversation_history
            )
            
            response = self.llm.generate_response(context, last_message)
            
            # Add to history
            self.conversation_history.append({
                "speaker": "agent",
                "text": response
            })
            
            # Simulate user response based on persona
            user_response = self._generate_user_response(persona)
            self.conversation_history.append({
                "speaker": "user",
                "text": user_response
            })
            
            # Check if conversation should end
            if self._should_end_conversation(user_response):
                break
        
        # Evaluate conversation
        return self.evaluator.evaluate(
            self.conversation_history,
            goal,
            persona
        )
    
    def _create_system_prompt(self, goal: ConversationGoal, persona: Persona) -> str:
        return f"""You are a voice agent trying to {goal.description}.
You are talking to a {persona.description}.
Focus on achieving the goal while handling the persona's characteristics appropriately."""

# Example usage:
def create_hotel_booking_test():
    goal = ConversationGoal(
        description="Book a hotel room for December 12th-24th",
        success_criteria={
            "booking_dates": {
                "start": "2024-12-12",
                "end": "2024-12-24"
            },
            "required_confirmations": ["booking_reference", "price"],
            "max_turns": 10
        }
    )
    
    persona = Persona(
        name="Angry Receptionist",
        description="An irritable hotel receptionist who is having a bad day",
        traits=["impatient", "curt", "easily annoyed"],
        constraints={
            "available_dates": {
                "start": "2024-12-12",
                "end": "2024-12-14"
            },
            "mood": "angry",
            "response_style": "curt"
        }
    )
    
    return goal, persona
