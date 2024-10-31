# llm_testing/runners/test_runner.py (updated)
from llm_testing.core.data_types import ConversationContext
from llm_testing.core.goals import ConversationGoal
from llm_testing.core.interfaces import LLMInterface
from llm_testing.core.personas import Persona
from llm_testing.core.evaluator import ConversationEvaluator
from typing import List, Dict, Any
import time

class GoalBasedTestRunner:
    def __init__(self, 
                 llm: LLMInterface,
                 evaluator: ConversationEvaluator):
        self.llm = llm
        self.evaluator = evaluator
        self.conversation_history: List[Dict[str, str]] = []
    
    def _generate_user_response(self, persona: Persona) -> str:
        """Generate user response based on persona and conversation history"""
        # Create a system prompt for the user simulator
        system_prompt = f"""You are simulating a {persona.description}.
Your mood is {persona.mood.value} and your communication style is {persona.response_style.value}.
You have the following constraints: {persona.constraints.to_dict()}
You should respond as this persona would, maintaining consistent behavior and knowledge.

Key traits to embody:
{' - ' + chr(10).join(persona.traits)}

Background information:
{persona.background_info}

Remember:
1. Stay in character
2. Respect the persona's constraints
3. Reflect the specified mood and communication style
4. Keep responses natural and conversational
"""
        
        # Create context with recent conversation history
        context = ConversationContext(
            system_prompt=system_prompt,
            conversation_history=self.conversation_history[-4:] if self.conversation_history else []
        )
        
        # Generate response using the same LLM
        response = self.llm.generate_response(
            context,
            "Generate the next user response as this persona. Respond in character, don't explain or add notes."
        )
        
        # Apply any response delays specified in constraints
        if persona.constraints.response_delay_ms:
            time.sleep(persona.constraints.response_delay_ms / 1000)
        
        return response

    def _should_end_conversation(self, last_response: str) -> bool:
        """Determine if the conversation should end"""
        end_indicators = [
            "goodbye",
            "thank you, goodbye",
            "have a nice day",
            "bye",
            "end of conversation",
            "*hangs up*"
        ]
        return any(indicator in last_response.lower() for indicator in end_indicators)

    def run_conversation_test(self,
                            goal: ConversationGoal,
                            persona: Persona,
                            max_turns: int = 10) -> ConversationEvaluation:
        self.conversation_history = []
        
        # Start with user inquiry
        initial_user_message = "Hi, I'd like to book a room"
        self.conversation_history.append({
            "speaker": "user",
            "text": initial_user_message
        })
        
        for _ in range(max_turns):
            # Get latest message
            last_message = self.conversation_history[-1]["text"]
            
            # If last message was from user, generate agent response
            if self.conversation_history[-1]["speaker"] == "user":
                context = ConversationContext(
                    system_prompt=self._create_system_prompt(goal, persona),
                    conversation_history=self.conversation_history
                )
                
                response = self.llm.generate_response(context, last_message)
                
                self.conversation_history.append({
                    "speaker": "agent",
                    "text": response
                })
            
            # If last message was from agent, generate user response
            else:
                user_response = self._generate_user_response(persona)
                self.conversation_history.append({
                    "speaker": "user",
                    "text": user_response
                })
                
                if self._should_end_conversation(user_response):
                    break
        
        return self.evaluator.evaluate(
            self.conversation_history,
            goal,
            persona
        )
