from core.data_types import ConversationContext
from core.goals import AgentTaskConfig
from core.interfaces import LLMInterface
from core.personas import CalleePersona
from core.evaluator import ConversationEvaluation, ConversationEvaluator
from typing import List, Dict, Any
import time

class GoalBasedTestRunner:
    def __init__(self, 
                 llm: LLMInterface,
                 evaluator: ConversationEvaluator):
        self.llm = llm
        self.evaluator = evaluator
        self.conversation_history: List[Dict[str, str]] = []
    
    def _generate_callee_response(self, persona: CalleePersona) -> str:
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

    def print_last_msg(self, turn_count: int):
        last_message = self.conversation_history[-1]
        print(f"[{turn_count}] {last_message['speaker']}: {last_message['text']}")

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
                            task_config: AgentTaskConfig,
                            persona: CalleePersona,
                            max_turns: int = 999) -> ConversationEvaluation:
        self.conversation_history = []
        
        self.conversation_history.append({
            "speaker": "callee",
            "text": persona.initial_message
        })
        
        self.print_last_msg(0)
        turn_count = 1
        while turn_count < max_turns:
            # Get latest message
            last_message = self.conversation_history[-1]["text"]
            
            # If last message was from callee, generate agent response
            if self.conversation_history[-1]["speaker"] == "callee":
                context = ConversationContext(
                    system_prompt=task_config.system_prompt,
                    conversation_history=self.conversation_history
                )
                
                response = self.llm.generate_response(context, last_message)
                
                self.conversation_history.append({
                    "speaker": "agent",
                    "text": response.response_content
                })            
            # If last message was from agent, generate callee response
            else:
                response = self._generate_callee_response(persona)
                self.conversation_history.append({
                    "speaker": "callee",
                    "text": response.response_content
                })
            
            # print last message from conversation history
            self.print_last_msg(turn_count)
            turn_count += 1
            
            if response.end_status:
                print(f"Conversation ended by {response.end_status.who_ended}. Reason: {response.end_status.reason}")
                break

        if turn_count >= max_turns:
            print(f"Warning: Conversation ended prematurely due to max turn limit of {max_turns}")
        
        return self.evaluator.evaluate(
            self.conversation_history,
            task_config
        )
