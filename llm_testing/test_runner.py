import json
from core.data_types import ConversationContext, EntitySpeaking, LLMResponse
from core.goals import AgentTaskConfig
from core.interfaces import LLMInterface
from core.personas import CalleePersona
from core.evaluator import ConversationEvaluation, ConversationEvaluator
from typing import List, Dict, Any, Optional

class GoalBasedTestRunner:
    def __init__(self, 
                 llm: LLMInterface,
                 evaluator: ConversationEvaluator):
        self.llm = llm
        self.evaluator = evaluator
        self.conversation_history: List[Dict[str, str]] = []
    
    def _generate_callee_response(self, persona: CalleePersona, agent_tools: Optional[List[Dict[str, Any]]] = []) -> LLMResponse:
        """Generate user response based on persona and conversation history"""
        # Create a system prompt for the user simulator
        system_prompt = f"""You are simulating a {persona.description[:1].lower() + persona.description[1:] if persona.description else 'person'}
Your mood is {persona.mood} and your communication style is {persona.response_style}.
You have the following additional context: {persona.additional_context}
You should respond as this persona would, maintaining consistent behavior and knowledge.

Key traits to embody:
{' - ' + chr(10).join(persona.traits)}

Background information:
{persona.background_info}

Remember:
1. Stay in character
2. Use the persona's additional context when generating responses
3. Reflect the specified mood and communication style
4. Keep responses natural and conversational
"""
        
        # Create context with recent conversation history
        context = ConversationContext(
            system_prompt=system_prompt,
            conversation_history=self.conversation_history[-4:] if self.conversation_history else []
        )
        
        response = self.llm.generate_response_with_conversation_history(context,
                                                            EntitySpeaking.CALLEE,
                                                            tools=agent_tools,
                                                            user_input="Generate the next user response as this persona. Respond in character, don't explain or add notes.")
        

        
        return response

    def print_last_msg(self, turn_count: int, persona: Optional[CalleePersona] = None):
        last_message = self.conversation_history[-1]
        if last_message["speaker"] == EntitySpeaking.CALLEE.value:
            print(f"[{turn_count}] {' '.join(persona.role.split('_')).title() if persona else 'Callee'}: {last_message['text']}")
        else:
            print(f"[{turn_count}] Voice Agent: {last_message['text']}")

    def run_conversation_test(self,
                            task_config: AgentTaskConfig,
                            persona: CalleePersona,
                            max_turns: int = 999) -> ConversationEvaluation:
        self.conversation_history = []
        
        self.conversation_history.append({
            "speaker": EntitySpeaking.CALLEE.value,
            "text": persona.initial_message
        })
        
        self.print_last_msg(0, persona)
        turn_count = 1
        while turn_count < max_turns:
            # If last message was from callee, generate agent response
            if self.conversation_history[-1]["speaker"] == EntitySpeaking.CALLEE.value:
                context = ConversationContext(
                    system_prompt=task_config.generate_system_prompt(),
                    conversation_history=self.conversation_history
                )
                
                response = self.llm.generate_response_with_conversation_history(context, EntitySpeaking.VOICE_AGENT,
                                                                                 task_config.tool_calls)
                
                self.conversation_history.append({
                    "speaker": EntitySpeaking.VOICE_AGENT.value,
                    "text": response.response_content
                })            
            # If last message was from the agent, generate callee response
            else:
                response = self._generate_callee_response(persona, task_config.tool_calls)
                self.conversation_history.append({
                    "speaker": EntitySpeaking.CALLEE.value,
                    "text": response.response_content
                })
            
            if response.tools_called and response.tools_called[0].function.name == "end_conversation":
                # remove last message from conversation history as it's None due to the tool call
                self.conversation_history.pop()

                arguments = json.loads(response.tools_called[0].function.arguments)
                reason = arguments.get('reason')
                who_ended_conversation = arguments.get('who_ended_conversation')
                termination_evidence = arguments.get('termination_evidence')
                print(f"\n*** Conversation ended by {who_ended_conversation}. Reason: {reason}. Evidence: {termination_evidence}")
                break
            else:
                self.print_last_msg(turn_count, persona)

            turn_count += 1
        
        if turn_count >= max_turns:
            print(f"Warning: Conversation ended prematurely due to max turn limit of {max_turns}")
        
        return self.evaluator.evaluate(
            self.conversation_history,
            task_config,
            persona
        )
