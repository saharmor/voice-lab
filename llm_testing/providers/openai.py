import json
import openai
from core.interfaces import LLMInterface
from core.data_types import ConversationContext, ConversationEndStatus, LLMResponse

class OpenAIProvider(LLMInterface):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.model = model
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

    def generate_response(self, context: ConversationContext, user_input: str) -> str:
        # Add conversation history
        messages = [{"role": "system", "content": context.system_prompt}]
        for msg in context.conversation_history:
            messages.append({
                "role": "assistant" if msg["speaker"] == "agent" else "user",
                "content": msg["text"]
            })

        if user_input:
            messages.append({"role": "user", "content": user_input})

        # TODO move this to the task level
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "end_conversation",
                    "description": """Call ONLY when conversation reaches clear end state by both sides exchanging farewell messages or one side explicitly stating they want to end the conversation.

                    DO NOT CALL if:
                    - Still negotiating/discussing
                    - Questions pending
                    - No explicit end statement
                    - Just discussing options

                    Must have clear evidence in final messages.
            """,
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "The specific reason why the conversation must end, which should directly reference one of the conditions listed above",
                                "enum": [
                                    "explicit_termination_request",
                                    "service_not_available",
                                    "price_agreement_not_reached",
                                    "customer_declined_service",
                                    "provider_declined_service"
                                ]
                            },
                            "who_ended_conversation": {
                                "type": "string",
                                "enum": ["agent", "callee"],
                                "description": "Who initiated the conversation ending. Must be supported by clear evidence in the conversation."
                            },
                            "termination_evidence": {
                                "type": "object",
                                "properties": {
                                    "final_messages": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        },
                                        "description": "Last 1-2 messages showing clear end reason"
                                    },
                                    "termination_type": {
                                        "type": "string",
                                        "enum": ["successful_completion", "early_termination"],
                                        "description": "Whether successful completion or early termination"
                                    }
                                },
                                "required": ["final_messages", "termination_type"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["reason", "who_ended_conversation", "termination_evidence"],
                        "additionalProperties": False
                    }
                }
            }
        ]
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                tools=tools
            )
            
            response_mesg = chat_completion.choices[0].message
            if response_mesg.tool_calls:
                arguments = json.loads(response_mesg.tool_calls[0].function.arguments)
                reason = arguments.get('reason')
                who_ended_conversation = arguments.get('who_ended_conversation')
                termination_evidence = arguments.get('termination_evidence')
                return LLMResponse(chat_completion.choices[0].message.content, ConversationEndStatus(reason, who_ended_conversation, termination_evidence))
            else:
                return LLMResponse(chat_completion.choices[0].message.content, None)
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")