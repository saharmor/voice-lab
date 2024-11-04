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
        messages = [{"role": "system", "content": context.system_prompt}]
        
        # TODO move this to the agent level
        tools = [
            {
            "name": "should_end_conversation",
            "description": "Determines if the conversation should end based on the context and message history",
            "parameters": {
                "type": "object",
                "properties": {
                    "should_end": {
                        "type": "boolean",
                        "description": "Whether the conversation should end"
                    },
                    "reason": {
                        "type": "string",
                        "description": "The reason why the conversation should end"
                    },
                    "who_ended_conversation": {
                        "type": "string",
                        "description": "Who ended the conversation, the agent or the callee"
                    }
                    }, 
                    "required": ["should_end", "reason", "who_ended_conversation"]
                }
            }
        ]

        # Add conversation history
        for msg in context.conversation_history:
            messages.append({
                "role": "assistant" if msg["speaker"] == "agent" else "user",
                "content": msg["text"]
            })
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                tools=tools
            )
            
            response_mesg = chat_completion.choices[0].message
            tool_call = response_mesg.tool_calls[0]
            if tool_call:
                arguments = json.loads(tool_call['function']['arguments'])
                should_end = arguments.get('should_end')
                reason = arguments.get('reason')
                who_ended_conversation = arguments.get('who_ended_conversation')
            else:
                should_end = False
                reason = None
                who_ended_conversation = None

            return LLMResponse(chat_completion.choices[0].message.content, ConversationEndStatus(should_end, reason, who_ended_conversation))
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")