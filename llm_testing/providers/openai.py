from typing import Any, Dict, List, Optional
import openai
from core.interfaces import LLMInterface
from core.data_types import ConversationContext, LLMResponse

class OpenAIProvider(LLMInterface):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.model = model
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

    def generate_response(self, context: ConversationContext, user_input: str, tools: Optional[List[Dict[str, Any]]] = None) -> str:
        # Add conversation history
        messages = [{"role": "system", "content": context.system_prompt}]
        for msg in context.conversation_history:
            messages.append({
                "role": "assistant" if msg["speaker"] == "agent" else "user",
                "content": msg["text"]
            })

        if user_input:
            messages.append({"role": "user", "content": user_input})
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                tools=tools
            )
            
            response_msg = chat_completion.choices[0].message
            return LLMResponse(response_msg.content, response_msg.tool_calls)
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")