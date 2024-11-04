from typing import List, Dict, Any
import openai
from core.interfaces import LLMInterface
from core.data_types import ConversationContext

class OpenAIProvider(LLMInterface):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.model = model
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

    def generate_response(self, context: ConversationContext, user_input: str) -> str:
        # Format conversation history
        messages = [{"role": "system", "content": context.system_prompt}]
        
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
            )
            
            return chat_completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")