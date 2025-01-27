from typing import Any, Dict, List, Optional
import openai
from pydantic import BaseModel
from ..interfaces import LLMInterface
from ..data_types import ConversationContext, EntitySpeaking, LLMResponse


class OpenAIProvider(LLMInterface):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.model = model
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

    def plain_call(self, system_prompt: str, messages: List[Dict[str, Any]]) -> LLMResponse:
        return self.generate_response([{"role": "system", "content": system_prompt}] + messages)
    
    def generate_response_with_conversation_history(self, context: ConversationContext,
                                                     entity_speaking: EntitySpeaking,
                                                     tools: Optional[List[Dict[str, Any]]] = None,
                                                     user_input: str = None) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": context.system_prompt}]
        for msg in context.conversation_history:
            role = "assistant" if msg["speaker"] == entity_speaking.value else "user"
            messages.append({"role": role, "content": msg["text"]})

        if user_input:
            messages.append({"role": "user", "content": user_input})

        return self.generate_response(messages, tools)

    def generate_response_with_structured_output(self, messages: List[Dict[str, Any]], response_format: BaseModel):
        try:
            chat_completion = self.client.beta.chat.completions.parse(
                messages=messages,
                model=self.model,
                response_format=response_format,
            )

            return chat_completion.choices[0].message.parsed
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
        
    def generate_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:        
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
            