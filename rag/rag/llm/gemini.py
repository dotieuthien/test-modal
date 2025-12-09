import os
import json
from typing import Dict, List, AsyncGenerator, Optional

from rag.llm.base import BaseLLM, ChatResponse
from rag.llm.types import Message


class GeminiLLM(BaseLLM):
    def __init__(self, client: Optional[object] = None, model: str = "gemini-2.5-flash", **kwargs):
        from google import genai

        if client:
            self.client = client
        else:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            self.client = genai.Client(api_key=api_key)
            
        self.model = model

    def _convert_messages(self, messages: List[Message]) -> List[Dict]:
        gemini_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "assistant":
                role = "model"
            elif role == "system":
                role = "user" 
            
            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text")})
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image/"):
                                # Extract mime_type and base64 data
                                # Format: data:image/jpeg;base64,......
                                try:
                                    header, base64_data = image_url.split(",", 1)
                                    mime_type = header.split(":")[1].split(";")[0]
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": mime_type,
                                            "data": base64_data
                                        }
                                    })
                                except Exception as e:
                                    print(f"Error parsing image data URI: {e}")
                            else:
                                # Handle standard URLs if supported or warn
                                # google-genai might support file_uri but usually requires upload first for public URLs
                                # For now, we assume data URIs as per DeepResearch implementation
                                pass
            
            gemini_messages.append({"role": role, "parts": parts})
        return gemini_messages

    async def chat(self, messages: List[Message], **kwargs) -> ChatResponse:
        contents = self._convert_messages(messages)
        
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents
        )
        
        return ChatResponse(
            content=response.text,
            total_tokens=response.usage_metadata.total_token_count if response.usage_metadata else 0,
        )

    async def stream_chat(self, messages: List[Message], **kwargs) -> AsyncGenerator[str, None]:
        contents = self._convert_messages(messages)
        
        # Note: generate_content_stream is the method for streaming
        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents
        ):
            if chunk.text:
                yield chunk.text
