import os
import json
from typing import Dict, List, AsyncGenerator, Optional

from rag.llm.base import BaseLLM, ChatResponse
from rag.llm.types import Message


class OpenAILLM(BaseLLM):
    def __init__(self, client: Optional[object] = None, model: str = "gpt-4o-mini", **kwargs):
        from openai import AsyncOpenAI

        if client:
            self.client = client
        else:
            self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    async def chat(self, messages: List[Message]) -> ChatResponse:
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return ChatResponse(
            content=completion.choices[0].message.content,
            total_tokens=completion.usage.total_tokens,
        )

    async def stream_chat(self, messages: List[Message]) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
