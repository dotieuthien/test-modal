import os
import json
from typing import Dict, List, AsyncGenerator

from rag.llm.base import BaseLLM, ChatResponse
from sse_starlette.sse import ServerSentEvent


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini", **kwargs):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    async def chat(self, messages: List[Dict]) -> ChatResponse:
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return ChatResponse(
            content=completion.choices[0].message.content,
            total_tokens=completion.usage.total_tokens,
        )

    async def stream_chat(self, messages: List[Dict]) -> AsyncGenerator[ServerSentEvent, None]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield ServerSentEvent(
                    data=json.dumps({"chunk": chunk.choices[0].delta.content})
                )
