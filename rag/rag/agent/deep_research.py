import asyncio
import json
from typing import AsyncGenerator

from sse_starlette.sse import ServerSentEvent

from rag.agent.base import RAGAgent
from rag.llm.base import BaseLLM
from rag.embedding.base import BaseEmbedding
from rag.vector_db.base import BaseVectorDB, RetrievalResult, deduplicate
from rag.agent.prompts import SUB_QUERY_PROMPT, REFLECTION_PROMPT


class DeepResearch(RAGAgent):
    def __init__(
        self,
        llm: BaseLLM,
        vector_db_client: BaseVectorDB,
        embedding_model: BaseEmbedding = None,
        multi_modal_embedding_model: BaseEmbedding = None,
        **kwargs,
    ):
        super().__init__(
            llm=llm,
            vector_db_client=vector_db_client,
            embedding_model=embedding_model,
            multi_modal_embedding_model=multi_modal_embedding_model,
            **kwargs
        )

    async def _generate_sub_queries(self, query: str):
        chat_response = await self.llm.chat(
            messages=[
                {
                    "role": "user",
                    "content": SUB_QUERY_PROMPT.format(original_query=query)
                }
            ]
        )
        response_content = chat_response.content

        return self.llm.literal_eval(response_content)

    async def _generate_sub_gap_queries(self, query, all_sub_queries, all_search_results):
        reflection_prompt = REFLECTION_PROMPT.format(
            original_query=query,
            sub_queries=all_sub_queries,
        )

        images = []
        for result in all_search_results:
            if result and result.payload and "image" in result.payload:
                image = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{result.payload['image']}"
                    },
                }
                images.append(image)

        messages = [
            {
                "role": "system",
                "content": "Determine whether additional search queries are needed based on the original query, list sub-queries and the provided image or images.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": reflection_prompt},
                ]
                + images,
            },
        ]

        chat_response = await self.llm.chat(
            messages=messages
        )
        response_content = chat_response.content
        
        print(response_content)

        return self.llm.literal_eval(response_content)

    async def _search_image_from_vector_db(
        self, 
        query: str,
        collection_name: str
    ) -> list[RetrievalResult]:
        if self.multi_modal_embedding_model is None:
            return []

        query_vector = await self.multi_modal_embedding_model.embed_queries.remote.aio([query])
        results = await self.vector_db_client.search_collection(
            collection=collection_name,
            query_vector=query_vector,
            count=3
        )
        return results

    # async def _search_text_from_vector_db(self, query: str) -> list[RetrievalResult]:
    #     if self.embedding_model is None:
    #         return []

    #     query_vector = await self.embedding_model.embed_queries.remote.aio([query])
    #     results = await self.vector_db_client.search_collection(
    #         collection_name="test",
    #         query_vector=query_vector,
    #         count=3
    #     )
    #     return results

    # async def _search_chunk_from_vector_db(self, query: str) -> list[RetrievalResult]:
    #     final_results = []
    #     tasks = [
    #         self._search_image_from_vector_db(query),
    #         self._search_text_from_vector_db(query)
    #     ]
    #     results = await asyncio.gather(*tasks)
    #     for result in results:
    #         final_results.extend(result)
    #     return final_results

    async def retrieve(
        self, 
        query: str, 
        collection_name: str, 
        **kwargs
    ):
        max_iter = kwargs.get("max_iter", 3)
        all_search_results = []
        all_sub_queries = []

        sub_queries = await self._generate_sub_queries(query)
        if not sub_queries or len(sub_queries) == 0:
            print("<think> No sub-queries generated. Stop the search.</think>\n")
            return [], []

        print(
            f"<think> Break down the original query into {len(sub_queries)} sub-queries: {sub_queries}</think>\n")
        sub_gap_queries = sub_queries

        all_sub_queries.extend(sub_queries)
        for i in range(max_iter):
            print(f">> Iteration {i+1}\n")
            search_result_from_vector_db = []
            search_result_from_internet = []  # TODO: implement internet search

            search_tasks = [
                self._search_image_from_vector_db(sub_query, collection_name)
                for sub_query in sub_gap_queries
            ]

            search_results = await asyncio.gather(*search_tasks)
            for search_result in search_results:
                search_result_from_vector_db.extend(search_result)
                
            search_result_from_vector_db = deduplicate(search_result_from_vector_db)
            all_search_results.extend(search_result_from_vector_db)

            if i == max_iter - 1:
                print(
                    "<think> Exceeded the maximum number of iterations. Stop the search.</think>\n")
                break

            print(
                f"<think> Reflecting on the search results and generating new sub-queries...</think>\n")

            sub_gap_queries = await self._generate_sub_gap_queries(
                query, all_sub_queries, all_search_results)
            if not sub_gap_queries or len(sub_gap_queries) == 0:
                print(
                    "<think> No further research is required. Stop the search.</think>\n")
                break

            print(
                f"<think> Generate {len(sub_gap_queries)} new sub-queries: {sub_gap_queries}</think>\n")
            all_sub_queries.extend(sub_gap_queries)

        all_search_results = deduplicate(all_search_results)

        return all_search_results, all_sub_queries

    async def augment(self, query: str, all_search_results: list[RetrievalResult], **kwargs):
        images = []
        for result in all_search_results:
            if result and result.payload and "image" in result.payload:
                image = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{result.payload['image']}"
                    },
                }
                images.append(image)

        payload = [
            {
                "role": "system",
                "content": "Your task is to answer to the user question based on the provided image or images.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                ]
                + images,
            },
        ]

        return payload

    async def generate(self, augmented, **kwargs):
        async for completion in self.llm.stream_chat(messages=augmented):
            yield completion

    async def process(
        self, 
        query: str,
        collection_name: str, 
        **kwargs
    ) -> AsyncGenerator[ServerSentEvent, None]:
        
        all_search_results, all_sub_queries = await self.retrieve(query, collection_name, **kwargs)

        yield ServerSentEvent(
            data=json.dumps(
                {
                    "results": [
                        {
                            "score": result.score,
                            "image": result.payload["image"],
                            "page": result.payload["page"],
                            "name": result.payload["name"],
                        }
                        for result in all_search_results
                    ]
                }
            ),
            event="sources",
        )

        augmented = await self.augment(query, all_search_results)
        async for completion in self.generate(augmented):
            yield completion
