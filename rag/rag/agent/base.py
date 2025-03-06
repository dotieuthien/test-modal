import json
from abc import ABC
from typing import Any, List, Tuple, AsyncGenerator

from sse_starlette.sse import ServerSentEvent

import cv2
import base64

from rag.llm.base import BaseLLM
from rag.embedding.base import BaseEmbedding
from rag.vector_db.base import BaseVectorDB

from rag.tools.pdf_to_image import images_from_pdf_bytes


class BaseAgent(ABC):
    def __init__(
        self, 
        llm: BaseLLM,
        vector_db_client: BaseVectorDB,
        embedding_model: BaseEmbedding = None,
        multi_modal_embedding_model: BaseEmbedding = None,
        **kwargs
    ):
        self.llm = llm
        self.vector_db_client = vector_db_client
        self.embedding_model = embedding_model
        self.multi_modal_embedding_model = multi_modal_embedding_model

    def invoke(self, query: str, **kwargs) -> Any:
        pass
    
    
class RAGAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass
    
    async def add_pdf(self, collection_name: str, name: str, pdf: bytes
                      ) -> AsyncGenerator[ServerSentEvent, None]:
        """
        Add a PDF to the vector database.
        Args:
            collection_name: The name of the collection to add the PDF to.
            name: The name of the PDF.
            pdf: The PDF to add to the vector database.
        Returns:
            A generator of ServerSentEvent objects.
        """
        await self.vector_db_client.create_collection(collection_name)

        embeddings: list[list[list[float]]] = []
        idx = 0
        batch_size = 4
        images = images_from_pdf_bytes(pdf)
        count = len(images)

        yield ServerSentEvent(
            data=json.dumps({"message": f"0 % of {count} pages indexed...\n"})
        )
        
        async for embedding in self.multi_modal_embedding_model.embed_images.remote_gen.aio(
            images, batch_size
        ):
            embeddings.append(embedding)
            if idx < count:
                percent = int(idx / count * 100)
                yield ServerSentEvent(
                    data=json.dumps(
                        {"message": f"{percent} % of {count} pages indexed...\n"}
                    )
                )
            idx += 1

        encoded_images = []

        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.imencode(".jpg", img)[1]
            img = base64.b64encode(img).decode("utf-8")  # type: ignore
            encoded_images.append(img)

        await self.vector_db_client.upsert_points(
            collection_name, name, embeddings, encoded_images
        )
        
    async def retrieve(self, query: str, **kwargs) -> Any:
        pass
    
    async def augment(self, query: str, **kwargs) -> Any:
        pass
    
    async def generate(self, query: str, **kwargs) -> Any:
        pass
    
    async def process(self, query: str, **kwargs) -> Any:
        pass

