from abc import ABC
from typing import Any, List, Tuple, AsyncGenerator

from sse_starlette.sse import ServerSentEvent

from rag.agent.base import RAGAgent
from rag.llm.base import BaseLLM
from rag.embedding.base import BaseEmbedding
from rag.vector_db.base import BaseVectorDB, RetrievalResult


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
    
    async def add_pdf(self, collection_name: str, pdf: bytes
                      ) -> AsyncGenerator[ServerSentEvent, None]:
        self.vec

    async def retrieve(self, query: str, **kwargs) -> Any:
        pass
    
    async def augment(self, query: str, **kwargs) -> Any:
        pass
    
    async def generate(self, query: str, **kwargs) -> Any:
        pass
    
    async def process(self, query: str, **kwargs) -> Any:
        pass

