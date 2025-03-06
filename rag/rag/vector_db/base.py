from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class RetrievalResult:
    def __init__(self, score: float, payload: dict[str, Any]):
        self.score = score
        self.payload = payload
        

class CollectionInfo:
    def __init__(self, collection_name: str, description: str):
        self.collection_name = collection_name
        self.description = description


class BaseVectorDB(ABC):
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    async def does_collection_exist(self, collection_name: str) -> bool:
        pass
    
    @abstractmethod
    async def create_collection(self, collection_name: str) -> Any:
        pass
    
    @abstractmethod
    async def search_collection(self, collection_name: str, query_vector: Any, count: int) -> Any:
        pass
    
    @abstractmethod
    async def upsert_points(self, collection_name: str, doc_name: str, embeddings: Any, encoded_images: Any) -> Any:
        pass
    