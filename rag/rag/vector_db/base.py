from abc import ABC, abstractmethod
import numpy as np
from typing import Any, List


class RetrievalResult:
    def __init__(self, id: int, score: float, payload: dict[str, Any]):
        self.id = id
        self.score = score
        self.payload = payload
        

def deduplicate(results: List[RetrievalResult]) -> List[RetrievalResult]:
    result_set = set()
    deduplicated_results = []
    for result in results:
        if result.payload["page"] not in result_set:
            result_set.add(result.payload["page"])
            deduplicated_results.append(result)
            
    return list(deduplicated_results)


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
    