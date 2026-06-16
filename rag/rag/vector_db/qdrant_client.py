import uuid
import os
import traceback

from qdrant_client import AsyncQdrantClient, models
from rag.vector_db.base import BaseVectorDB, RetrievalResult


class InMemoryQdrant(BaseVectorDB):
    def __init__(self, timeout: int = None) -> None:
        self.timeout = timeout or int(os.environ.get("QDRANT_TIMEOUT", "300"))  # Default 5 minutes
        self.client = AsyncQdrantClient(":memory:", timeout=self.timeout)

    async def does_collection_exist(self, name: str) -> bool:
        return await self.client.collection_exists(name)

    async def create_collection(self, name: str):
        exists = await self.client.collection_exists(name)

        if exists:
            return

        return await self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )

    async def search_collection(
        self, collection: str, query_vector: list[list[float]], count: int
    ) -> list[RetrievalResult]:
        try:
            # Check if collection exists first
            exists = await self.client.collection_exists(collection)
            if not exists:
                print(f"Collection '{collection}' does not exist. Returning empty results.")
                return []

            print(f"Querying collection '{collection}' with vector shape: {len(query_vector)}x{len(query_vector[0]) if query_vector else 0}, limit: {count}")
            result = await self.client.query_points(
                collection, query=query_vector, limit=count
            )
            return [
                RetrievalResult(
                    id=point.id,
                    score=point.score,
                    payload=point.payload or {}
                )
                for point in result.points
            ]
        except Exception as e:
            print(f"Error searching collection '{collection}': {type(e).__name__}: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            print(f"Query vector shape: {len(query_vector)}x{len(query_vector[0]) if query_vector else 0}")
            # Check if there's __cause__ or __context__
            if hasattr(e, '__cause__') and e.__cause__:
                print(f"Cause: {e.__cause__}")
            if hasattr(e, '__context__') and e.__context__:
                print(f"Context: {e.__context__}")
            return []

    async def upsert_points(
        self,
        collection: str,
        pdf_name: str,
        embeddings: list[list[list[float]]],
        encoded_images: list[str],
    ) -> models.UpdateResult:
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "page": idx + 1,
                    "name": pdf_name,
                    "image": encoded_images[idx],
                },
            )
            for idx, embedding in enumerate(embeddings)
        ]
        return await self.client.upsert(collection, points)


class QdrantCloudClient(BaseVectorDB):
    def __init__(self, url: str = None, port: int = None, api_key: str = None, timeout: int = None) -> None:
        self.url = url or os.environ.get("QDRANT_URL")
        self.port = port or os.environ.get("QDRANT_PORT")
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY")
        self.timeout = timeout or int(os.environ.get("QDRANT_TIMEOUT", "300"))  # Default 5 minutes

        if not self.url:
            raise ValueError("QDRANT_URL environment variable or url argument not set")

        if self.port:
            self.port = int(self.port)

        self.client = AsyncQdrantClient(
            url=self.url,
            port=self.port,
            api_key=self.api_key,
            timeout=self.timeout
        )

    async def does_collection_exist(self, name: str) -> bool:
        return await self.client.collection_exists(name)

    async def create_collection(self, name: str):
        exists = await self.client.collection_exists(name)

        if exists:
            return

        return await self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )

    async def search_collection(
        self, collection: str, query_vector: list[list[float]], count: int
    ) -> list[RetrievalResult]:
        try:
            # Check if collection exists first
            exists = await self.client.collection_exists(collection)
            if not exists:
                print(f"Collection '{collection}' does not exist. Returning empty results.")
                return []

            print(f"Querying collection '{collection}' with vector shape: {len(query_vector)}x{len(query_vector[0]) if query_vector else 0}, limit: {count}")
            result = await self.client.query_points(
                collection, query=query_vector, limit=count
            )
            return [
                RetrievalResult(
                    id=point.id,
                    score=point.score,
                    payload=point.payload or {}
                )
                for point in result.points
            ]
        except Exception as e:
            print(f"Error searching collection '{collection}': {type(e).__name__}: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            print(f"Query vector shape: {len(query_vector)}x{len(query_vector[0]) if query_vector else 0}")
            # Check if there's __cause__ or __context__
            if hasattr(e, '__cause__') and e.__cause__:
                print(f"Cause: {e.__cause__}")
            if hasattr(e, '__context__') and e.__context__:
                print(f"Context: {e.__context__}")
            return []

    async def upsert_points(
        self,
        collection: str,
        pdf_name: str,
        embeddings: list[list[list[float]]],
        encoded_images: list[str],
    ) -> models.UpdateResult:
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "page": idx + 1,
                    "name": pdf_name,
                    "image": encoded_images[idx],
                },
            )
            for idx, embedding in enumerate(embeddings)
        ]

        # Batch upsert to avoid payload too large errors
        # Using a small batch size (e.g., 1) because each point contains a base64 image
        batch_size = 1
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self.client.upsert(collection, batch)

        return models.UpdateResult(operation_id=0, status=models.UpdateStatus.COMPLETED)