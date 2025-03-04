import modal

import json
from pathlib import Path
from vrag.image import web_image
from vrag.colpali import ColPaliModel
from modal_app import app

static_path = Path(__file__).with_name("frontend").joinpath("dist").resolve()

MODELS_DIR = "/llama_models"
volume = modal.Volume.from_name("llama_models", create_if_missing=True)


colpali = ColPaliModel()

@app.function(
    image=web_image,
    concurrency_limit=1,
    container_idle_timeout=300,
    timeout=600,
    allow_concurrent_inputs=10,
    volumes={MODELS_DIR: volume},
    secrets=[modal.Secret.from_name("openai")],
    mounts=[
       modal.Mount.from_local_python_packages("vrag"),
    ],
)
@modal.asgi_app()
def serve():
    from uuid import UUID
    import uuid
    from pydantic import BaseModel
    from fastapi import FastAPI, UploadFile, File
    from sse_starlette.sse import EventSourceResponse, ServerSentEvent
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    
    from vrag.vrag import VRAG
    from vrag.qdrant_client import InMemoryQdrant


    class SearchRequest(BaseModel):
        query: str
        instance_id: UUID
        count: int = 3


    web_app = FastAPI()

    qdrant = InMemoryQdrant()

    VisionRAG = VRAG(colpali=colpali, qdrant=qdrant)

    @web_app.post("/collections")
    async def create_collection(files: list[UploadFile] = File(...)):
        name = str(uuid.uuid4())
        filenames = []
        byte_files = []

        async def read_files():
            for file in files:
                content = await file.read()
                filenames.append(file.filename or "file has no name")
                byte_files.append((name, file.filename or "file has no name", content))

        await read_files()

        async def event_generator():
            yield ServerSentEvent(
                data=json.dumps({"message": f"Indexing {len(byte_files)} files"})
            )
            for idx, byte_file in enumerate(byte_files):
                yield ServerSentEvent(
                    data=json.dumps(
                        {"message": f"Indexing file {idx + 1} / {len(byte_files)}"}
                    )
                )
                try:
                    async for state in VisionRAG.add_pdf(*byte_file):
                        yield state
                except Exception as e:
                    yield json.dumps({"error": str(e)})
            yield ServerSentEvent(
                data=json.dumps({"id": name, "filenames": filenames}), event="complete"
            )

        return EventSourceResponse(event_generator())

    @web_app.post("/search")
    async def search(query: SearchRequest):
        can_query = await qdrant.does_collection_exist(str(query.instance_id))

        async def event_generator():
            if not can_query:
                yield ServerSentEvent(
                    data=json.dumps(
                        {
                            "message": "The index has been deleted or does not exist. Please re-add the files."
                        }
                    )
                )
                return
            async for stage in VisionRAG.run_vrag(str(query.instance_id), query.query, query.count):
                yield stage

        return EventSourceResponse(event_generator())

    return web_app