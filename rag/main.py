import modal
from pathlib import Path
import json
import asyncio
from rag.image import web_image
from rag.embedding.colpali import ColPaliModel
from modal_app import app


colpali = ColPaliModel()

seed_data_dir = Path(__file__).parent / "seed_data"
seed_data_remote_dir = Path("/root/seed_data")
seed_data_mount = modal.Mount.from_local_dir(
    seed_data_dir,
    remote_path=seed_data_remote_dir,
)

MINUTES = 60

@app.function(
    image=web_image,
    concurrency_limit=1,
    container_idle_timeout=600,
    timeout=60 * MINUTES,
    allow_concurrent_inputs=10,
    secrets=[
        modal.Secret.from_name("openai"),
        modal.Secret.from_name("googlecloud-secret"),
        modal.Secret.from_name("qdrant-secret"),
    ],
    mounts=[
        modal.Mount.from_local_python_packages("rag"),
        seed_data_mount,
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

    from rag.vector_db.qdrant_client import InMemoryQdrant, QdrantCloudClient
    from rag.agent.deep_research import DeepResearch
    # from rag.agent.vision_rag import VRAG
    from rag.llm.openai import OpenAILLM
    from rag.llm.gemini import GeminiLLM

    import gradio as gr
    import time

    class SearchRequest(BaseModel):
        query: str
        instance_id: UUID
        count: int = 3

    web_app = FastAPI()

    import os
    if os.environ.get("QDRANT_URL"):
        qdrant = QdrantCloudClient()
    else:
        qdrant = InMemoryQdrant()

    deep_research = DeepResearch(
        llm=GeminiLLM(),
        vector_db_client=qdrant,
        multi_modal_embedding_model=colpali
    )

    @web_app.post("/collections")
    async def create_collection(files: list[UploadFile] = File(...)):
        name = str(uuid.uuid4())
        filenames = []
        byte_files = []

        async def read_files():
            for file in files:
                content = await file.read()
                filenames.append(file.filename or "file has no name")
                byte_files.append(
                    (name, file.filename or "file has no name", content))

        await read_files()

        async def event_generator():
            yield ServerSentEvent(
                data=json.dumps(
                    {"message": f"Indexing {len(byte_files)} files"})
            )
            for idx, byte_file in enumerate(byte_files):
                yield ServerSentEvent(
                    data=json.dumps(
                        {"message":
                            f"Indexing file {idx + 1} / {len(byte_files)}"}
                    )
                )
                try:
                    async for state in deep_research.add_pdf(*byte_file):
                        yield ServerSentEvent(data=json.dumps(state))
                except Exception as e:
                    yield ServerSentEvent(data=json.dumps({"error": str(e)}))
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
            async for stage in deep_research.process(str(query.instance_id), query.query, query.count):
                if "event" in stage:
                    yield ServerSentEvent(
                        data=json.dumps(stage["data"]),
                        event=stage["event"]
                    )
                else:
                    yield ServerSentEvent(data=json.dumps(stage))

        return EventSourceResponse(event_generator())

    async def upload_pdf(pdf_file):
        if pdf_file is None:
            return "Please select a PDF file to upload", None

        collection_id = "demo_collection"
        # Gradio provides the file path directly
        with open(pdf_file.name, 'rb') as f:
            pdf_bytes = f.read()
        filename = Path(pdf_file.name).name

        messages = ["Starting to index PDF..."]
        async for state in deep_research.add_pdf(collection_id, filename, pdf_bytes):
            if "message" in state:
                messages.append(state["message"])

        messages.append(f"PDF indexed successfully!")
        return "\n".join(messages), collection_id

        # try:
        #     async for state in deep_research.add_pdf(collection_id, filename, pdf_bytes):
        #         if "message" in state:
        #             messages.append(state["message"])

        #     messages.append(f"PDF indexed successfully!")
        #     return "\n".join(messages), collection_id
        # except Exception as e:
        #     return f"Error indexing PDF: {str(e)}", None

    async def process_query(collection_id, query):
        messages = []
        images = []
        results_data = []  # For DataFrame

        async for event in deep_research.process(query, collection_id):
            if event.get("event") == "sources":
                data = event["data"]
                # Store all results
                images = []  # Reset images for new batch
                results_data = []  # Reset results for new batch
                for result in data["results"]:
                    # Convert base64 to PIL Image
                    import base64
                    from io import BytesIO
                    from PIL import Image

                    image_data = base64.b64decode(result["image"])
                    image = Image.open(BytesIO(image_data))
                    images.append(image)

                    results_data.append([
                        result["score"],
                        result["page"],
                        result["name"]
                    ])

            if event.get("event") == "chunk":
                messages.append(event["data"])

            response_text = "".join(messages)
            yield (
                response_text,
                images,
                results_data
            )

    # # bulk index all pdf files in seed_data
    # async def bulk_index_seed_data():
    #     seed_data_dir = Path("/root/seed_data")
    #     if not seed_data_dir.exists():
    #         print(f"Seed data directory {seed_data_dir} does not exist.")
    #         return

    #     print(f"Scanning for PDFs in {seed_data_dir}...")
    #     pdf_files = list(seed_data_dir.rglob("*.pdf"))
    #     total_files = len(pdf_files)
    #     print(f"Found {total_files} PDF files.")

    #     collection_id = "demo_collection"

    #     for idx, pdf_path in enumerate(pdf_files, start=1):
    #         try:
    #             # Check if file is already indexed (optional optimization, skipping for now)
    #             print(f"[{idx}/{total_files}] Indexing {pdf_path.name}...")
    #             start_time = time.time()

    #             with open(pdf_path, "rb") as f:
    #                 pdf_bytes = f.read()

    #             async for state in deep_research.add_pdf(collection_id, pdf_path.name, pdf_bytes):
    #                 if "message" in state:
    #                     print(f"[{pdf_path.name}] {state['message'].strip()}")

    #             elapsed_time = time.time() - start_time
    #             print(f"Successfully indexed {pdf_path.name} in {elapsed_time:.2f} seconds")
    #         except Exception as e:
    #             elapsed_time = time.time() - start_time
    #             print(f"Error indexing {pdf_path.name} after {elapsed_time:.2f} seconds: {e}")
    
    # asyncio.run(bulk_index_seed_data())

    with gr.Blocks(title="RAG Demo") as demo:
        gr.Markdown("# RAG Demo")
        gr.Markdown("Upload a PDF and ask questions about its contents.")

        with gr.Tabs():
            with gr.Tab("Upload PDF"):
                pdf_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                )
                upload_button = gr.Button("Upload and Index")
                upload_output = gr.Textbox(
                    label="Upload Status",
                    lines=10,
                    interactive=False
                )
                upload_collection_id = gr.Textbox(
                    lines=1,
                    placeholder="Enter the Collection ID from the Upload tab...",
                    label="Collection ID"
                )

                upload_button.click(
                    fn=upload_pdf,
                    inputs=[pdf_input],
                    outputs=[upload_output, upload_collection_id]
                )

            with gr.Tab("Query PDF"):
                query_collection_id = gr.Textbox(
                    lines=1,
                    value="demo_collection",
                    label="Collection ID"
                )
                upload_collection_id.change(
                    lambda x: x, inputs=[upload_collection_id], outputs=[query_collection_id])
                query_input = gr.Textbox(
                    lines=2,
                    placeholder="Enter your query here...",
                    label="Query"
                )
                query_button = gr.Button("Submit Query")

                with gr.Row():
                    with gr.Column():
                        response_output = gr.Textbox(
                            label="Response",
                            lines=10,
                            interactive=False,
                        )
                    with gr.Column():
                        image_output = gr.Gallery(
                            label="Retrieved Images",
                        )
                        score_output = gr.Dataframe(
                            headers=["Score", "Page", "Document Name"],
                            label="Results"
                        )

                query_button.click(
                    fn=process_query,
                    inputs=[query_collection_id, query_input],
                    outputs=[
                        response_output,
                        image_output,
                        score_output
                    ],
                    show_progress=True
                )

    return gr.mount_gradio_app(web_app, demo, path="/")
