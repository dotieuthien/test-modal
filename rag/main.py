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

    import gradio as gr

    class SearchRequest(BaseModel):
        query: str
        instance_id: UUID
        count: int = 3

    web_app = FastAPI()

    qdrant = InMemoryQdrant()

    vision_rag = VRAG(colpali=colpali, qdrant=qdrant)

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
                    async for state in vision_rag.add_pdf(*byte_file):
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
            async for stage in vision_rag.run_vrag(str(query.instance_id), query.query, query.count):
                yield stage

        return EventSourceResponse(event_generator())

    async def upload_pdf(pdf_file):
        if pdf_file is None:
            return "Please select a PDF file to upload", None

        collection_id = str(uuid.uuid4())
        # Gradio provides the file path directly
        with open(pdf_file.name, 'rb') as f:
            pdf_bytes = f.read()
        filename = Path(pdf_file.name).name

        messages = ["Starting to index PDF..."]
        try:
            async for state in vision_rag.add_pdf(collection_id, filename, pdf_bytes):
                if isinstance(state, ServerSentEvent):
                    data = json.loads(state.data)
                    if "message" in data:
                        messages.append(data["message"])
                else:
                    data = json.loads(state)
                    if "message" in data:
                        messages.append(data["message"])

            messages.append(f"PDF indexed successfully!")
            return "\n".join(messages), collection_id
        except Exception as e:
            return f"Error indexing PDF: {str(e)}", None

    async def process_query(collection_id, query):
        messages = []
        images = []
        results_data = []  # For DataFrame
        current_message = ""

        async for event in vision_rag.run_vrag(collection_id, query):
            if isinstance(event, ServerSentEvent):
                data = json.loads(event.data)
                if event.event == "sources":
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
                    yield (
                        current_message,
                        images,
                        results_data  # Will be displayed as DataFrame
                    )

                try:
                    current_message += data["chunk"]
                    yield (
                        current_message,
                        images,
                        results_data
                    )
                except:
                    pass

    with gr.Blocks(title="Visual RAG System", css=".gradio-container { width: 1200px !important; }") as demo:
        gr.Markdown("# Visual RAG System")
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
                    placeholder="Enter the Collection ID from the Upload tab...",
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
                            interactive=False
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
                    ]
                )

    return gr.mount_gradio_app(web_app, demo, path="/")
