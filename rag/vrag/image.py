import modal


colpali_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "colpali_engine==0.3.0",
        "torch",
        "transformers==4.44.2",
        "einops==0.8.0",
        "vidore_benchmark==4.0.1",
    )
    .pip_install("numpy==2.1.1")
    .pip_install("opencv_python_headless==4.10.0.84")
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "opencv_python_headless==4.10.0.84",
        "pydantic==2.9.1",
        "pypdfium2==4.30.0",
        "fastapi==0.114.2",
        "qdrant_client==1.11.2",
        "sse-starlette==2.1.3",
        "openai==1.44.1",
        "httpx==0.27.2",
    )
    .pip_install("numpy==2.1.1")
)