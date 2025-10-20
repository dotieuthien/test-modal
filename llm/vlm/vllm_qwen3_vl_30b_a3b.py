import modal
from pathlib import Path


vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "GPUtil",
        "vllm==v0.11.0",
        "qwen-vl-utils==0.0.14",
        "gradio",
        "requests",
        "modelscope_studio",
        "dashscope",
    )
    .run_commands("apt-get update")
    .run_commands("apt-get install -y nvtop")
    .pip_install("openai")
)

MODELS_DIR = "/llama_models"
MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"

volume = modal.Volume.from_name("llama_models", create_if_missing=True)

app = modal.App("example-vllm-qwen3-vl-30b-a3b")

N_GPU = 2  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
# auth token. for production use, replace with a modal.Secret
TOKEN = "super-secret-token"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES


# gradio app folder
gradio_dir = Path(__file__).parent / "gradio_ui"
gradio_remote_dir = Path("/root/gradio_ui")

if not gradio_dir.exists():
    raise RuntimeError(
        "gradio_dir directory not found! Place the folder in the same directory."
    )
gradio_mount = modal.Mount.from_local_dir(
    gradio_dir,
    remote_path=gradio_remote_dir,
)


@app.function(
    image=vllm_image,
    gpu=f"A100:{N_GPU}",
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=1000,
    volumes={MODELS_DIR: volume},
    mounts=[gradio_mount],
)
@modal.asgi_app()
def serve():
    import fastapi
    import vllm.entrypoints.openai.api_server as api_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import (
        OpenAIServingCompletion,
    )

    from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                        OpenAIServingModels)
    from vllm.usage.usage_lib import UsageContext
    
    # import gradio app
    import sys
    sys.path.append("/root/gradio_ui")
    
    import gradio as gr
    from gradio_ui.gradio_app import demo as gradio_app

    
    def print_system_info():
        import psutil
        import GPUtil

        # Memory info
        mem = psutil.virtual_memory()
        print(
            f"Memory: Total={mem.total / (1024**3):.2f}GB, Available={mem.available / (1024**3):.2f}GB")

        # GPU info
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            print(
                f"GPU {i}: {gpu.name}, Memory Total={gpu.memoryTotal}MB, Memory Used={gpu.memoryUsed}MB")

    volume.reload()  # ensure we have the latest version of the weights

    # create a fastAPI app that uses vLLM's OpenAI-compatible router
    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com 🚀",
        version="0.0.1",
        docs_url="/docs",
    )

    router = fastapi.APIRouter()

    # wrap vllm's router in auth router
    router.include_router(api_server.router)
    # add authed vllm to our fastAPI app
    web_app.include_router(router)

    engine_args = AsyncEngineArgs(
        model=MODELS_DIR + "/" + MODEL_NAME,
        gpu_memory_utilization=0.90,
        max_model_len=8096,
        tensor_parallel_size=N_GPU,
        enable_expert_parallel=True,
        # capture the graph for faster inference, but slower cold starts (30s > 20s)
        enforce_eager=False,
        limit_mm_per_prompt={"image": 10},
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    model_config = get_model_config(engine)

    request_logger = RequestLogger(max_log_len=2048)

    base_model_paths = [
        BaseModelPath(name=MODEL_NAME.split("/")[1], model_path=MODEL_NAME)
    ]

    openai_serving_models = OpenAIServingModels(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=[],
    )

    api_server.models = lambda s: openai_serving_models

    api_server.chat = lambda s: OpenAIServingChat(
        engine,
        model_config,
        openai_serving_models,
        chat_template=None,
        chat_template_content_format="auto",
        response_role="assistant",
        request_logger=request_logger,
    )

    api_server.completion = lambda s: OpenAIServingCompletion(
        engine,
        model_config,
        openai_serving_models,
        request_logger=request_logger,
    )

    web_app.state.enable_server_load_tracking = True
    web_app.state.server_load_metrics = 0
    gradio_app.queue(default_concurrency_limit=100, max_size=100)
    web_app = gr.mount_gradio_app(web_app, gradio_app, path="/chat")

    return web_app


def get_model_config(engine):
    import asyncio

    try:  # adapted from vLLM source -- https://github.com/vllm-project/vllm/blob/507ef787d85dec24490069ffceacbd6b161f4f72/vllm/entrypoints/openai/api_server.py#L235C1-L247C1
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    return model_config
