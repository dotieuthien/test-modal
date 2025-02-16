import modal


vllm_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.6.3post1", 
    "fastapi[standard]==0.115.4",
    "GPUtil"
)

MODELS_DIR = "/llama_models"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_REVISION = "a7a06a1cc11b4514ce9edcde0e3ca1d16e5ff2fc"

volume = modal.Volume.from_name("llama_models", create_if_missing=True)

app = modal.App("example-vllm-openai-compatible")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
TOKEN = "super-secret-token"  # auth token. for production use, replace with a modal.Secret

MINUTES = 60  # seconds
HOURS = 60 * MINUTES


@app.function(
    image=vllm_image,
    gpu=f"A10G:{N_GPU}",
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=1000,
    volumes={MODELS_DIR: volume},
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
    from vllm.entrypoints.openai.serving_engine import BaseModelPath
    from vllm.usage.usage_lib import UsageContext
    
    
    def print_system_info():
        import psutil
        import GPUtil

        # Memory info
        mem = psutil.virtual_memory()
        print(f"Memory: Total={mem.total / (1024**3):.2f}GB, Available={mem.available / (1024**3):.2f}GB")
        
        # GPU info
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}, Memory Total={gpu.memoryTotal}MB, Memory Used={gpu.memoryUsed}MB")


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
        tensor_parallel_size=N_GPU,
        gpu_memory_utilization=0.90,
        max_model_len=8096,
        enforce_eager=False,  # capture the graph for faster inference, but slower cold starts (30s > 20s)
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    model_config = get_model_config(engine)

    request_logger = RequestLogger(max_log_len=2048)

    base_model_paths = [
        BaseModelPath(name=MODEL_NAME.split("/")[1], model_path=MODEL_NAME)
    ]

    api_server.chat = lambda s: (
        print_system_info(),
        OpenAIServingChat(
            engine,
            model_config=model_config,
            base_model_paths=base_model_paths,
            chat_template=None,
            response_role="assistant",
            lora_modules=[],
            prompt_adapters=[],
            request_logger=request_logger,
        )
    )[1]

    api_server.completion = lambda s: (
        print_system_info(),
        OpenAIServingCompletion(
            engine,
            model_config=model_config,
            base_model_paths=base_model_paths,
            lora_modules=[],
            prompt_adapters=[],
            request_logger=request_logger,
        )
    )[1]

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