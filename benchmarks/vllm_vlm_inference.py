import modal


vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-runtime-ubuntu22.04", add_python="3.12")
    .pip_install(
        "GPUtil",
        "transformers>=4.48.2",
        "vllm==v0.7.3", 
    )
    .run_commands("apt-get update")
    .run_commands("apt-get install -y nvtop")
)

MODELS_DIR = "/llama_models"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

volume = modal.Volume.from_name("llama_models", create_if_missing=True)

app = modal.App("example-vllm-openai-compatible")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
TOKEN = "super-secret-token"  # auth token. for production use, replace with a modal.Secret

MINUTES = 60  # seconds
HOURS = 60 * MINUTES


@app.function(
    image=vllm_image,
    gpu=f"L4:{N_GPU}",
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

    from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                        OpenAIServingModels)
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
    
    openai_serving_models = OpenAIServingModels(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=[],
        prompt_adapters=[],
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
    
    # Run system info printing in a separate thread
    import threading
    import time

    def periodic_system_info():
        time.sleep(10)  # Wait for 10 seconds initially
        while True:
            print_system_info()
            time.sleep(10)  # Print every minute

    threading.Thread(target=periodic_system_info, daemon=True).start()

    return web_app


@app.function(volumes={TRACE_DIR: traces}, **config)
def profile(
    function,
    label: str = None,
    steps: int = 3,
    schedule=None,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
    print_rows: int = 0,
    **kwargs,
):
    from uuid import uuid4

    if isinstance(function, str):
        try:
            function = app.registered_functions[function]
        except KeyError:
            raise ValueError(f"Function {function} not found")
    function_name = function.tag

    output_dir = (
        TRACE_DIR
        / (function_name + (f"_{label}" if label else ""))
        / str(uuid4())
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if schedule is None:
        if steps < 3:
            raise ValueError(
                "Steps must be at least 3 when using default schedule"
            )
        schedule = {"wait": 1, "warmup": 1, "active": steps - 2, "repeat": 0}

    schedule = torch.profiler.schedule(**schedule)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        for _ in range(steps):
            function.local(**kwargs)  # <-- here we wrap the target Function
            prof.step()

    if print_rows:
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=print_rows
            )
        )

    trace_path = sorted(
        output_dir.glob("**/*.pt.trace.json"),
        key=lambda pth: pth.stat().st_mtime,
        reverse=True,
    )[0]

    print(f"trace saved to {trace_path.relative_to(TRACE_DIR)}")

    return trace_path.read_text(), trace_path.relative_to(TRACE_DIR)


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