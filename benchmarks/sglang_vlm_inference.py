import modal


sglang_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install(  # add sglang and some Python dependencies
        "GPUtil",
        # "transformers==4.47.1",
        "numpy<2",
        "fastapi[standard]==0.115.4",
        "pydantic==2.9.2",
        "starlette==0.41.2",
        "torch==2.5.1",
        "sglang[all]>=0.4.3",
        # as per sglang website: https://sgl-project.github.io/start/install.html
        extra_options="--find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python",
    )
    .run_commands("apt-get update")
    .run_commands("apt-get install -y nvtop")
)

MODELS_DIR = "/llama_models"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

MODEL_CHAT_TEMPLATE = "qwen2-vl"

volume = modal.Volume.from_name("llama_models", create_if_missing=True)

app = modal.App("example-sglang-openai-compatible")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
# auth token. for production use, replace with a modal.Secret
TOKEN = "super-secret-token"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES


@app.function(
    image=sglang_image,
    gpu=f"A100-80GB:{N_GPU}",
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=1000,
    volumes={MODELS_DIR: volume},
)
@modal.asgi_app()
def serve():
    import sglang
    from sglang.srt.entrypoints.http_server import app as api_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.entrypoints.http_server import (
        _launch_subprocesses, 
        _wait_and_warmup,  
        set_global_state,
        _GlobalState
    )

    volume.reload()  # ensure we have the latest version of the weights

    server_args = prepare_server_args(
        [
            "--model-path", MODELS_DIR + "/" + MODEL_NAME, 
            "--chat-template", MODEL_CHAT_TEMPLATE,
        ]
    )
    pipe_finish_writer = None
    tokenizer_manager, scheduler_info = _launch_subprocesses(server_args=server_args)
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            scheduler_info=scheduler_info,
        )
    )
    
    
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

    # Run system info printing in a separate thread
    import threading
    import time

    def periodic_system_info():
        time.sleep(10)  # Wait for 10 seconds initially
        while True:
            print_system_info()
            time.sleep(10)  # Print every minute

    threading.Thread(target=periodic_system_info, daemon=True).start()
    
    return api_server


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
