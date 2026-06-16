import modal
from pathlib import Path


vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "GPUtil",
        "vllm==0.10.2",
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
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
volume = modal.Volume.from_name("llama_models", create_if_missing=True)

FAST_BOOT = False

app = modal.App("example-vllm-openai-compatible")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
# auth token. for production use, replace with a modal.Secret
TOKEN = "super-secret-token"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"L4:{N_GPU}",
    max_containers=1,
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    volumes={
        MODELS_DIR: volume,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(
    max_inputs=100
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODELS_DIR + "/" + MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)