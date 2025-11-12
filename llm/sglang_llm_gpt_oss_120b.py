import modal


sglang_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "GPUtil",
        "gradio",
        "requests",
        "modelscope_studio",
        "dashscope",
    )
    .pip_install("openai")
    .run_commands("apt-get update")
    .run_commands(
        "apt-get install -y bash "
        "build-essential "
        "git "
        "git-lfs "
        "curl "
        "ca-certificates "
        "libglib2.0-0 "
        "libsndfile1-dev "
        "libgl1 "
        "nvtop "
        "libnuma1"
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .run_commands(
        'uv pip install --system '
        '"sglang" --prerelease=allow'
    )
)

MODELS_DIR = "/llama_models"
MODEL_NAME = "openai/gpt-oss-120b"

volume = modal.Volume.from_name("llama_models", create_if_missing=True)

app = modal.App("gpt-oss-120b-sglang-openai-compatible")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
# auth token. for production use, replace with a modal.Secret
TOKEN = "super-secret-token"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES
SGLANG_PORT = 8000


@app.function(
    image=sglang_image,
    gpu=f"A100-80GB:{N_GPU}",
    max_containers=1,
    container_idle_timeout=30 * MINUTES,
    timeout=24 * HOURS,
    volumes={
        MODELS_DIR: volume,
    },
)
@modal.concurrent(
    max_inputs=100
)
@modal.web_server(port=SGLANG_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODELS_DIR + "/" + MODEL_NAME,
        # "--mem-fraction-static",  "0.95",
        "--context-length", "32768",
        "--tp", str(N_GPU),
        "--host", "0.0.0.0",
        "--port", str(SGLANG_PORT),
    ]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)
