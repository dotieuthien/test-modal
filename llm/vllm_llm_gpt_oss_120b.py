import modal
from pathlib import Path


vllm_image = (
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
        "nvtop"
    )
    .pip_install(
        "torch==2.8.0",
        "torchvision"
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .run_commands(
        "uv self update"
    )
    .run_commands(
        'uv pip install --system '
        'vllm[flashinfer]==0.11'
    )
    .pip_install("lmcache==0.3.7")
)

lmcache_config_path = Path(__file__).parent / "lmcache_config.yaml"
lmcache_config_remote_path = Path("/root/lmcache_config.yaml")

lmcache_config_mount = modal.Mount.from_local_file(
    lmcache_config_path,
    remote_path=lmcache_config_remote_path,
)

MODELS_DIR = "/llama_models"
MODEL_NAME = "openai/gpt-oss-120b"

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
volume = modal.Volume.from_name("llama_models", create_if_missing=True)

FAST_BOOT = False

app = modal.App(
    "gpt-oss-120b-vllm-openai-compatible", 
    mounts=[
        lmcache_config_mount
    ]
)

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
# auth token. for production use, replace with a modal.Secret
TOKEN = "super-secret-token"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",
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
        # 'LMCACHE_CONFIG_FILE="/root/lmcache_config.yaml"',
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODELS_DIR + "/" + MODEL_NAME,
        "--served-model-name", MODEL_NAME,
        # "--gpu-memory-utilization",  "0.95",
        "--max-model-len", "32768",
        # "--max-num-seqs", "8",
        # "--max-num-batched-tokens", "16384",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--async-scheduling",
        # "--kv-transfer-config",
        # '\'{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}\''
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)