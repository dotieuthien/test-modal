import modal


MODELS_DIR = "/llama_models"

DEFAULT_NAME = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
DEFAULT_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"

GGUF_MODELS_NAME = "Qwen/Qwen2-7B-Instruct-GGUF"

volume = modal.Volume.from_name("llama_models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        [
            "huggingface_hub",  # download models from the Hugging Face Hub
            "hf-transfer",  # download models faster with Rust
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


MINUTES = 60
HOURS = 60 * MINUTES


app = modal.App(
    image=image, secrets=[modal.Secret.from_name("huggingface-secret")]
)


@app.function(
    volumes={MODELS_DIR: volume}, 
    timeout=4 * HOURS)
def download_model(
    model_name, 
    force_download=False
):
    from huggingface_hub import snapshot_download

    volume.reload()

    snapshot_download(
        model_name,
        local_dir=MODELS_DIR + "/" + model_name,
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],  # Ensure safetensors
        force_download=force_download,
    )

    volume.commit()
    
    
@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS)
def download_gguf_model(model_name, force_download=False):
    from huggingface_hub import snapshot_download

    volume.reload()

    snapshot_download(
        model_name,
        local_dir=MODELS_DIR + "/" + model_name,
        allow_patterns = [
            "qwen2-7b-instruct-q5_k_m.gguf",
        ],
        force_download=force_download,
    )

    volume.commit()


@app.local_entrypoint()
def main(
    model_name: str = DEFAULT_NAME,
    gguf_model_name: str = GGUF_MODELS_NAME,
    force_download: bool = False,
):
    download_model.remote(model_name, force_download)
    # download_gguf_model.remote(gguf_model_name, force_download)