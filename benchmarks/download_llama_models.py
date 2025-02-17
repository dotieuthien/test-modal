import modal


MODELS_DIR = "/llama_models"

DEFAULT_NAME = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_REVISION = "a7a06a1cc11b4514ce9edcde0e3ca1d16e5ff2fc"

GGUF_MODELS_NAME = "tensorblock/Qwen2-VL-7B-Instruct-GGUF"

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
    image=image, secrets=[modal.Secret.from_name("huggingface")]
)


@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS)
def download_model(model_name, force_download=False):
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
            "Qwen2-VL-7B-Instruct-Q5_K_M.gguf",
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
    download_gguf_model.remote(gguf_model_name, force_download)