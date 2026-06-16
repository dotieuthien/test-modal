import logging
from io import BytesIO
from typing import Annotated
from pydantic import Field

import modal
from modal.runner import deploy_app



logger = logging.getLogger(__name__)

MINUTES = 60  # seconds
VARIANT = "schnell"  # or "dev", but note [dev] requires you to accept terms and conditions on HF
NUM_INFERENCE_STEPS = 4  # use ~50 for [dev], smaller for [schnell]
IMAGE_FORMAT = "JPEG"

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11").entrypoint([])
diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"

flux_image = (
    cuda_dev_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "invisible_watermark==0.2.0",
        "transformers==4.44.0",
        "huggingface_hub[hf_transfer]==0.26.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
        "torch==2.5.0",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "numpy<2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"})
)

flux_image = flux_image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    }
)

app_name = "mcp-tool-flux"
app = modal.App(app_name)

with flux_image.imports():
    import torch
    from diffusers import FluxPipeline


@app.cls(
    gpu="L40S",
    container_idle_timeout=5 * MINUTES,
    image=flux_image,
    volumes={
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
    },
    enable_memory_snapshot=True,
)
class Model:
    @modal.enter(snap=True)
    def load(self):
        print("🔄 Loading model on CPU to utilize memory snapshot...")
        pipe = FluxPipeline.from_pretrained(f"black-forest-labs/FLUX.1-{VARIANT}", torch_dtype=torch.bfloat16)
        self.pipe = _optimize(pipe)

    @modal.enter(snap=False)
    def setup(self):
        print("🔄 Moving model to GPU...")
        self.pipe = self.pipe.to("cuda")

    @modal.method()
    def inference(self, prompt: str) -> bytes:
        print("🎨 Generating image...")
        out = self.pipe(
            prompt,
            output_type="pil",
            num_inference_steps=NUM_INFERENCE_STEPS,
        ).images[0]

        byte_stream = BytesIO()
        out.save(byte_stream, format=IMAGE_FORMAT)
        return byte_stream.getvalue()


def _optimize(pipe):
    # fuse QKV projections in Transformer and VAE
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()

    # switch memory layout to Torch's preferred, channels_last
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    return pipe


async def process_flux_txt2img(prompt: Annotated[str, Field(description="The prompt to generate an image for")]):
    """Let's you generate an image using the Flux model."""
    try:
        cls = modal.Cls.from_name(app_name, Model.__name__)
        image_bytes = await cls().inference.remote.aio(prompt)
    except Exception as e:
        # await ctx.info("App not found. Deploying...")
        logger.info("App not found. Deploying...")
        deploy_app(app)
        
        cls = modal.Cls.from_name(app_name, Model.__name__)
        image_bytes = await cls().inference.remote.aio(prompt)
        
    return image_bytes



if __name__ == "__main__":
    import asyncio
    asyncio.run(process_flux_txt2img("a beautiful image of a cat"))
