from io import BytesIO
from pathlib import Path
import time

import modal

app = modal.App("example-qwen-image")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
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
        "torch",
        "torchvision",
        "torchaudio",
        "git+https://github.com/huggingface/diffusers",
        "transformers",
        "accelerate",
        "https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.2/nunchaku-1.0.2+torch2.10-cp311-cp311-linux_x86_64.whl"
    )
)

N_GPU = 1
MINUTES = 60
HOURS = 60 * MINUTES

# Download the model

MODEL_NAME = "Qwen/Qwen-Image-Edit-2509"

CACHE_DIR = "/cache"
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

secrets = [modal.Secret.from_name("huggingface-secret")]

image = image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": str(CACHE_DIR)})


with image.imports():
    import torch
    import math
    from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.utils import load_image
    from PIL import Image


@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes=volumes,
    secrets=secrets,
    timeout=24 * HOURS,
)
class Model:
    @modal.enter()
    def enter(self):
        print(f"Loading {MODEL_NAME}...")

        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=None)

    @modal.method()
    def inference(
        self,
        image_bytes: bytes,
        prompt: str,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 12,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = " ",
        seed: int | None = None,
    ) -> bytes:
        start_time = time.time()

        init_image = load_image(Image.open(BytesIO(image_bytes))).resize((512, 512))

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        pipe_start = time.time()
        with torch.inference_mode():
            output = self.pipe(
                image=init_image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                generator=generator,
                num_images_per_prompt=1,
            )
            image = output.images[0]
        pipe_latency = time.time() - pipe_start

        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")

        total_latency = time.time() - start_time

        print(f"Inference latency: {total_latency:.3f}s (model: {pipe_latency:.3f}s)")

        return byte_stream.getvalue()


@app.cls(
    image=image,
    gpu="L4",
    volumes=volumes,
    secrets=secrets,
    timeout=24 * HOURS,
)
class NunchakuModel:
    """
    Nunchaku-optimized Qwen-Image-Edit Lightning model for ultra-fast image editing.
    Uses SVDQ quantization for 4-step or 8-step inference with low memory usage.

    Performance: 2-4x faster than standard model with minimal quality loss.
    Memory: Can run on GPUs with as little as 3-4GB VRAM using per-layer offloading.
    """

    @modal.enter()
    def enter(self):
        """
        Initialize the nunchaku-optimized Lightning model.

        Args:
            num_inference_steps: 4 or 8 steps (default: 4 for fastest inference)
            rank: 32 or 128 (default: 32 for speed, 128 for quality)
        """
        # Import nunchaku modules after installation
        from nunchaku import NunchakuQwenImageTransformer2DModel
        from nunchaku.utils import get_gpu_memory, get_precision
        
        num_inference_steps = 4
        rank = 32

        self.num_inference_steps = 4
        self.rank = 32

        print(f"Loading Qwen-Image-Edit Lightning with nunchaku...")
        print(f"  - Precision: {get_precision()}")
        print(f"  - GPU Memory: {get_gpu_memory():.1f}GB")

        # Configure Lightning scheduler
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        # Load nunchaku-optimized transformer
        model_path = f"nunchaku-tech/nunchaku-qwen-image-edit-2509/svdq-{get_precision()}_r{rank}-qwen-image-edit-2509-lightningv2.0-{num_inference_steps}steps.safetensors"
        print(f"  - Loading model: {model_path}")

        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            model_path,
            cache_dir=CACHE_DIR,
        )

        # Create pipeline with nunchaku transformer
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )

        # Configure memory offloading based on GPU memory
        gpu_memory = get_gpu_memory()
        if gpu_memory > 18:
            print("  - Using model CPU offloading (>18GB VRAM)")
            self.pipe.enable_model_cpu_offload()
        else:
            print(f"  - Using per-layer offloading (<18GB VRAM)")
            # Use per-layer offloading for low VRAM (requires only 3-4GB)
            num_blocks_on_gpu = max(1, int(gpu_memory / 6))  # Adjust based on VRAM
            print(f"    - Blocks on GPU: {num_blocks_on_gpu}")
            transformer.set_offload(
                True,
                use_pin_memory=False,
                num_blocks_on_gpu=num_blocks_on_gpu
            )
            self.pipe._exclude_from_cpu_offload.append("transformer")
            self.pipe.enable_sequential_cpu_offload()

        self.pipe.set_progress_bar_config(disable=None)
        print("Nunchaku Lightning model loaded successfully!")

    @modal.method()
    def inference(
        self,
        image_bytes: bytes | list[bytes],
        prompt: str,
        true_cfg_scale: float = 1.0,
        num_inference_steps: int = None,
        seed: int | None = None,
    ) -> bytes:
        """
        Ultra-fast image editing with nunchaku optimization.

        Args:
            image_bytes: Single image or list of images as bytes
            prompt: Text description for editing (can reference multiple images)
            true_cfg_scale: CFG scale (default: 1.0 for Lightning)
            num_inference_steps: Override default steps if needed
            seed: Random seed for reproducibility (optional)

        Returns:
            PNG image as bytes

        Example:
            # Single image edit
            output = inference(image_bytes, "make it a cartoon style")

            # Multi-image composition
            output = inference(
                [image1_bytes, image2_bytes, image3_bytes],
                "Let the person in image 1 sit on the chair in image 2, with the dog in image 3"
            )
        """
        start_time = time.time()

        # Use default num_inference_steps if not specified
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps

        # Load image(s)
        load_start = time.time()
        if isinstance(image_bytes, bytes):
            # Single image
            images = [load_image(Image.open(BytesIO(image_bytes))).convert("RGB")]
        else:
            # Multiple images
            images = [load_image(Image.open(BytesIO(img_bytes))).convert("RGB") for img_bytes in image_bytes]
        load_time = time.time() - load_start

        # Set random seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        # Run inference
        pipe_start = time.time()
        with torch.inference_mode():
            output = self.pipe(
                image=images if len(images) > 1 else images[0],
                prompt=prompt,
                true_cfg_scale=true_cfg_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
            output_image = output.images[0]
        pipe_latency = time.time() - pipe_start

        # Convert to bytes
        byte_stream = BytesIO()
        output_image.save(byte_stream, format="PNG")

        total_latency = time.time() - start_time

        print(f"Nunchaku Lightning Edit - Total: {total_latency:.3f}s (load: {load_time:.3f}s, inference: {pipe_latency:.3f}s)")

        return byte_stream.getvalue()


@app.local_entrypoint()
def main(
    mode: str = "nunchaku",  # "edit" or "nunchaku"
    image_path: str = None,  # Required for all modes
    output_path: str = None,
    prompt: str = None,
    num_iterations: int = 10,
    num_inference_steps: int = None,
    seed: int = 0,
):
    """
    Unified entrypoint for Qwen-Image inference.

    Modes:
    - "edit": Standard image editing (12 steps, full model)
    - "nunchaku": Ultra-fast image editing (4 steps, nunchaku SVDQ)

    Args:
        mode: Inference mode ("edit" or "nunchaku")
        image_path: Input image path (required for all modes)
        output_path: Output image path (auto-generated if not provided)
        prompt: Text prompt for editing
        num_iterations: Number of inference runs (default: 1)
        num_inference_steps: Number of steps (default: 12 for edit, 4 for nunchaku)
        seed: Random seed for reproducibility (default: 0)
    """

    # Set default values based on mode
    if mode == "edit":
        if image_path is None:
            image_path = Path(__file__).parent / "demo_images/dog.png"
        if prompt is None:
            prompt = "A cute dog wizard inspired by Gandalf from Lord of the Rings, featuring detailed fantasy elements in Studio Ghibli style"
        if output_path is None:
            output_path = Path(__file__).parent / "demo_images/edit_output.png"
        if num_inference_steps is None:
            num_inference_steps = 12
    elif mode == "nunchaku":
        if image_path is None:
            image_path = Path(__file__).parent / "demo_images/dog.png"
        if prompt is None:
            prompt = "A cute dog wizard inspired by Gandalf from Lord of the Rings, featuring detailed fantasy elements in Studio Ghibli style"
        if output_path is None:
            output_path = Path(__file__).parent / "demo_images/nunchaku_output.png"
        if num_inference_steps is None:
            num_inference_steps = 4
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'edit' or 'nunchaku'")

    # Convert paths to Path objects
    if isinstance(output_path, str):
        output_path = Path(output_path)
    if image_path and isinstance(image_path, str):
        image_path = Path(image_path)

    # Create output directory
    output_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*80}")
    print(f"Mode: {mode.upper()}")
    print(f"Prompt: {prompt}")
    print(f"Input image: {image_path}")
    print(f"Output path: {output_path}")
    print(f"Iterations: {num_iterations}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    # Run inference iterations
    for i in range(num_iterations):
        iteration_start = time.time()

        if num_iterations > 1:
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")

        # Load input image
        load_start = time.time()
        print(f"📖 Loading input image from {image_path}")
        input_image_bytes = image_path.read_bytes()
        load_time = time.time() - load_start
        print(f"✓ Image loaded in {load_time:.3f}s ({len(input_image_bytes) / 1024:.1f} KB)")

        # Run inference
        inference_start = time.time()
        if mode == "edit":
            print(f"✏️  Editing image with standard model...")
            output_image_bytes = Model().inference.remote(
                input_image_bytes,
                prompt,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
        else:  # nunchaku mode
            print(f"⚡ Editing image with NunchakuModel...")
            output_image_bytes = NunchakuModel().inference.remote(
                input_image_bytes,
                prompt,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
        inference_time = time.time() - inference_start
        print(f"✓ Inference completed in {inference_time:.3f}s")

        # Save output image
        save_start = time.time()
        print(f"💾 Saving output image to {output_path}")
        output_path.write_bytes(output_image_bytes)
        save_time = time.time() - save_start
        print(f"✓ Image saved in {save_time:.3f}s ({len(output_image_bytes) / 1024:.1f} KB)")

        # Total iteration time
        iteration_time = time.time() - iteration_start

        # Print detailed latency breakdown
        print(f"\n📊 Latency Breakdown (Iteration {i+1}):")
        print(f"  - Image loading:     {load_time:8.3f}s ({load_time/iteration_time*100:5.1f}%)")
        print(f"  - Remote inference:  {inference_time:8.3f}s ({inference_time/iteration_time*100:5.1f}%)")
        print(f"  - Image saving:      {save_time:8.3f}s ({save_time/iteration_time*100:5.1f}%)")
        print(f"  - Total:             {iteration_time:8.3f}s")

    print(f"\n✅ All {num_iterations} iteration(s) completed successfully!")
    print(f"📁 Final output saved to: {output_path}")