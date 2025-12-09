import modal
from typing import Dict, Any, List, Tuple
from pathlib import Path


unsloth_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install(
        "unsloth==2025.11.1",
    )
    .pip_install(
        "einops",
        "easydict",
        "addict",
        "matplotlib"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"})
)

# Create volumes for model storage and caching
CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("deepseek-ocr-outputs", create_if_missing=True)

MINUTES = 60
HOURS = 60 * MINUTES

test_image = Path(__file__).parent / "test.jpg"
test_image_remote = Path("/root/test.jpg")

test_image_mount = modal.Mount.from_local_file(
    test_image,
    remote_path=test_image_remote,
)

app = modal.App(
    "deepseek-ocr-unsloth", 
    mounts=[
        test_image_mount
    ])


def create_datacollator_class():
    """Factory function to create the DeepSeekOCRDataCollator class"""
    import torch
    import math
    from dataclasses import dataclass
    from typing import Dict, List, Any, Tuple
    from PIL import Image, ImageOps
    from torch.nn.utils.rnn import pad_sequence
    import io

    @dataclass
    class DeepSeekOCRDataCollator:
        """
        Custom datacollator for DeepSeek OCR finetuning

        Args:
            tokenizer: Tokenizer
            model: Model
            image_size: Size for image patches (default: 640)
            base_size: Size for global view (default: 1024)
            crop_mode: Whether to use dynamic cropping for large images
            train_on_responses_only: If True, only train on assistant responses
        """
        tokenizer: Any
        model: Any
        image_size: int = 640
        base_size: int = 1024
        crop_mode: bool = True
        image_token_id: int = 128815
        train_on_responses_only: bool = True

        def __init__(
            self,
            tokenizer,
            model,
            image_size: int = 640,
            base_size: int = 1024,
            crop_mode: bool = True,
            train_on_responses_only: bool = True,
        ):
            self.tokenizer = tokenizer
            self.model = model
            self.image_size = image_size
            self.base_size = base_size
            self.crop_mode = crop_mode
            self.image_token_id = 128815
            self.dtype = model.dtype
            self.train_on_responses_only = train_on_responses_only

            # Import here to avoid issues at module level
            from deepseek_ocr.modeling_deepseekocr import (
                BasicImageTransform,
                dynamic_preprocess,
            )

            self.dynamic_preprocess = dynamic_preprocess
            self.image_transform = BasicImageTransform(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                normalize=True
            )
            self.patch_size = 16
            self.downsample_ratio = 4

            if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                self.bos_id = tokenizer.bos_token_id
            else:
                self.bos_id = 0
                print(f"Warning: tokenizer has no bos_token_id, using default: {self.bos_id}")

        def deserialize_image(self, image_data) -> Image.Image:
            """Convert image data to PIL Image in RGB mode"""
            if isinstance(image_data, Image.Image):
                return image_data.convert("RGB")
            elif isinstance(image_data, dict) and 'bytes' in image_data:
                image_bytes = image_data['bytes']
                image = Image.open(io.BytesIO(image_bytes))
                return image.convert("RGB")
            else:
                raise ValueError(f"Unsupported image format: {type(image_data)}")

        def process_image(self, image: Image.Image) -> Tuple[List, List, List, List, Tuple[int, int]]:
            """Process a single image based on crop_mode"""
            images_list = []
            images_crop_list = []
            images_spatial_crop = []

            if self.crop_mode:
                if image.size[0] <= 640 and image.size[1] <= 640:
                    crop_ratio = (1, 1)
                    images_crop_raw = []
                else:
                    images_crop_raw, crop_ratio = self.dynamic_preprocess(
                        image, min_num=2, max_num=9,
                        image_size=self.image_size, use_thumbnail=False
                    )

                global_view = ImageOps.pad(
                    image, (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean)
                )
                images_list.append(self.image_transform(global_view).to(self.dtype))

                width_crop_num, height_crop_num = crop_ratio
                images_spatial_crop.append([width_crop_num, height_crop_num])

                if width_crop_num > 1 or height_crop_num > 1:
                    for crop_img in images_crop_raw:
                        images_crop_list.append(
                            self.image_transform(crop_img).to(self.dtype)
                        )

                num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
                num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

                tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
                tokenized_image += [self.image_token_id]

                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                        num_queries * height_crop_num)

            else:
                crop_ratio = (1, 1)
                images_spatial_crop.append([1, 1])

                if self.base_size <= 640:
                    resized_image = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                    images_list.append(self.image_transform(resized_image).to(self.dtype))
                else:
                    global_view = ImageOps.pad(
                        image, (self.base_size, self.base_size),
                        color=tuple(int(x * 255) for x in self.image_transform.mean)
                    )
                    images_list.append(self.image_transform(global_view).to(self.dtype))

                num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
                tokenized_image = ([self.image_token_id] * num_queries + [self.image_token_id]) * num_queries
                tokenized_image += [self.image_token_id]

            return images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio

        def process_single_sample(self, messages: List[Dict]) -> Dict[str, Any]:
            """Process a single conversation into model inputs"""
            from deepseek_ocr.modeling_deepseekocr import text_encode

            images = []
            for message in messages:
                if "images" in message and message["images"]:
                    for img_data in message["images"]:
                        if img_data is not None:
                            pil_image = self.deserialize_image(img_data)
                            images.append(pil_image)

            if not images:
                raise ValueError("No images found in sample")

            tokenized_str = []
            images_seq_mask = []
            images_list, images_crop_list, images_spatial_crop = [], [], []

            prompt_token_count = -1
            assistant_started = False
            image_idx = 0

            tokenized_str.append(self.bos_id)
            images_seq_mask.append(False)

            for message in messages:
                role = message["role"]
                content = message["content"]

                if role == "<|Assistant|>":
                    if not assistant_started:
                        prompt_token_count = len(tokenized_str)
                        assistant_started = True
                    content = f"{content.strip()} {self.tokenizer.eos_token}"

                text_splits = content.split('<image>')

                for i, text_sep in enumerate(text_splits):
                    tokenized_sep = text_encode(self.tokenizer, text_sep, bos=False, eos=False)
                    tokenized_str.extend(tokenized_sep)
                    images_seq_mask.extend([False] * len(tokenized_sep))

                    if i < len(text_splits) - 1:
                        if image_idx >= len(images):
                            raise ValueError("Data mismatch: Found '<image>' token but no corresponding image")

                        image = images[image_idx]
                        img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(image)

                        images_list.extend(img_list)
                        images_crop_list.extend(crop_list)
                        images_spatial_crop.extend(spatial_crop)

                        tokenized_str.extend(tok_img)
                        images_seq_mask.extend([True] * len(tok_img))

                        image_idx += 1

            if image_idx != len(images):
                raise ValueError(f"Data mismatch: Found {len(images)} images but only {image_idx} '<image>' tokens")

            if not assistant_started:
                print("Warning: No assistant message found in sample. Masking all tokens.")
                prompt_token_count = len(tokenized_str)

            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)

            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)

            return {
                "input_ids": torch.tensor(tokenized_str, dtype=torch.long),
                "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
                "images_ori": images_ori,
                "images_crop": images_crop,
                "images_spatial_crop": images_spatial_crop_tensor,
                "prompt_token_count": prompt_token_count,
            }

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            """Collate batch of samples"""
            batch_data = []

            for feature in features:
                try:
                    processed = self.process_single_sample(feature['messages'])
                    batch_data.append(processed)
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue

            if not batch_data:
                raise ValueError("No valid samples in batch")

            input_ids_list = [item['input_ids'] for item in batch_data]
            images_seq_mask_list = [item['images_seq_mask'] for item in batch_data]
            prompt_token_counts = [item['prompt_token_count'] for item in batch_data]

            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            images_seq_mask = pad_sequence(images_seq_mask_list, batch_first=True, padding_value=False)

            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels[images_seq_mask] = -100

            if self.train_on_responses_only:
                for idx, prompt_count in enumerate(prompt_token_counts):
                    if prompt_count > 0:
                        labels[idx, :prompt_count] = -100

            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            images_batch = []
            for item in batch_data:
                images_batch.append((item['images_crop'], item['images_ori']))

            images_spatial_crop = torch.cat([item['images_spatial_crop'] for item in batch_data], dim=0)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "images": images_batch,
                "images_seq_mask": images_seq_mask,
                "images_spatial_crop": images_spatial_crop,
            }

    return DeepSeekOCRDataCollator


@app.function(
    image=unsloth_image,
    gpu="T4",
    timeout=24 * HOURS,
    max_containers=1,
    volumes={
        CACHE_DIR: cache_volume,
        OUTPUT_DIR: output_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],  # For downloading models
)
def train_deepseek_ocr(
    dataset_name: str = "hezarai/parsynth-ocr-200k",
    dataset_split: str = "train[:1000]",
    num_train_epochs: int = 1,
    max_steps: int = 60,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 16,
    image_size: int = 640,
    base_size: int = 1024,
    crop_mode: bool = True,
):
    """
    Finetune DeepSeek OCR model using Unsloth

    Args:
        dataset_name: HuggingFace dataset name
        dataset_split: Dataset split to use
        num_train_epochs: Number of training epochs (set to None if using max_steps)
        max_steps: Maximum training steps (set to None if using num_train_epochs)
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        image_size: Image size for patches
        base_size: Base size for global view
        crop_mode: Whether to use dynamic cropping
    """
    import torch
    from unsloth import FastVisionModel, is_bf16_supported
    from transformers import AutoModel, Trainer, TrainingArguments
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    import os

    # Suppress warnings
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

    print("Starting DeepSeek OCR Finetuning with Unsloth")

    print("\n[1/6] Downloading DeepSeek OCR model...")
    model_path = f"{CACHE_DIR}/deepseek_ocr"

    if not os.path.exists(model_path):
        snapshot_download(
            "unsloth/DeepSeek-OCR", 
            local_dir=model_path,
        )
        cache_volume.commit()
    else:
        print(f"Model already exists at {model_path}")

    print("\n[2/6] Loading model and tokenizer...")
    
    # from transformers import AutoConfig
    # model_config = AutoConfig.from_pretrained(
    #     model_path,
    #     trust_remote_code=True
    # )

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=False,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing="unsloth",
    )
    print("Model loaded successfully!")

    print("\n[3/6] Adding LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print(f"LoRA adapters added (r={lora_r}, alpha={lora_alpha})")

    print(f"\n[4/6] Loading dataset: {dataset_name} [{dataset_split}]...")
    dataset = load_dataset(dataset_name, split=dataset_split)
    dataset = dataset.rename_column("image_path", "image")

    # Convert to conversation format
    instruction = "<image>\nFree OCR. "

    def convert_to_conversation(sample):
        conversation = [
            {
                "role": "<|User|>",
                "content": instruction,
                "images": [sample['image']]
            },
            {
                "role": "<|Assistant|>",
                "content": sample["text"]
            },
        ]
        return {"messages": conversation}

    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    print(f"Dataset prepared: {len(converted_dataset)} samples")

    print("\n[5/6] Creating data collator...")

    # Add CACHE_DIR to sys.path so deepseek_ocr module can be imported
    import sys
    sys.path.insert(0, CACHE_DIR)
    print(f"Added {CACHE_DIR} to Python path")

    # Get the datacollator class from factory function
    DeepSeekOCRDataCollator = create_datacollator_class()

    data_collator = DeepSeekOCRDataCollator(
        tokenizer=tokenizer,
        model=model,
        image_size=image_size,
        base_size=base_size,
        crop_mode=crop_mode,
        train_on_responses_only=True,
    )

    print("\n[6/6] Starting training...")
    FastVisionModel.for_training(model)

    output_dir = f"{OUTPUT_DIR}/checkpoints"

    # Build training arguments based on whether using max_steps or num_train_epochs
    training_args_dict = {
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": 5,
        "learning_rate": learning_rate,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.001,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "fp16": not is_bf16_supported(),
        "bf16": is_bf16_supported(),
        "output_dir": output_dir,
        "report_to": "none",
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
    }

    # Add either max_steps or num_train_epochs (not both)
    if max_steps and max_steps > 0:
        training_args_dict["max_steps"] = max_steps
        # num_train_epochs is commented out when using max_steps
    else:
        training_args_dict["num_train_epochs"] = num_train_epochs

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,  # Must use!
        train_dataset=converted_dataset,
        args=TrainingArguments(**training_args_dict),
    )

    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU: {gpu_stats.name}")
    print(f"Max memory: {max_memory} GB")
    print(f"Reserved: {start_gpu_memory} GB")
    print("=" * 80)

    # Train!
    trainer_stats = trainer.train()

    # Show final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    print(f"Time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
    print(f"Peak memory: {used_memory} GB ({used_percentage}%)")
    print(f"Training memory: {used_memory_for_lora} GB ({lora_percentage}%)")
    
    print("\n[7/6] Saving merged model...")
    merged_output_dir = f"{OUTPUT_DIR}/unsloth_finetune"
    model.save_pretrained_merged(merged_output_dir, tokenizer)
    output_volume.commit()

    print(f"Merged model saved to: {merged_output_dir}")
    print("=" * 80)

    return {
        "status": "success",
        "training_time_minutes": round(trainer_stats.metrics['train_runtime']/60, 2),
        "peak_memory_gb": used_memory,
        "merged_output_dir": merged_output_dir,
    }


@app.function(
    image=unsloth_image,
    gpu="T4",
    timeout=60 * 30,
    volumes={
        CACHE_DIR: cache_volume,
        OUTPUT_DIR: output_volume,
    },
)
def test_finetuned_model(image_path: str = None):
    """
    Test the finetuned model on an image

    Args:
        image_path: Path to test image (optional, will use saved test image if not provided)
    """
    import torch
    from unsloth import FastVisionModel
    from transformers import AutoModel
    import os
    import sys

    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

    print("Loading finetuned merged model...")

    merged_model_path = f"{OUTPUT_DIR}/unsloth_finetune"

    if not os.path.exists(merged_model_path):
        raise ValueError(f"Merged model not found at {merged_model_path}. Please run training first.")

    # Add CACHE_DIR to sys.path for deepseek_ocr module
    sys.path.insert(0, CACHE_DIR)
    print(f"Added {CACHE_DIR} to Python path")

    # Copy model.safetensors.index.json from base model to merged model path
    base_model_path = f"{CACHE_DIR}/deepseek_ocr"
    import shutil
    index_file_src = f"{base_model_path}/model.safetensors.index.json"
    index_file_dst = f"{merged_model_path}/model.safetensors.index.json"

    if os.path.exists(index_file_src) and not os.path.exists(index_file_dst):
        print(f"Copying model.safetensors.index.json to {merged_model_path}...")
        shutil.copy(index_file_src, index_file_dst)

    # Load merged model directly
    print(f"Loading merged model from {merged_model_path}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        merged_model_path,
        load_in_4bit=False,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing="unsloth",
    )

    FastVisionModel.for_inference(model)

    # Test inference
    if not image_path:
        image_path = "/root/test.jpg"

    if not os.path.exists(image_path):
        print(f"Test image not found at {image_path}")
        return {"status": "error", "message": "Test image not found"}

    prompt = "<image>\nFree OCR. "

    print(f"Running inference on: {image_path}")

    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=f"{OUTPUT_DIR}/inference_results",
        image_size=640,
        base_size=1024,
        crop_mode=True,
        save_results=True,
        test_compress=False
    )

    print("Inference completed!")
    print("Result:")
    print(result)

    return {
        "status": "success",
        "result": result
    }


@app.local_entrypoint()
def main(
    command: str = "train",
    dataset: str = "hezarai/parsynth-ocr-200k",
    split: str = "train[:1000]",
    max_steps: int = 60,
    epochs: int = None,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    test_image: str = None,
):
    """
    Main entry point for finetuning

    Commands:
        train: Start training
        test: Test finetuned model

    Examples:
        # Quick test (60 steps)
        modal run finetune_deepseek_ocr_unsloth.py

        # Full training (1 epoch)
        modal run finetune_deepseek_ocr_unsloth.py --command train --epochs 1 --max-steps 0

        # Custom dataset
        modal run finetune_deepseek_ocr_unsloth.py --dataset "your/dataset" --split "train[:5000]"

        # Test model
        modal run finetune_deepseek_ocr_unsloth.py --command test --test-image "/path/to/image.jpg"
    """

    if command == "train":
        print("Starting training on Modal...")
        result = train_deepseek_ocr.remote(
            dataset_name=dataset,
            dataset_split=split,
            num_train_epochs=epochs if epochs else 1,
            max_steps=max_steps if max_steps > 0 else None,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
        )
        print("\n" + "=" * 80)
        print("Training Result:")
        print("=" * 80)
        for key, value in result.items():
            print(f"{key}: {value}")

    elif command == "test":
        print("Testing finetuned model...")
        result = test_finetuned_model.remote(image_path=test_image)
        print("\n" + "=" * 80)
        print("Test Result:")
        print("=" * 80)
        print(result)

    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, test")