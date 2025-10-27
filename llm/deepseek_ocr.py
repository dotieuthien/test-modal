import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import time
import uuid


# Create Modal image with required dependencies
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install(
        "fastapi[standard]",
        "pydantic>=2.0",
        "pillow",
        "requests",
        "transformers",
        "accelerate",
        "torch",
    )
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
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .run_commands(
        'uv pip install --system -U '
        '"triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels" '
        'vllm --pre --extra-index-url https://wheels.vllm.ai/nightly'
    )
)

MODELS_DIR = "/llama_models"
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
QWEN_MODEL_NAME = "Qwen/Qwen3-0.6B"

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
volume = modal.Volume.from_name("llama_models", create_if_missing=True)

app = modal.App("deepseek-ocr-openai-compatible")

N_GPU = 1
MINUTES = 60
HOURS = 60 * MINUTES

# FastAPI app
web_app = FastAPI(title="DeepSeek-OCR API", version="1.0.0")


# Pydantic models for request/response
class ImageUrl(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: str
    image_url: ImageUrl


class TextContent(BaseModel):
    type: str
    text: str


class Message(BaseModel):
    role: str
    content: Union[str, List[Union[TextContent, ImageContent]]]


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = MODEL_NAME
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 8192
    stream: Optional[bool] = False


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Latency(BaseModel):
    preprocessing_ms: float
    inference_ms: float
    model_total_ms: float
    request_total_ms: float


class CombinedLatency(BaseModel):
    ocr_ms: float
    qwen_ms: float
    total_ms: float
    request_total_ms: float


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage
    latency: Optional[Union[Latency, CombinedLatency]] = None
    ocr_text: Optional[str] = None  # For combined workflow


class Model(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str
    data: List[Model]


# Modal class for model management
@app.cls(
    image=vllm_image,
    gpu=f"A100:{N_GPU}",
    container_idle_timeout=10 * MINUTES,
    timeout=24 * HOURS,
    volumes={
        MODELS_DIR: volume,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
class DeepSeekOCRModel:
    def __init__(self):
        self.llm = None
        self.qwen_model = None
        self.qwen_tokenizer = None

    @modal.enter()
    def load_model(self):
        """Load the DeepSeek-OCR and Qwen models on container startup"""
        from vllm import LLM
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading DeepSeek OCR model: {MODEL_NAME}")
        self.llm = LLM(
            model=MODELS_DIR + "/" + MODEL_NAME,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
            tensor_parallel_size=N_GPU,
        )
        print("DeepSeek OCR model loaded successfully!")

        print(f"Loading Qwen model: {QWEN_MODEL_NAME}")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR + "/" + QWEN_MODEL_NAME)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            MODELS_DIR + "/" + QWEN_MODEL_NAME,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Qwen model loaded successfully!")

    @modal.method()
    def generate(self, prompts: List[str], images: List[str], sampling_params_dict: Dict[str, Any]):
        """Generate OCR output for given images"""
        from vllm import SamplingParams
        from PIL import Image
        import base64
        import io

        start_time = time.time()

        # Convert base64 images to PIL Images
        preprocessing_start = time.time()
        pil_images = []
        for img_data in images:
            if isinstance(img_data, str):
                # Decode base64
                img_bytes = base64.b64decode(img_data)
                pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            else:
                pil_images.append(img_data)

        # Prepare model input
        model_input = []
        for prompt, image in zip(prompts, pil_images):
            model_input.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            })
        preprocessing_time = time.time() - preprocessing_start

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=sampling_params_dict.get("temperature", 0.0),
            max_tokens=sampling_params_dict.get("max_tokens", 8192),
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # <td>, </td>
            ),
            skip_special_tokens=False,
        )

        # Generate
        inference_start = time.time()
        outputs = self.llm.generate(model_input, sampling_params)
        inference_time = time.time() - inference_start

        # Extract text from outputs
        results = [output.outputs[0].text for output in outputs]

        total_time = time.time() - start_time

        return {
            "results": results,
            "latency": {
                "preprocessing_ms": round(preprocessing_time * 1000, 2),
                "inference_ms": round(inference_time * 1000, 2),
                "total_ms": round(total_time * 1000, 2)
            }
        }

    @modal.method()
    def generate_with_qwen(self, user_prompt: str, image: str, ocr_max_tokens: int = 8192, qwen_max_tokens: int = 8192):
        """
        Combined workflow: OCR extraction + Qwen processing
        1. Extract text from image using DeepSeek OCR
        2. Combine user prompt with OCR text
        3. Process with Qwen LLM to answer the user's question
        """
        from vllm import SamplingParams
        from PIL import Image
        import base64
        import io

        start_time = time.time()

        # Step 1: Run OCR on the image
        ocr_start = time.time()

        # Convert base64 image to PIL Image
        img_bytes = base64.b64decode(image)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Prepare model input for OCR
        model_input = [{
            "prompt": "<image>\nFree OCR.",
            "multi_modal_data": {"image": pil_image}
        }]

        # Create sampling params for OCR
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=ocr_max_tokens,
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},
            ),
            skip_special_tokens=False,
        )

        # Generate OCR output
        outputs = self.llm.generate(model_input, sampling_params)
        ocr_text = outputs[0].outputs[0].text
        ocr_time = time.time() - ocr_start

        # Step 2: Process with Qwen LLM
        qwen_start = time.time()

        # Combine user prompt with OCR text
        combined_prompt = f"{user_prompt}\n\nOCR Text:\n{ocr_text}"

        messages = [
            {"role": "user", "content": combined_prompt}
        ]

        # Prepare Qwen input
        text = self.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.qwen_tokenizer([text], return_tensors="pt").to(self.qwen_model.device)

        # Generate Qwen response
        generated_ids = self.qwen_model.generate(
            **model_inputs,
            max_new_tokens=qwen_max_tokens
        )

        # Decode output
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        qwen_output = self.qwen_tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        qwen_time = time.time() - qwen_start

        total_time = time.time() - start_time

        return {
            "ocr_text": ocr_text,
            "final_answer": qwen_output,
            "latency": {
                "ocr_ms": round(ocr_time * 1000, 2),
                "qwen_ms": round(qwen_time * 1000, 2),
                "total_ms": round(total_time * 1000, 2)
            }
        }


# Global model instance
model = DeepSeekOCRModel()


# FastAPI routes
@web_app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL_NAME}


@web_app.get("/v1/models", response_model=ModelList)
async def list_models():
    """OpenAI-compatible models endpoint"""
    return ModelList(
        object="list",
        data=[
            Model(
                id=MODEL_NAME,
                object="model",
                created=int(time.time()),
                owned_by="deepseek",
            ),
            Model(
                id="silverai/silverai-ocr",
                object="model",
                created=int(time.time()),
                owned_by="silverai",
            )
        ],
    )


@web_app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with model-based routing:
    - model="deepseek-ai/DeepSeek-OCR" → Pure OCR extraction
    - model="silverai-ocr" → OCR + Qwen LLM processing
    """
    request_start = time.time()

    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not yet supported")

    # Check which workflow to use based on model name
    use_qwen = request.model == "silverai/silverai-ocr"

    if use_qwen:
        # Combined OCR + Qwen workflow
        user_prompt = None
        image_data = None

        for message in request.messages:
            content = message.content

            if isinstance(content, str):
                user_prompt = content
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, TextContent):
                        text_parts.append(part.text)
                    elif isinstance(part, ImageContent):
                        image_url = part.image_url.url
                        if image_url.startswith("data:image"):
                            image_data = image_url.split(",", 1)[1]
                        else:
                            image_data = image_url

                if text_parts:
                    user_prompt = " ".join(text_parts)

        # Validate inputs
        if not image_data:
            raise HTTPException(status_code=400, detail="No image provided in the request")
        if not user_prompt:
            raise HTTPException(status_code=400, detail="No user prompt provided")

        # Call combined workflow
        output = model.generate_with_qwen.remote(
            user_prompt=user_prompt,
            image=image_data,
            ocr_max_tokens=request.max_tokens,
            qwen_max_tokens=request.max_tokens
        )

        # Calculate total request latency
        request_total_ms = round((time.time() - request_start) * 1000, 2)

        # Format response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            object="chat.completion",
            created=int(time.time()),
            model="silverai/silverai-ocr",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=output["final_answer"],
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=len(user_prompt) + len(output["ocr_text"]),
                completion_tokens=len(output["final_answer"]),
                total_tokens=len(user_prompt) + len(output["ocr_text"]) + len(output["final_answer"]),
            ),
            latency=CombinedLatency(
                ocr_ms=output["latency"]["ocr_ms"],
                qwen_ms=output["latency"]["qwen_ms"],
                total_ms=output["latency"]["total_ms"],
                request_total_ms=request_total_ms,
            ),
            ocr_text=output["ocr_text"],
        )

    else:
        # Pure DeepSeek OCR workflow
        prompts = []
        images = []

        for message in request.messages:
            content = message.content

            # Handle different content formats
            if isinstance(content, str):
                # Simple text message - use as-is
                prompts.append(content)
            elif isinstance(content, list):
                # Multi-modal content
                text_parts = []
                image_data = None

                for part in content:
                    if isinstance(part, TextContent):
                        text_parts.append(part.text)
                    elif isinstance(part, ImageContent):
                        image_url = part.image_url.url
                        # Extract base64 data
                        if image_url.startswith("data:image"):
                            # Format: data:image/png;base64,<data>
                            image_data = image_url.split(",", 1)[1]
                        else:
                            # Direct base64 or URL
                            image_data = image_url
                        images.append(image_data)

                # Construct prompt from user's text
                if text_parts:
                    prompt_text = " ".join(text_parts)
                    prompts.append(prompt_text)
                else:
                    # No text provided, use minimal prompt
                    prompts.append("<image>")

        # If no images found, return error
        if not images:
            raise HTTPException(status_code=400, detail="No images provided in the request")

        # Ensure prompts match images count
        if len(prompts) < len(images):
            prompts.extend(["<image>"] * (len(images) - len(prompts)))

        # Generate OCR results
        sampling_params = {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        generation_output = model.generate.remote(prompts, images, sampling_params)

        # Extract results and latency
        results = generation_output["results"]
        model_latency = generation_output["latency"]

        # Calculate total request latency
        request_total_ms = round((time.time() - request_start) * 1000, 2)

        # Format response in OpenAI format
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            object="chat.completion",
            created=int(time.time()),
            model=MODEL_NAME,
            choices=[
                ChatCompletionChoice(
                    index=i,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=result,
                    ),
                    finish_reason="stop",
                )
                for i, result in enumerate(results)
            ],
            usage=Usage(
                prompt_tokens=sum(len(p) for p in prompts),
                completion_tokens=sum(len(r) for r in results),
                total_tokens=sum(len(p) for p in prompts) + sum(len(r) for r in results),
            ),
            latency=Latency(
                preprocessing_ms=model_latency["preprocessing_ms"],
                inference_ms=model_latency["inference_ms"],
                model_total_ms=model_latency["total_ms"],
                request_total_ms=request_total_ms,
            ),
        )

    return response


# Serve FastAPI app with Modal
@app.function(
    image=vllm_image,
    scaledown_window=300,
    allow_concurrent_inputs=100,
    timeout=24 * HOURS,
)
@modal.asgi_app()
def fastapi_app():
    return web_app
    
    
if __name__ == "__main__":
    """Test client demonstrating both workflows"""
    import os
    import sys
    from openai import OpenAI
    import base64
    from pathlib import Path

    # Get deployment URL
    base_url = "https://dotieuthien--deepseek-ocr-openai-compatible-fastapi-app.modal.run/v1"

    # Initialize OpenAI client
    client = OpenAI(
        base_url=base_url,
        api_key="empty",
    )

    # Test image path
    image_path = "/home/thiendo1/Desktop/test-modal/llm/images/test/3.png"

    # Encode image to base64
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')

    print(f"Image size: {len(base64_image)} bytes (base64)")
    print("=" * 80)

    # ========== Example 1: Pure DeepSeek OCR ==========
    print("\n[Example 1] Pure DeepSeek OCR Extraction")
    print("=" * 80)
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-OCR",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<image>\nFree OCR."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=8192,
            temperature=0.0,
        )

        print("OCR Result:")
        print("-" * 80)
        print(response.choices[0].message.content)
        print("-" * 80)
        print(f"Model: {response.model}")
        print(f"Tokens: {response.usage.total_tokens}")

        if hasattr(response, 'latency') and response.latency:
            latency = response.latency
            if isinstance(latency, dict):
                print(f"\nLatency: Preprocessing={latency['preprocessing_ms']:.2f}ms, "
                      f"Inference={latency['inference_ms']:.2f}ms, "
                      f"Total={latency['request_total_ms']:.2f}ms")
            else:
                print(f"\nLatency: Preprocessing={latency.preprocessing_ms:.2f}ms, "
                      f"Inference={latency.inference_ms:.2f}ms, "
                      f"Total={latency.request_total_ms:.2f}ms")

    except Exception as e:
        print(f"Error in Example 1: {e}")
        import traceback
        traceback.print_exc()

    # ========== Example 2: SilverAI OCR (OCR + Qwen) ==========
    print("\n\n[Example 2] SilverAI OCR (DeepSeek OCR + Qwen LLM)")
    print("=" * 80)
    try:
        prompt = """You are a data formatter. Convert the OCR text into a structured JSON format.

Instructions:
- Analyze the OCR text and identify all key-value pairs, fields, and structured data
- Create a valid JSON object that represents the document's structure
- Use descriptive keys based on the field names or labels found in the text
- Return ONLY valid JSON, no additional explanation or text

Example:
If OCR text contains "Date: 2024-01-15, Total: $250.00"
Return: {"date": "2024-01-15", "total": "$250.00"}"""

        response = client.chat.completions.create(
            model="silverai/silverai-ocr",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=8192,
            temperature=0.0,
        )

        print("Qwen LLM Answer:")
        print("-" * 80)
        print(response.choices[0].message.content)
        print("-" * 80)

        if hasattr(response, 'ocr_text') and response.ocr_text:
            ocr_preview = str(response.ocr_text)[:300] + "..." if len(str(response.ocr_text)) > 300 else str(response.ocr_text)
            print(f"\nRaw OCR Text Preview:\n{ocr_preview}\n")

        print(f"Model: {response.model}")
        print(f"Tokens: {response.usage.total_tokens}")

        if hasattr(response, 'latency') and response.latency:
            latency = response.latency
            if isinstance(latency, dict):
                print(f"\nLatency: OCR={latency['ocr_ms']:.2f}ms, "
                      f"Qwen={latency['qwen_ms']:.2f}ms, "
                      f"Total={latency['request_total_ms']:.2f}ms")
            else:
                print(f"\nLatency: OCR={latency.ocr_ms:.2f}ms, "
                      f"Qwen={latency.qwen_ms:.2f}ms, "
                      f"Total={latency.request_total_ms:.2f}ms")

    except Exception as e:
        print(f"Error in Example 2: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)