# DeepSeek OCR API - Deployment & Testing Guide

## Overview

This service provides two OCR workflows:
- **DeepSeek OCR** (`deepseek-ai/DeepSeek-OCR`): Pure OCR text extraction
- **SilverAI OCR** (`silverai/silverai-ocr`): OCR + Qwen LLM for intelligent document analysis

## Prerequisites

1. Install Modal CLI:
```bash
pip install modal
```

2. Authenticate with Modal:
```bash
modal token new
```

3. Ensure models are downloaded in `/llama_models/`:
   - `deepseek-ai/DeepSeek-OCR`
   - `Qwen/Qwen3-0.6B`

## Deployment

Deploy the service to Modal:
```bash
modal deploy deepseek_ocr.py
```

This will:
- Build the container image with all dependencies
- Load both DeepSeek OCR and Qwen models
- Create the FastAPI endpoint
- Return a public URL (e.g., `https://[username]--deepseek-ocr-openai-compatible-fastapi-app.modal.run`)

## Testing

Run the test client:
```bash
python deepseek_ocr.py
```

This will execute two examples:
1. **Pure OCR**: Extract all text from an image
2. **OCR + Qwen**: Answer questions about document content

### Example Output

```
[Example 1] Pure DeepSeek OCR Extraction
OCR Result: [extracted text...]
Latency: Preprocessing=50ms, Inference=1200ms, Total=1300ms

[Example 2] SilverAI OCR (DeepSeek OCR + Qwen LLM)
Qwen LLM Answer: [intelligent response...]
Latency: OCR=1200ms, Qwen=800ms, Total=2050ms
```

## API Usage

### Using OpenAI Python Client

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="https://[your-url].modal.run/v1",
    api_key="empty"
)

# Read and encode image
with open("image.png", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode('utf-8')

# Option 1: Pure OCR
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-OCR",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "<image>\nFree OCR."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }]
)

# Option 2: OCR + Qwen LLM
response = client.chat.completions.create(
    model="silverai/silverai-ocr",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is the total amount in this invoice?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }]
)
```

## Available Endpoints

- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Main OCR endpoint (model-based routing)

## Latency Tracking

Both workflows include detailed latency metrics:

**Pure OCR:**
- `preprocessing_ms`: Image decoding time
- `inference_ms`: OCR model execution time
- `model_total_ms`: Total model processing time
- `request_total_ms`: End-to-end request time

**OCR + Qwen:**
- `ocr_ms`: DeepSeek OCR processing time
- `qwen_ms`: Qwen LLM processing time
- `total_ms`: Combined model processing time
- `request_total_ms`: End-to-end request time

## Configuration

Edit `deepseek_ocr.py` to customize:
- `MODELS_DIR`: Path to model files (default: `/llama_models`)
- `MODEL_NAME`: DeepSeek OCR model name
- `QWEN_MODEL_NAME`: Qwen LLM model name
- `N_GPU`: Number of GPUs to use (default: 1)
- GPU type: Change `gpu="L4:1"` in `@app.cls` decorator

## Troubleshooting

**Models not found:**
- Ensure models are downloaded to the correct path in Modal volume
- Check `MODELS_DIR` matches your Modal volume mount point

**Out of memory:**
- Reduce `max_tokens` in requests
- Use a larger GPU type (e.g., `A100:1`)

**Slow cold starts:**
- Models load on first request (~30-60 seconds)
- Subsequent requests use warm containers
- Adjust `container_idle_timeout` to keep containers warm longer