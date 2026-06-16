# Test Modal

A collection of experimental projects for deploying and benchmarking ML/AI workloads on [Modal](https://modal.com/) — a serverless GPU computing platform.

## Directory Structure

```
test-modal/
├── llm/              # Deploy & benchmark LLMs (vLLM, SGLang)
├── rag/              # Vision RAG with ColPali + Qdrant
├── tts/              # Text-to-Speech with TensorRT-LLM (F5-TTS)
├── agentic_code/     # AI agent (LangGraph) + Figma integration
├── modal-mcp-tools/  # MCP server for Claude/AI tools
├── diffusion/        # Image generation (Flux)
├── test-grpc/        # gRPC & FastAPI server
├── triton/           # NVIDIA Triton inference
└── profiling/        # PyTorch profiling
```

## Key Projects

### LLM Server (`llm/`)

Deploy and benchmark large LLMs on Modal with multi-GPU support.

- **GPT-OSS-120B** — vLLM, 1-4 A100-80GB GPUs, OpenAI-compatible API
- **Qwen3-235B** — AWQ quantization, 4 GPUs
- **Benchmark** — Compare throughput, latency, TTFT, TPOT between vLLM and SGLang

```bash
modal deploy llm/vllm_llm_gpt_oss_120b.py
modal run llm/vllm_benchmark.py
```

### Vision RAG (`rag/`)

Multi-modal RAG: upload PDFs, generate vision embeddings (ColPali), perform semantic search via Qdrant, and answer queries with OpenAI/Gemini.

- FastAPI + Gradio web UI
- Streaming responses (SSE)
- DeepResearch agent

```bash
modal deploy rag/main.py
```

**Required secrets:** `openai`, `googlecloud-secret`, `qdrant-secret`

### Text-to-Speech (`tts/`)

F5-TTS with TensorRT-LLM + Triton Inference Server on L4 GPU.

```bash
modal run tts/trtllm_f5_tts.py       # Build & test
modal deploy tts/trtllm_f5_tts.py    # Deploy
python tts/test_client_http.py       # Test
```

### AI Agent (`agentic_code/`)

LangGraph-based agent that analyzes Figma designs and generates Python code. Uses Modal Sandbox for safe code execution.

```bash
modal run agentic_code/agent.py --question "Your question"
```

### MCP Tools (`modal-mcp-tools/`)

MCP server providing a `flux_txt2img` tool (text-to-image) for Claude and other AI assistants.

```bash
python modal-mcp-tools/main.py
```

## Requirements

- Python 3.10+
- [Modal CLI](https://modal.com/docs/getting-started)

```bash
pip install modal
modal token new
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Platform | Modal, CUDA 12.4+ |
| Inference | vLLM, SGLang, TensorRT-LLM, Triton |
| Models | GPT-OSS-120B, Qwen3-235B, F5-TTS, Flux, ColPali |
| Frameworks | LangGraph, FastAPI, Gradio, FastMCP |
| Vector DB | Qdrant |
