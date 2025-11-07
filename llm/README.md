# vLLM Server Deployment Guide

This guide explains how to deploy vLLM servers on Modal for GPT-OSS-120B and Qwen-235B models.

## Prerequisites

- Modal account with API token configured
- Modal CLI installed (`pip install modal`)
- Authenticated with Modal (`modal token new`)

## Available Servers

### 1. GPT-OSS-120B Server
**File**: `vllm_llm_inference.py`
**Model**: `openai/gpt-oss-120b`
**App Name**: `gpt-oss120b-vllm-openai-compatible`

### 2. Qwen-235B Server
**File**: `vllm_llm_qwen_235b.py`
**Model**: `koushd/Qwen3-235B-A22B-Instruct-2507-AWQ`
**App Name**: `qwen-235b-vllm-openai-compatible`

## Deployment Commands

### Deploy GPT-OSS-120B with 1 GPU

```bash
modal deploy /Users/dotieuthien/Documents/rnd/test-modal/llm/vllm_llm_inference.py
```

### Deploy GPT-OSS-120B with 4 GPUs

To deploy with 4 GPUs, first edit the file:

```python
# In vllm_llm_inference.py
N_GPU = 4  # Change from 1 to 4
```

Then deploy:

```bash
modal deploy vllm_llm_inference.py
```


### Deploy Qwen-235B (4 GPUs Required)

```bash
modal deploy vllm_llm_qwen_235b.py
```

## Benchmarking

### Overview

The `vllm_benchmark.py` script provides comprehensive benchmarking capabilities for deployed vLLM servers using public datasets and custom tests.

**File**: `vllm_benchmark.py`
**App Name**: `vllm-benchmark-client`

### Available Benchmarks

1. **ShareGPT** - Basic LLM text generation benchmark
2. **VisionArena** - Vision-language model benchmark
3. **Custom Images** - Benchmark with local images
4. **Structured Output** - JSON/Grammar/Regex/Choice generation benchmarks

### Configuration

Before running benchmarks, update the server configuration in `vllm_benchmark.py`:

```python
BENCHMARK_CONFIG = {
    "server_url": "https://your-workspace--gpt-oss120b-vllm-openai-compatible-serve.modal.run",
    "model_name": "openai/gpt-oss-120b",
    "served_model_name": "openai/gpt-oss-120b",
}
```

Or for Qwen-235B:

```python
BENCHMARK_CONFIG = {
    "server_url": "https://your-workspace--qwen-235b-vllm-openai-compatible-serve.modal.run",
    "model_name": "koushd/Qwen3-235B-A22B-Instruct-2507-AWQ",
    "served_model_name": "koushd/Qwen3-235B-A22B-Instruct-2507-AWQ",
}
```

### Running All Benchmarks

To run all configured benchmarks:

```bash
modal run vllm_benchmark.py
```

This will execute all enabled benchmarks in the `main()` function (lines 794-803 in the script). By default, only ShareGPT is enabled.