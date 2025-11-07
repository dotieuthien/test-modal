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

### Benchmark Results

Performance comparison across different model and GPU configurations:

| **Metric**                         | **Description**                       | **GPT-OSS (TP = 4, 4×A100)** | **GPT-OSS (TP = 2, 2×A100)** | **GPT-OSS (TP = 1, 1×A100)** | **Qwen3-235B (TP = 4, 4×A100)** |
| :--------------------------------- | :------------------------------------ | :--------------------------: | :--------------------------: | :--------------------------: | :-----------------------------: |
| **Duration (s)**                   | Total benchmark runtime               |           **25.23**          |             34.14            |             52.57            |              62.38              |
| **Total Input Tokens**             | Tokens received as input              |            22 946            |            22 946            |            22 946            |              23 260             |
| **Total Output Tokens**            | Tokens generated as output            |            21 664            |            21 624            |            21 691            |              21 359             |
| **Request Throughput (req/s)**     | Requests handled per second           |           **3.96**           |             2.93             |             1.90             |               1.60              |
| **Total Token Throughput (tok/s)** | Input + output tokens per second      |         **1 768.32**         |           1 305.38           |            849.15            |              715.30             |
| **Max Output Tokens/s**            | Peak generation speed                 |          **1 102.0**         |             814.0            |             725.0            |              432.0              |
| **Max Concurrent Requests**        | Highest simultaneous requests handled |            **17**            |              16              |              14              |                15               |
| **Mean TTFT (ms)**                 | Avg. time to first token              |            399.07            |          **390.59**          |            333.19            |            **229.32**           |
| **p99 TTFT (ms)**                  | 99th-percentile first-token latency   |            773.73            |          **597.93**          |           1 411.02           |            **728.41**           |
| **Mean TPOT (ms)**                 | Avg. time per output token            |           **9.88**           |             13.66            |             22.15            |              25.91              |
| **p99 TPOT (ms)**                  | 99th-percentile token generation time |           **23.58**          |           **24.02**          |             27.64            |              30.58              |
| **Mean ITL (ms)**                  | Avg. inter-token latency              |           **8.91**           |             13.15            |             21.45            |              25.47              |
| **p99 ITL (ms)**                   | 99th-percentile inter-token latency   |          **102.38**          |          **107.38**          |             79.68            |              196.60             |


**Key Findings**:
- **GPT-OSS (TP=4, 4xA100)**: Best throughput with 9.88ms mean TPOT
- **GPT-OSS (TP=1, 1xA100)**: Lower throughput but acceptable latency for single-user scenarios
- **Qwen3-235B (TP=4, 4xA100)**: Fastest TTFT at 229.32ms, suitable for large-scale deployment

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