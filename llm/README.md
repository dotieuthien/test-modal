# vLLM Server Deployment Guide

This guide explains how to deploy vLLM servers on Modal for GPT-OSS-120B and Qwen-235B models.

## Prerequisites

- Modal account with API token configured
- Modal CLI installed (`pip install modal`)
- Authenticated with Modal (`modal token new`)

## Available Servers

### 1. GPT-OSS-120B Server
**File**: `vllm_llm_gpt_oss_120b.py`
**Model**: `openai/gpt-oss-120b`
**App Name**: `gpt-oss-120b-vllm-openai-compatible`

### 2. Qwen-235B Server
**File**: `vllm_llm_qwen_235b.py`
**Model**: `koushd/Qwen3-235B-A22B-Instruct-2507-AWQ`
**App Name**: `qwen-235b-vllm-openai-compatible`

## Deployment Commands

### Deploy GPT-OSS-120B with 1 GPU

```bash
modal deploy /Users/dotieuthien/Documents/rnd/test-modal/llm/vllm_llm_gpt_oss_120b.py
```

### Deploy GPT-OSS-120B with 4 GPUs

To deploy with 4 GPUs, first edit the file:

```python
# In vllm_llm_gpt_oss_120b.py
N_GPU = 4  # Change from 1 to 4
```

Then deploy:

```bash
modal deploy vllm_llm_gpt_oss_120b.py
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

### Benchmark Results with max_concurrency = 10

Performance comparison using vllm across different model and GPU configurations:

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

Comparison using sglang across different GPU configurations:

| **Metric**                         | **Description**                       | **GPT-OSS (TP = 4, 4×A100)** | **GPT-OSS (TP = 2, 2×A100)** | **GPT-OSS (TP = 1, 1×A100)** |
| :--------------------------------- | :------------------------------------ | :--------------------------: | :--------------------------: | :--------------------------: |
| **Duration (s)**                   | Total benchmark runtime               |           **119.71**         |            156.22            |            130.45            |
| **Total Input Tokens**             | Tokens received as input              |            22 946            |            22 946            |            22 946            |
| **Total Output Tokens**            | Tokens generated as output            |            21 601            |          **21 691**          |            21 599            |
| **Request Throughput (req/s)**     | Requests handled per second           |           **0.835**          |             0.640            |             0.767            |
| **Total Token Throughput (tok/s)** | Input + output tokens per second      |          **372.13**          |            285.74            |            341.47            |
| **Max Output Tokens/s**            | Peak generation speed                 |          **711.0**           |             468.0            |             328.0            |
| **Max Concurrent Requests**        | Highest simultaneous requests handled |            **14**            |            **14**            |              13              |
| **Mean TTFT (ms)**                 | Avg. time to first token              |          **1 486.44**        |           2 591.34           |           2 093.12           |
| **p99 TTFT (ms)**                  | 99th-percentile first-token latency   |          **7 936.21**        |           9 098.50           |           9 574.11           |
| **Mean TPOT (ms)**                 | Avg. time per output token            |           **43.61**          |             59.57            |             52.45            |
| **p99 TPOT (ms)**                  | 99th-percentile token generation time |          **131.86**          |            257.16            |            183.44            |
| **Mean ITL (ms)**                  | Avg. inter-token latency              |           **44.75**          |             58.76            |             51.29            |
| **p99 ITL (ms)**                   | 99th-percentile inter-token latency   |          **263.19**          |            338.01            |            291.40            |

### Benchmark Results with max_concurrency = 1

Performance comparison between vLLM and SGLang across different GPU configurations:

| **Metric**                         | **Description**                       | **vLLM (TP = 4, 4×A100)** | **vLLM (TP = 1, 1×A100)** | **SGLang (TP = 1, 1×A100)** |
| :--------------------------------- | :------------------------------------ | :-----------------------: | :-----------------------: | :-------------------------: |
| **Duration (s)**                   | Total benchmark runtime               |        **138.32**         |           170.02          |           271.35            |
| **Total Input Tokens**             | Tokens received as input              |          22 946           |          22 946           |           22 946            |
| **Total Output Tokens**            | Tokens generated as output            |        **21 539**         |           21 613          |           21 691            |
| **Request Throughput (req/s)**     | Requests handled per second           |        **0.723**          |            0.588          |            0.369            |
| **Total Token Throughput (tok/s)** | Input + output tokens per second      |        **321.62**         |           262.09          |           164.50            |
| **Max Output Tokens/s**            | Peak generation speed                 |        **443.0**          |            376.0          |            216.0            |
| **Max Concurrent Requests**        | Highest simultaneous requests handled |            4              |              4            |              4              |
| **Mean TTFT (ms)**                 | Avg. time to first token              |           198.23          |        **186.27**         |           587.04            |
| **Median TTFT (ms)**               | Median time to first token            |        **134.71**         |           142.65          |           187.25            |
| **p99 TTFT (ms)**                  | 99th-percentile first-token latency   |        **1 238.54**       |           906.35          |          3 592.38           |
| **Mean TPOT (ms)**                 | Avg. time per output token            |        **6.68**           |             7.23          |            10.97            |
| **Median TPOT (ms)**               | Median time per output token          |        **5.01**           |             6.90          |             9.20            |
| **p99 TPOT (ms)**                  | 99th-percentile token generation time |        **34.97**          |            11.97          |            33.82            |
| **Mean ITL (ms)**                  | Avg. inter-token latency              |        **5.53**           |             7.25          |             9.85            |
| **Median ITL (ms)**                | Median inter-token latency            |        **0.098**          |             0.014         |             0.41            |
| **p99 ITL (ms)**                   | 99th-percentile inter-token latency   |        **30.17**          |            35.58          |            36.63            |


**Key Findings**:
- **vLLM (TP=4, 4xA100)**: Best overall performance with 321.62 tok/s throughput and fastest runtime (138.32s)
- **vLLM (TP=1, 1xA100)**: 1.59× faster than SGLang (262.09 vs 164.50 tok/s) with 37% shorter runtime
- **vLLM vs SGLang**: vLLM consistently outperforms SGLang across all metrics - 3.15× faster mean TTFT on single GPU


### Available Benchmarks

1. **ShareGPT** - Basic LLM text generation benchmark
2. **VisionArena** - Vision-language model benchmark
3. **Custom Images** - Benchmark with local images
4. **Structured Output** - JSON/Grammar/Regex/Choice generation benchmarks

### Configuration

Before running benchmarks, update the server configuration in `vllm_benchmark.py`:

```python
BENCHMARK_CONFIG = {
    "server_url": "https://your-workspace--gpt-oss-120b-vllm-openai-compatible-serve.modal.run",
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