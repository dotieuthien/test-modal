#!/usr/bin/env python3
"""
ShareGPT Benchmark - vLLM-style implementation
Based on vLLM's serve.py and endpoint_request_func.py

Run with:
    python vllm_sharegpt_benchmark.py
"""

import asyncio
import aiohttp
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from tqdm.asyncio import tqdm


class StreamedResponseHandler:
    """Handles streaming HTTP responses by accumulating chunks until complete
    messages are available."""

    def __init__(self):
        self.buffer = ""

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        """Add a chunk of bytes to the buffer and return any complete
        messages."""
        chunk_str = chunk_bytes.decode("utf-8")
        self.buffer += chunk_str

        messages = []

        # Split by double newlines (SSE message separator)
        while "\n\n" in self.buffer:
            message, self.buffer = self.buffer.split("\n\n", 1)
            message = message.strip()
            if message:
                messages.append(message)

        # if self.buffer is not empty, check if it is a complete message
        # by removing data: prefix and check if it is a valid JSON
        if self.buffer.startswith("data: "):
            message_content = self.buffer.removeprefix("data: ").strip()
            if message_content == "[DONE]":
                messages.append(self.buffer.strip())
                self.buffer = ""
            elif message_content:
                try:
                    json.loads(message_content)
                    messages.append(self.buffer.strip())
                    self.buffer = ""
                except json.JSONDecodeError:
                    # Incomplete JSON, wait for more chunks.
                    pass

        return messages


@dataclass
class RequestFuncInput:
    """The input for the request function."""
    prompt: str
    api_url: str
    model: str
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass
class RequestFuncOutput:
    """The output of the request function including metrics."""
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies
    error: str = ""
    start_time: float = 0.0
    prompt_len: int = 0


@dataclass
class BenchmarkMetrics:
    completed: int
    failed: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    p99_itl_ms: float
    mean_e2el_ms: float
    median_e2el_ms: float
    p99_e2el_ms: float


# Configuration
CONFIG = {
    "server_url": "https://aifarm.mservice.com.vn/internal/wedjat-llm-vllm-gpt-oss-120b/v1/chat/completions",
    "model_name": "gpt-oss-120b",
    # "server_url": "https://dotieuthien--gpt-oss120b-vllm-openai-compatible-serve.modal.run/v1/chat/completions",
    # "model_name": "openai/gpt-oss-120b",
    "bearer_token": "f0e9d8c7b6a5f4e3d2c1b0a9f8e7d6c5b4a3b2a1",
    "sharegpt_dataset_path": "/home/thiendo1/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json",
    "results_dir": "./vllm_benchmark_results",
}


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """
    The async request function for the OpenAI Chat Completions API.
    Based on vLLM's implementation.
    """
    api_url = request_func_input.api_url

    payload = {
        "model": request_func_input.model,
        "messages": [
            {"role": "user", "content": request_func_input.prompt},
        ],
        "temperature": request_func_input.temperature,
        "max_tokens": request_func_input.max_tokens,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CONFIG['bearer_token']}",
    }

    output = RequestFuncOutput()
    output.prompt_len = len(request_func_input.prompt.split())  # Simple tokenization

    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st

    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                handler = StreamedResponseHandler()
                async for chunk_bytes in response.content.iter_any():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    messages = handler.add_chunk(chunk_bytes)
                    for message in messages:
                        # NOTE: SSE comments (often used as pings) start with
                        # a colon. These are not JSON data payload and should
                        # be skipped.
                        if message.startswith(":"):
                            continue

                        chunk = message.removeprefix("data: ")

                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")

                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft
                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")

                            most_recent_timestamp = timestamp

                output.generated_text = generated_text
                output.success = True
                output.latency = most_recent_timestamp - st
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


def load_sharegpt_prompts(
    dataset_path: str,
    num_prompts: int = 100,
    filter_image_conversations: bool = True
) -> list[str]:
    """
    Load text-only prompts from ShareGPT dataset.

    Args:
        dataset_path: Path to local ShareGPT JSON file
        num_prompts: Number of prompts to extract
        filter_image_conversations: Whether to filter out conversations with images

    Returns:
        List of prompts
    """
    print(f"Loading ShareGPT dataset from: {dataset_path}")

    # Load the ShareGPT dataset from local JSON file
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    print(f"Dataset loaded with {len(dataset)} conversations")
    print(f"Extracting text-only prompts...")

    prompts = []
    count = 0

    for idx, example in enumerate(dataset):
        if count >= num_prompts:
            break

        # Each example has a "conversations" field with a list of messages
        conversations = example.get("conversations", [])

        # Filter out conversations with images if requested
        if filter_image_conversations:
            has_image = any("image" in msg.get("value", "").lower() or
                          "<image>" in msg.get("value", "")
                          for msg in conversations)
            if has_image:
                continue

        # Extract user prompts (messages with "from": "human")
        for msg in conversations:
            if msg.get("from") == "human" and count < num_prompts:
                prompt = msg.get("value", "")

                # Skip empty or very short prompts
                if len(prompt.strip()) < 10:
                    continue

                prompts.append(prompt)
                count += 1
                if count >= num_prompts:
                    break

    print(f"Extracted {len(prompts)} text-only prompts\n")
    return prompts


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    dur_s: float,
    selected_percentiles: list[float] = [99]
) -> BenchmarkMetrics:
    """
    Calculate the metrics for the benchmark.
    Based on vLLM's calculate_metrics function.
    """
    total_input = 0
    completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []

    for output in outputs:
        if output.success:
            output_len = output.output_tokens

            if output_len:
                total_input += output.prompt_len
                tpot = 0
                if output_len > 1:
                    latency_minus_ttft = output.latency - output.ttft
                    tpot = latency_minus_ttft / (output_len - 1)
                    tpots.append(tpot)

                itls += output.itl
                ttfts.append(output.ttft)
                e2els.append(output.latency)
                completed += 1

    if completed == 0:
        print("WARNING: All requests failed!")

    metrics = BenchmarkMetrics(
        completed=completed,
        failed=len(outputs) - completed,
        total_input=total_input,
        total_output=sum(o.output_tokens for o in outputs if o.success),
        request_throughput=completed / dur_s,
        output_throughput=sum(o.output_tokens for o in outputs if o.success) / dur_s,
        total_token_throughput=(total_input + sum(o.output_tokens for o in outputs if o.success)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        p99_e2el_ms=np.percentile(e2els or 0, 99) * 1000,
    )

    return metrics


def print_metrics(metrics: BenchmarkMetrics, benchmark_duration: float, max_concurrency: int):
    """Print benchmark metrics in vLLM style."""
    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.failed))
    print("{:<40} {:<10}".format("Maximum request concurrency:", max_concurrency))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput))

    print("{s:{c}^{n}}".format(s=" Time to First Token (TTFT) ", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))

    print("{s:{c}^{n}}".format(s=" Time per Output Token (TPOT) ", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))

    print("{s:{c}^{n}}".format(s=" Inter-Token Latency (ITL) ", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))

    print("{s:{c}^{n}}".format(s=" End-to-End Latency (E2EL) ", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean E2EL (ms):", metrics.mean_e2el_ms))
    print("{:<40} {:<10.2f}".format("Median E2EL (ms):", metrics.median_e2el_ms))
    print("{:<40} {:<10.2f}".format("P99 E2EL (ms):", metrics.p99_e2el_ms))
    print("=" * 50)


async def benchmark(
    api_url: str,
    model: str,
    prompts: list[str],
    max_concurrency: int,
    max_tokens: int,
    temperature: float,
    disable_tqdm: bool = False,
) -> tuple[list[RequestFuncOutput], float]:
    """
    Run the benchmark with concurrent requests.
    Based on vLLM's benchmark function.
    """
    # Reuses connections across requests to reduce TLS handshake overhead
    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        force_close=False,
    )

    session = aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
    )

    print("Starting benchmark...")
    pbar = None if disable_tqdm else tqdm(total=len(prompts))

    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_request_func(prompt: str):
        async with semaphore:
            request_input = RequestFuncInput(
                prompt=prompt,
                api_url=api_url,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return await async_request_openai_chat_completions(
                request_func_input=request_input,
                session=session,
                pbar=pbar
            )

    benchmark_start_time = time.perf_counter()

    # Create all tasks
    tasks = [asyncio.create_task(limited_request_func(prompt)) for prompt in prompts]

    # Execute all requests concurrently
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    await session.close()

    return outputs, benchmark_duration


def save_results(
    results: dict[str, Any],
    timestamp: str,
) -> Path:
    """Save benchmark results to JSON file."""
    results_dir = Path(CONFIG["results_dir"]) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    result_file = results_dir / "benchmark_results.json"

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {result_file}")
    return result_file


async def main_async(
    num_prompts: int = 100,
    max_concurrency: int = 10,
    max_tokens: int = 512,
    temperature: float = 0.0,
    filter_image_conversations: bool = True,
    save_to_file: bool = True,
):
    """Main async function to run the benchmark."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("vLLM-style ShareGPT Benchmark")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Server: {CONFIG['server_url']}")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Number of prompts: {num_prompts}")
    print(f"Max concurrency: {max_concurrency}")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Filter image conversations: {filter_image_conversations}")
    print("=" * 60 + "\n")

    # Load ShareGPT prompts
    prompts = load_sharegpt_prompts(
        CONFIG["sharegpt_dataset_path"],
        num_prompts=num_prompts,
        filter_image_conversations=filter_image_conversations
    )

    if not prompts:
        print("ERROR: No prompts loaded!")
        return

    # Run benchmark
    outputs, benchmark_duration = await benchmark(
        api_url=CONFIG["server_url"],
        model=CONFIG["model_name"],
        prompts=prompts,
        max_concurrency=max_concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Calculate metrics
    metrics = calculate_metrics(outputs, benchmark_duration)

    # Print metrics
    print_metrics(metrics, benchmark_duration, max_concurrency)

    # Save results
    if save_to_file:
        results = {
            "timestamp": timestamp,
            "config": {
                "server_url": CONFIG["server_url"],
                "model": CONFIG["model_name"],
                "num_prompts": num_prompts,
                "max_concurrency": max_concurrency,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "filter_image_conversations": filter_image_conversations,
            },
            "summary": {
                "benchmark_duration": benchmark_duration,
                "completed": metrics.completed,
                "failed": metrics.failed,
                "total_input_tokens": metrics.total_input,
                "total_output_tokens": metrics.total_output,
                "request_throughput": metrics.request_throughput,
                "output_throughput": metrics.output_throughput,
                "total_token_throughput": metrics.total_token_throughput,
                "mean_ttft_ms": metrics.mean_ttft_ms,
                "median_ttft_ms": metrics.median_ttft_ms,
                "p99_ttft_ms": metrics.p99_ttft_ms,
                "mean_tpot_ms": metrics.mean_tpot_ms,
                "median_tpot_ms": metrics.median_tpot_ms,
                "p99_tpot_ms": metrics.p99_tpot_ms,
                "mean_itl_ms": metrics.mean_itl_ms,
                "median_itl_ms": metrics.median_itl_ms,
                "p99_itl_ms": metrics.p99_itl_ms,
                "mean_e2el_ms": metrics.mean_e2el_ms,
                "median_e2el_ms": metrics.median_e2el_ms,
                "p99_e2el_ms": metrics.p99_e2el_ms,
            },
            "detailed_results": [
                {
                    "success": output.success,
                    "prompt_len": output.prompt_len,
                    "output_tokens": output.output_tokens,
                    "ttft": output.ttft,
                    "latency": output.latency,
                    "itl_count": len(output.itl),
                    "error": output.error if output.error else None,
                }
                for output in outputs
            ]
        }
        save_results(results, timestamp)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE!")
    print("=" * 60)


def main():
    """Main entry point."""
    asyncio.run(main_async(
        num_prompts=100,                 # Number of prompts to extract from ShareGPT
        max_concurrency=10,              # Number of concurrent requests
        max_tokens=512,                  # Max tokens to generate
        temperature=0.0,                 # 0.0 = greedy sampling
        filter_image_conversations=True, # Filter out conversations with images
        save_to_file=True,               # Save results to JSON
    ))


if __name__ == "__main__":
    main()