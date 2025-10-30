import modal
import subprocess
import os
import json
import base64
from datetime import datetime


app = modal.App("vllm-benchmark-client")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Install git for cloning vLLM repo
    .pip_install(
        "vllm",
        "datasets",
        "huggingface-hub",
        "Pillow",
        "aiohttp",
        "numpy",
    )
    .run_commands(
        # Clone vLLM repo to get benchmark scripts
        "cd /opt && git clone --depth 1 https://github.com/vllm-project/vllm.git",
    )
)

MODELS_DIR = "/llama_models"
IMAGES_DIR = "/custom_images"
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

volume = modal.Volume.from_name("llama_models", create_if_missing=True)

# Mount for local test images
local_images_mount = modal.Mount.from_local_dir(
    "/home/thiendo1/Desktop/test-modal/llm/images/test",
    remote_path=IMAGES_DIR,
)


# Global configuration - shared across all benchmarks
BENCHMARK_CONFIG = {
    "server_url": "https://dotieuthien--gpt-oss120b-vllm-openai-compatible-serve.modal.run",
    "model_name": "openai/gpt-oss-120b",
    "served_model_name": "openai/gpt-oss-120b",
}


def get_result_dir(timestamp: str, dataset_name: str) -> str:
    """Generate result directory path with timestamp and dataset name."""
    return f"{MODELS_DIR}/benchmark_results/{timestamp}/{dataset_name}"


@app.function(
    image=image,
    gpu="T4",
    max_containers=1,
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=100,
    volumes={
        MODELS_DIR: volume,
    },
)
def run_sharegpt_benchmark(
    timestamp: str,
    num_prompts: int = 100,
    backend: str = "vllm",
    endpoint: str = "/v1/completions",
    max_concurrency: int = 10,
    save_results: bool = True,
):
    """
    Run ShareGPT benchmark for basic LLM testing.

    Args:
        timestamp: Timestamp for organizing results
        num_prompts: Number of prompts to use
        backend: Backend type (vllm, openai-chat)
        endpoint: API endpoint path
        max_concurrency: Maximum concurrent requests
        save_results: Whether to save results
    """
    from huggingface_hub import hf_hub_download

    dataset_name = "sharegpt"
    print(f"[ShareGPT Benchmark] Starting with {num_prompts} prompts...")
    print(f"Server: {BENCHMARK_CONFIG['server_url']}")
    print(f"Model: {BENCHMARK_CONFIG['served_model_name']}")

    # Download ShareGPT dataset
    print("Downloading ShareGPT dataset from Hugging Face...")
    downloaded_file = hf_hub_download(
        repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
        filename="ShareGPT_V3_unfiltered_cleaned_split.json",
        repo_type="dataset",
        cache_dir=MODELS_DIR,
    )
    print(f"Dataset downloaded to {downloaded_file}")

    # Prepare model path
    model_path = f"{MODELS_DIR}/{BENCHMARK_CONFIG['model_name']}"

    # Build benchmark command
    cmd = [
        "vllm", "bench", "serve",
        "--backend", backend,
        "--model", model_path,
        "--served-model-name", BENCHMARK_CONFIG["served_model_name"],
        "--endpoint", endpoint,
        "--dataset-name", dataset_name,
        "--dataset-path", downloaded_file,
        "--num-prompts", str(num_prompts),
        "--base-url", BENCHMARK_CONFIG["server_url"],
    ]

    # Add optional parameters
    if max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(max_concurrency)])

    if save_results:
        cmd.append("--save-result")

    # Create result directory with timestamp and dataset name
    result_dir = get_result_dir(timestamp, dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    cmd.extend(["--result-dir", result_dir])

    # Print command for debugging
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print()

    # Run benchmark
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Read and print result files
        if save_results:
            print("\n" + "="*50)
            print(f"Results saved to: {result_dir}")
            print("="*50)
            for file in os.listdir(result_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(result_dir, file)
                    print(f"\n--- {file} ---")
                    with open(file_path, "r") as f:
                        print(f.read())

        return {
            "status": "success",
            "dataset": dataset_name,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "result_dir": result_dir
        }

    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed with error code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {
            "status": "failed",
            "dataset": dataset_name,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }


@app.function(
    image=image,
    gpu="T4",
    max_containers=1,
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=100,
    volumes={
        MODELS_DIR: volume,
    },
)
def run_visionarena_benchmark(
    timestamp: str,
    num_prompts: int = 100,
    backend: str = "openai-chat",
    endpoint: str = "/v1/chat/completions",
    hf_dataset_path: str = "lmarena-ai/VisionArena-Chat",
    hf_split: str = "train",
    hf_subset: str = None,
    max_concurrency: int = 10,
    save_results: bool = True,
):
    """
    Run VisionArena benchmark for Vision Language Models.

    Args:
        timestamp: Timestamp for organizing results
        num_prompts: Number of prompts to use
        backend: Backend type (openai-chat, vllm)
        endpoint: API endpoint path
        hf_dataset_path: HuggingFace dataset path
        hf_split: Dataset split (train, test, validation)
        hf_subset: Dataset subset/configuration name
        max_concurrency: Maximum concurrent requests
        save_results: Whether to save results
    """
    dataset_name = "visionarena"
    print(f"[VisionArena Benchmark] Starting with {num_prompts} prompts...")
    print(f"Server: {BENCHMARK_CONFIG['server_url']}")
    print(f"Model: {BENCHMARK_CONFIG['served_model_name']}")
    print(f"Dataset: {hf_dataset_path}")

    # Prepare model path
    model_path = f"{MODELS_DIR}/{BENCHMARK_CONFIG['model_name']}"

    # Build benchmark command
    cmd = [
        "vllm", "bench", "serve",
        "--backend", backend,
        "--model", model_path,
        "--served-model-name", BENCHMARK_CONFIG["served_model_name"],
        "--endpoint", endpoint,
        "--dataset-name", "hf",
        "--dataset-path", hf_dataset_path,
        "--hf-split", hf_split,
        "--num-prompts", str(num_prompts),
        "--base-url", BENCHMARK_CONFIG["server_url"],
    ]

    # Add optional subset parameter
    if hf_subset:
        cmd.extend(["--hf-subset", hf_subset])

    # Add optional parameters
    if max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(max_concurrency)])

    if save_results:
        cmd.append("--save-result")

    # Create result directory with timestamp and dataset name
    result_dir = get_result_dir(timestamp, dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    cmd.extend(["--result-dir", result_dir])

    # Print command for debugging
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print()

    # Run benchmark
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Read and print result files
        if save_results:
            print("\n" + "="*50)
            print(f"Results saved to: {result_dir}")
            print("="*50)
            for file in os.listdir(result_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(result_dir, file)
                    print(f"\n--- {file} ---")
                    with open(file_path, "r") as f:
                        print(f.read())

        return {
            "status": "success",
            "dataset": dataset_name,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "result_dir": result_dir
        }

    except subprocess.CalledProcessError as e:
        print(f"VisionArena benchmark failed with error code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {
            "status": "failed",
            "dataset": dataset_name,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }


async def _run_custom_images_benchmark_async(
    timestamp: str,
    num_prompts: int = None,
    prompt_text: str = "Describe this image in detail.",
    max_concurrency: int = 10,
    save_results: bool = True,
):
    """
    Async implementation of custom benchmark with local images.
    """
    import asyncio
    import aiohttp
    import time
    import numpy as np

    dataset_name = "custom_images"
    print(f"[Custom Images Benchmark] Starting...")
    print(f"Server: {BENCHMARK_CONFIG['server_url']}")
    print(f"Model: {BENCHMARK_CONFIG['served_model_name']}")
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Max concurrency: {max_concurrency}")

    # List all image files in the directory
    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    print(f"Found {len(image_files)} images")

    # Limit number of images if specified
    if num_prompts is not None:
        image_files = image_files[:num_prompts]
        print(f"Using {len(image_files)} images")

    # Prepare image data with base64 encoding
    requests_data = []
    for img_file in image_files:
        img_path = os.path.join(IMAGES_DIR, img_file)

        # Read and encode image as base64
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine image type
        if img_file.lower().endswith('.png'):
            img_type = "png"
        else:
            img_type = "jpeg"

        image_url = f"data:image/{img_type};base64,{img_data}"

        requests_data.append({
            "image_name": img_file,
            "image_url": image_url,
            "prompt": prompt_text
        })

    print(f"\nPrepared {len(requests_data)} requests")

    # Benchmark metrics storage
    results = []

    # Semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrency)

    async def make_request(session, request_data, request_id):
        """Make a single streaming request and collect metrics"""
        async with semaphore:
            request_payload = {
                "model": BENCHMARK_CONFIG["served_model_name"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": request_data["prompt"]},
                            {"type": "image_url", "image_url": {"url": request_data["image_url"]}}
                        ]
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.0,
                "stream": True
            }

            metrics = {
                "request_id": request_id,
                "image_name": request_data["image_name"],
                "success": False,
                "error": None,
                "start_time": None,
                "ttft": None,  # Time to first token
                "end_time": None,
                "latency": None,  # End-to-end latency
                "output_tokens": 0,
                "generated_text": "",
                "itl": [],  # Inter-token latencies
            }

            try:
                metrics["start_time"] = time.time()

                async with session.post(
                    f"{BENCHMARK_CONFIG['server_url']}/v1/chat/completions",
                    json=request_payload,
                    timeout=aiohttp.ClientTimeout(total=600)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        metrics["error"] = f"HTTP {response.status}: {error_text}"
                        return metrics

                    first_token_time = None
                    last_token_time = None

                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line or line == "data: [DONE]":
                            continue

                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])

                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")

                                    if content:
                                        current_time = time.time()

                                        # Record time to first token
                                        if first_token_time is None:
                                            first_token_time = current_time
                                            metrics["ttft"] = first_token_time - metrics["start_time"]
                                        else:
                                            # Record inter-token latency
                                            if last_token_time is not None:
                                                metrics["itl"].append(current_time - last_token_time)

                                        last_token_time = current_time
                                        metrics["output_tokens"] += 1
                                        metrics["generated_text"] += content

                            except json.JSONDecodeError:
                                continue

                metrics["end_time"] = time.time()
                metrics["latency"] = metrics["end_time"] - metrics["start_time"]
                metrics["success"] = True

            except Exception as e:
                metrics["error"] = str(e)
                metrics["end_time"] = time.time()
                if metrics["start_time"]:
                    metrics["latency"] = metrics["end_time"] - metrics["start_time"]

            return metrics

    # Create aiohttp session with connection pooling
    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        ttl_dns_cache=300,
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        print("\nStarting benchmark requests...")
        benchmark_start_time = time.time()

        # Create all tasks
        tasks = [
            make_request(session, req_data, i)
            for i, req_data in enumerate(requests_data)
        ]

        # Execute all requests
        results = await asyncio.gather(*tasks)

        benchmark_duration = time.time() - benchmark_start_time

    # Calculate metrics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'='*60}")
    print(f" Custom Images Benchmark Results ")
    print(f"{'='*60}")
    print(f"{'Successful requests:':<40} {len(successful):<10}")
    print(f"{'Failed requests:':<40} {len(failed):<10}")
    print(f"{'Maximum request concurrency:':<40} {max_concurrency:<10}")
    print(f"{'Benchmark duration (s):':<40} {benchmark_duration:<10.2f}")

    if successful:
        # Calculate TTFT metrics
        ttfts = [r["ttft"] for r in successful if r["ttft"] is not None]
        if ttfts:
            print(f"\n{'-'*60}")
            print(f" Time to First Token (TTFT)")
            print(f"{'-'*60}")
            print(f"{'Mean TTFT (ms):':<40} {np.mean(ttfts) * 1000:<10.2f}")
            print(f"{'Median TTFT (ms):':<40} {np.median(ttfts) * 1000:<10.2f}")
            print(f"{'P99 TTFT (ms):':<40} {np.percentile(ttfts, 99) * 1000:<10.2f}")

        # Calculate ITL metrics
        all_itls = []
        for r in successful:
            all_itls.extend(r["itl"])

        if all_itls:
            print(f"\n{'-'*60}")
            print(f" Inter-Token Latency (ITL)")
            print(f"{'-'*60}")
            print(f"{'Mean ITL (ms):':<40} {np.mean(all_itls) * 1000:<10.2f}")
            print(f"{'Median ITL (ms):':<40} {np.median(all_itls) * 1000:<10.2f}")
            print(f"{'P99 ITL (ms):':<40} {np.percentile(all_itls, 99) * 1000:<10.2f}")

        # Calculate E2EL metrics
        latencies = [r["latency"] for r in successful]
        print(f"\n{'-'*60}")
        print(f" End-to-End Latency (E2EL)")
        print(f"{'-'*60}")
        print(f"{'Mean E2EL (ms):':<40} {np.mean(latencies) * 1000:<10.2f}")
        print(f"{'Median E2EL (ms):':<40} {np.median(latencies) * 1000:<10.2f}")
        print(f"{'P99 E2EL (ms):':<40} {np.percentile(latencies, 99) * 1000:<10.2f}")

        # Calculate TPOT metrics
        tpots = []
        for r in successful:
            if r["output_tokens"] > 1 and r["ttft"] is not None:
                tpot = (r["latency"] - r["ttft"]) / (r["output_tokens"] - 1)
                tpots.append(tpot)

        if tpots:
            print(f"\n{'-'*60}")
            print(f" Time per Output Token (TPOT)")
            print(f"{'-'*60}")
            print(f"{'Mean TPOT (ms):':<40} {np.mean(tpots) * 1000:<10.2f}")
            print(f"{'Median TPOT (ms):':<40} {np.median(tpots) * 1000:<10.2f}")
            print(f"{'P99 TPOT (ms):':<40} {np.percentile(tpots, 99) * 1000:<10.2f}")

        # Throughput metrics
        total_output_tokens = sum(r["output_tokens"] for r in successful)
        print(f"\n{'-'*60}")
        print(f" Throughput Metrics")
        print(f"{'-'*60}")
        print(f"{'Request throughput (req/s):':<40} {len(successful) / benchmark_duration:<10.2f}")
        print(f"{'Output token throughput (tok/s):':<40} {total_output_tokens / benchmark_duration:<10.2f}")
        print(f"{'Total output tokens:':<40} {total_output_tokens:<10}")

    print(f"{'='*60}\n")

    # Save results if requested
    if save_results:
        result_dir = get_result_dir(timestamp, dataset_name)
        os.makedirs(result_dir, exist_ok=True)

        result_file = os.path.join(result_dir, "benchmark_results.json")

        result_data = {
            "timestamp": timestamp,
            "benchmark_duration": benchmark_duration,
            "server_url": BENCHMARK_CONFIG['server_url'],
            "model": BENCHMARK_CONFIG['served_model_name'],
            "num_images": len(requests_data),
            "max_concurrency": max_concurrency,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "results": results
        }

        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)

        print(f"Results saved to: {result_file}")

    return {
        "status": "success",
        "dataset": dataset_name,
        "num_images": len(requests_data),
        "completed": len(successful),
        "failed": len(failed),
        "duration": benchmark_duration,
        "result_dir": result_dir if save_results else None,
    }


@app.function(
    image=image,
    gpu="T4",
    max_containers=1,
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=100,
    volumes={
        MODELS_DIR: volume,
    },
    mounts=[local_images_mount],
)
def run_custom_images_benchmark(
    timestamp: str,
    num_prompts: int = None,
    prompt_text: str = "Describe this image in detail.",
    max_concurrency: int = 10,
    save_results: bool = True,
):
    """
    Run custom benchmark with local images by directly calling OpenAI chat/completions endpoint.

    This is a synchronous wrapper around the async implementation.

    Args:
        timestamp: Timestamp for organizing results
        num_prompts: Number of images to use (None = all images)
        prompt_text: Text prompt to use for all images
        max_concurrency: Maximum concurrent requests
        save_results: Whether to save results
    """
    import asyncio

    # Run the async function using asyncio.run()
    return asyncio.run(
        _run_custom_images_benchmark_async(
            timestamp=timestamp,
            num_prompts=num_prompts,
            prompt_text=prompt_text,
            max_concurrency=max_concurrency,
            save_results=save_results,
        )
    )


@app.function(
    image=image,
    gpu="T4",
    max_containers=1,
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=100,
    volumes={
        MODELS_DIR: volume,
    },
)
def run_structured_output_benchmark(
    timestamp: str,
    dataset_type: str = "json",  # json, grammar, regex, choice, xgrammar_bench
    num_prompts: int = 100,
    request_rate: int = 10,
    backend: str = "vllm",
    structured_output_ratio: float = None,  # Only for JSON
    structure_type: str = None,  # Only for grammar
    save_results: bool = True,
):
    """
    Run structured output benchmark for various generation formats.

    Supports:
    - JSON Schema: dataset_type="json", structured_output_ratio=1.0
    - Grammar-based: dataset_type="grammar", structure_type="grammar"
    - Regex-based: dataset_type="regex"
    - Choice-based: dataset_type="choice"
    - XGrammar: dataset_type="xgrammar_bench"

    Args:
        timestamp: Timestamp for organizing results
        dataset_type: Type of dataset (json, grammar, regex, choice, xgrammar_bench)
        num_prompts: Number of prompts to use
        request_rate: Request rate (req/s)
        backend: Backend type (vllm)
        structured_output_ratio: Ratio of structured outputs (for JSON)
        structure_type: Structure type (for grammar)
        save_results: Whether to save results
    """
    dataset_name = f"structured_{dataset_type}"
    print(f"[Structured Output - {dataset_type.upper()}] Starting with {num_prompts} prompts...")
    print(f"Server: {BENCHMARK_CONFIG['server_url']}")
    print(f"Model: {BENCHMARK_CONFIG['served_model_name']}")
    print(f"Request Rate: {request_rate} req/s")

    # Prepare model path
    model_path = f"{MODELS_DIR}/{BENCHMARK_CONFIG['model_name']}"

    # Use benchmark script from cloned vLLM repo
    benchmark_script = "/opt/vllm/benchmarks/benchmark_serving_structured_output.py"

    if not os.path.exists(benchmark_script):
        raise FileNotFoundError(
            f"vLLM benchmark script not found at {benchmark_script}. "
            "The vLLM repo should have been cloned during image build."
        )

    print(f"Using benchmark script: {benchmark_script}")

    # Build benchmark command
    cmd = [
        "python3", benchmark_script,
        "--backend", backend,
        "--model", BENCHMARK_CONFIG["served_model_name"],
        "--dataset", dataset_type,
        "--request-rate", str(request_rate),
        "--num-prompts", str(num_prompts),
        "--base-url", BENCHMARK_CONFIG["server_url"],
    ]

    # Add dataset-specific parameters
    if dataset_type == "json" and structured_output_ratio is not None:
        cmd.extend(["--structured-output-ratio", str(structured_output_ratio)])

    # if dataset_type == "grammar" and structure_type:
    #     cmd.extend(["--structure-type", structure_type])

    # Create result directory with timestamp and dataset name
    result_dir = get_result_dir(timestamp, dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    # Print command for debugging
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print()

    # Run benchmark
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Save output to file if requested
        if save_results:
            output_file = os.path.join(result_dir, f"results_{dataset_type}.txt")
            with open(output_file, "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\n=== STDERR ===\n")
                    f.write(result.stderr)

            print("\n" + "="*50)
            print(f"Results saved to: {result_dir}")
            print(f"Output file: {output_file}")
            print("="*50)

        return {
            "status": "success",
            "dataset": dataset_name,
            "dataset_type": dataset_type,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "result_dir": result_dir
        }

    except subprocess.CalledProcessError as e:
        print(f"Structured output benchmark failed with error code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {
            "status": "failed",
            "dataset": dataset_name,
            "dataset_type": dataset_type,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }


@app.local_entrypoint()
def main():
    """
    Run all configured benchmarks with shared configuration.
    Results are organized by timestamp and dataset name.

    Directory structure:
        /llama_models/benchmark_results/
            └── YYYYMMDD_HHMMSS/
                ├── sharegpt/
                │   └── results.json
                ├── visionarena/
                │   └── results.json
                ├── structured_json/
                │   └── results_json.txt
                ├── structured_grammar/
                │   └── results_grammar.txt
                └── ...
    """
    # Generate timestamp for this benchmark run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*60)
    print("vLLM Benchmark Suite")
    print("="*60)
    print(f"Timestamp: {timestamp}")
    print(f"Server: {BENCHMARK_CONFIG['server_url']}")
    print(f"Model: {BENCHMARK_CONFIG['served_model_name']}")
    print("="*60)

    # List of benchmarks to run
    benchmarks = [
        "sharegpt",
        # "visionarena",
        # "custom_images",
        "structured_json",
        "structured_grammar",
        "structured_regex",
        "structured_choice",
        "structured_xgrammar",
    ]
    results = {}

    for benchmark_type in benchmarks:
        print("\n" + "="*60)
        print(f"Starting {benchmark_type.upper()} benchmark...")
        print("="*60)

        try:
            if benchmark_type == "sharegpt":
                result = run_sharegpt_benchmark.remote(timestamp=timestamp)
            elif benchmark_type == "visionarena":
                result = run_visionarena_benchmark.remote(timestamp=timestamp)
            elif benchmark_type == "custom_images":
                result = run_custom_images_benchmark.remote(
                    timestamp=timestamp,
                    num_prompts=None,  # Use all images
                    prompt_text="Describe this image in detail.",
                    max_concurrency=10,
                    save_results=True,
                )
            elif benchmark_type == "structured_json":
                result = run_structured_output_benchmark.remote(
                    timestamp=timestamp,
                    dataset_type="json",
                    structured_output_ratio=1.0
                )
            elif benchmark_type == "structured_grammar":
                result = run_structured_output_benchmark.remote(
                    timestamp=timestamp,
                    dataset_type="grammar",
                    structure_type="grammar"
                )
            elif benchmark_type == "structured_regex":
                result = run_structured_output_benchmark.remote(
                    timestamp=timestamp,
                    dataset_type="regex"
                )
            elif benchmark_type == "structured_choice":
                result = run_structured_output_benchmark.remote(
                    timestamp=timestamp,
                    dataset_type="choice"
                )
            elif benchmark_type == "structured_xgrammar":
                result = run_structured_output_benchmark.remote(
                    timestamp=timestamp,
                    dataset_type="xgrammar_bench"
                )
            else:
                print(f"Unknown benchmark type: {benchmark_type}")
                continue

            results[benchmark_type] = result

            print("\n" + "-"*60)
            print(f"{benchmark_type.upper()} Benchmark Complete!")
            print(f"Status: {result['status']}")
            if result['status'] == 'success':
                print(f"Results: {result['result_dir']}")
            print("-"*60)

        except Exception as e:
            print(f"Error running {benchmark_type} benchmark: {e}")
            results[benchmark_type] = {
                "status": "error",
                "error": str(e)
            }

    # Print final summary
    print("\n" + "="*60)
    print("ALL BENCHMARKS COMPLETE!")
    print("="*60)
    print(f"Timestamp: {timestamp}")
    print(f"Results directory: {MODELS_DIR}/benchmark_results/{timestamp}/")
    print("\nSummary:")
    for btype, result in results.items():
        status = result.get('status', 'unknown')
        print(f"  {btype:15s}: {status}")
    print("="*60)

    return results