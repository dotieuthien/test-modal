import modal
import subprocess
import os
from datetime import datetime


app = modal.App("vllm-benchmark-client")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Install git for cloning vLLM repo
    .pip_install(
        "vllm",
        "datasets",
        "huggingface-hub",
    )
    .run_commands(
        # Clone vLLM repo to get benchmark scripts
        "cd /opt && git clone --depth 1 https://github.com/vllm-project/vllm.git",
    )
)

MODELS_DIR = "/llama_models"
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

volume = modal.Volume.from_name("llama_models", create_if_missing=True)


# Global configuration - shared across all benchmarks
BENCHMARK_CONFIG = {
    "server_url": "https://dotieuthien--example-vllm-openai-compatible-serve.modal.run",
    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "served_model_name": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
}


def get_result_dir(timestamp: str, dataset_name: str) -> str:
    """Generate result directory path with timestamp and dataset name."""
    return f"{MODELS_DIR}/benchmark_results/{timestamp}/{dataset_name}"


@app.function(
    image=image,
    gpu="T4",
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=1000,
    volumes={
        MODELS_DIR: volume,
    },
)
def run_sharegpt_benchmark(
    timestamp: str,
    num_prompts: int = 10,
    backend: str = "vllm",
    endpoint: str = "/v1/completions",
    max_concurrency: int = None,
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
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=1000,
    volumes={
        MODELS_DIR: volume,
    },
)
def run_visionarena_benchmark(
    timestamp: str,
    num_prompts: int = 1000,
    backend: str = "openai-chat",
    endpoint: str = "/v1/chat/completions",
    hf_dataset_path: str = "lmarena-ai/VisionArena-Chat",
    hf_split: str = "train",
    hf_subset: str = None,
    max_concurrency: int = None,
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


@app.function(
    image=image,
    gpu="T4",
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=1000,
    volumes={
        MODELS_DIR: volume,
    },
)
def run_structured_output_benchmark(
    timestamp: str,
    dataset_type: str = "json",  # json, grammar, regex, choice, xgrammar_bench
    num_prompts: int = 1000,
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
        "visionarena",
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