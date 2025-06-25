from pathlib import Path
from typing import Optional

import modal


GPU_CONFIG = "A100-80GB:3"  # for DeepSeek-R1, literal `None` for phi-4
app = modal.App("example-llama-cpp")





# You can trigger inference from the command line with

# ```bash
# modal run llama_cpp.py
# ```

# To try out Phi-4 instead, use the `--model` argument:

# ```bash
# modal run llama_cpp.py --model="phi-4"
# ```

# Note that this will run for up to 30 minutes, which costs ~$5.
# To allow it to proceed even if your local terminal fails,
# add the `--detach` flag after `modal run`.
# See below for details on getting the outputs.

# You can pass prompts with the `--prompt` argument and set the maximum number of tokens
# with the `--n-predict` argument.

# Additional arguments for `llama-cli` are passed as a string like `--args="--foo 1 --bar"`.

# For convenience, we set a number of sensible defaults for DeepSeek-R1,
# following the suggestions by the team at unsloth,
# who [quantized the model to 1.58 bit](https://unsloth.ai/blog/deepseekr1-dynamic).


DEFAULT_DEEPSEEK_R1_ARGS = [  # good default llama.cpp cli args for deepseek-r1
    "--cache-type-k",
    "q4_0",
    "--threads",
    "12",
    "-no-cnv",
    "--prio",
    "2",
    "--temp",
    "0.6",
    "--ctx-size",
    "8192",
]

DEFAULT_PHI_ARGS = [  # good default llama.cpp cli args for phi-4
    "--threads",
    "16",
    "-no-cnv",
    "--ctx-size",
    "16384",
]

DEFAULT_LLAMA_4_ARGS = [
    "--threads",
    "32",
    "-no-cnv",
    "--prio",
    "3",
    "--temp",
    "0.6",
    "--ctx-size",
    "16384",
]

LLAMA_CPP_RELEASE = "b5753"
MINUTES = 60

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands("git clone https://github.com/ggerganov/llama.cpp --branch " + LLAMA_CPP_RELEASE) # clone with release tag
    .run_commands(
        "cmake llama.cpp -B llama.cpp/build "
        "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON "
    )
    .run_commands(  # this one takes a few minutes!
        "cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli"
    )
    .run_commands("cp llama.cpp/build/bin/llama-* llama.cpp")
    .entrypoint([])  # remove NVIDIA base container entrypoint
)

model_cache = modal.Volume.from_name("llamacpp-cache", create_if_missing=True)
cache_dir = "/root/.cache/llama.cpp"

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=download_image, volumes={cache_dir: model_cache}, timeout=30 * MINUTES
)
def download_model(repo_id, allow_patterns, revision: Optional[str] = None):
    from huggingface_hub import snapshot_download

    print(f"🦙 downloading model from {repo_id} if not present")

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=cache_dir,
        allow_patterns=allow_patterns,
    )

    model_cache.commit()  # ensure other Modal Functions can see our writes before we quit

    print("🦙 model loaded")


results = modal.Volume.from_name("llamacpp-results", create_if_missing=True)
results_dir = "/root/results"


@app.function(
    image=image,
    volumes={cache_dir: model_cache, results_dir: results},
    gpu=GPU_CONFIG,
    timeout=30 * MINUTES,
)
def llama_cpp_inference(
    model_entrypoint_file: str,
    prompt: Optional[str] = None,
    n_predict: int = -1,
    args: Optional[list[str]] = None,
    store_output: bool = True,
):
    import subprocess
    from uuid import uuid4

    if prompt is None:
        prompt = DEFAULT_PROMPT  # see end of file
    if "llama-4" in model_entrypoint_file.lower():
        # format is <|header_start|>user<|header_end|>\n\nWhat is 1+1?<|eot|>
        prompt = "<|header_start|>user<|header_end|>\n\n" + prompt + "<|eot|><|header_start|>assistant<|header_end|>\n\n"
    if "deepseek" in model_entrypoint_file.lower():
        prompt = "<｜User｜>" + prompt + "<think>"
    if args is None:
        args = []

    # set layers to "off-load to", aka run on, GPU
    if GPU_CONFIG is not None:
        n_gpu_layers = 9999  # all
    else:
        n_gpu_layers = 0

    if store_output:
        result_id = str(uuid4())
        print(f"🦙 running inference with id:{result_id}")

    command = [
        "/llama.cpp/llama-cli",
        "--model",
        f"{cache_dir}/{model_entrypoint_file}",
        "--n-gpu-layers",
        str(n_gpu_layers),
        "--prompt",
        prompt,
        "--n-predict",
        str(n_predict),
    ] + args

    print("🦙 running commmand:", command, sep="\n\t")
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False
    )

    stdout, stderr = collect_output(p)

    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, command, stdout, stderr)

    if store_output:  # save results to a Modal Volume if requested
        print(f"🦙 saving results for {result_id}")
        result_dir = Path(results_dir) / result_id
        result_dir.mkdir(
            parents=True,
        )
        (result_dir / "out.txt").write_text(stdout)
        (result_dir / "err.txt").write_text(stderr)

    return stdout


DEFAULT_PROMPT = """Create a Flappy Bird game in Python. You must include these things:

    You must use pygame.
    The background color should be randomly chosen and is a light shade. Start with a light blue color.
    Pressing SPACE multiple times will accelerate the bird.
    The bird's shape should be randomly chosen as a square, circle or triangle. The color should be randomly chosen as a dark color.
    Place on the bottom some land colored as dark brown or yellow chosen randomly.
    Make a score shown on the top right side. Increment if you pass pipes and don't hit them.
    Make randomly spaced pipes with enough space. Color them randomly as dark green or light brown or a dark gray shade.
    When you lose, show the best score. Make the text inside the screen. Pressing q or Esc will quit the game. Restarting is pressing SPACE again.

The final game should be inside a markdown section in Python. Check your code for errors and fix them before the final markdown section."""


def stream_output(stream, queue, write_stream):
    """Reads lines from a stream and writes to a queue and a write stream."""
    for line in iter(stream.readline, b""):
        line = line.decode("utf-8", errors="replace")
        write_stream.write(line)
        write_stream.flush()
        queue.put(line)
    stream.close()


def collect_output(process):
    """Collect up the stdout and stderr of a process while still streaming it out."""
    import sys
    from queue import Queue
    from threading import Thread

    stdout_queue = Queue()
    stderr_queue = Queue()

    stdout_thread = Thread(
        target=stream_output, args=(process.stdout, stdout_queue, sys.stdout)
    )
    stderr_thread = Thread(
        target=stream_output, args=(process.stderr, stderr_queue, sys.stderr)
    )
    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()
    process.wait()

    stdout_collected = "".join(stdout_queue.queue)
    stderr_collected = "".join(stderr_queue.queue)

    return stdout_collected, stderr_collected


@app.local_entrypoint()
def main(
    prompt: Optional[str] = None,
    model: str = "DeepSeek-R1",  # or "phi-4"
    n_predict: int = -1,  # max number of tokens to predict, -1 is infinite
    args: Optional[str] = None,  # string of arguments to pass to llama.cpp's cli
):
    """Run llama.cpp inference on Modal for phi-4 or deepseek r1."""
    import shlex

    org_name = "unsloth"
    # two sample models: the diminuitive phi-4 and the chonky deepseek r1
    if model.lower() == "phi-4":
        model_name = "phi-4-GGUF"
        quant = "Q2_K"
        model_entrypoint_file = f"phi-4-{quant}.gguf"
        model_pattern = f"*{quant}*"
        revision = None
        parsed_args = DEFAULT_PHI_ARGS if args is None else shlex.split(args)
    elif model.lower() == "llama-4-maverick-17b-128e-instruct":
        model_name = "Llama-4-Maverick-17B-128E-Instruct-GGUF"
        quant = "UD-Q2_K_XL"
        model_entrypoint_file = (
            f"{quant}/Llama-4-Maverick-17B-128E-Instruct-{quant}-00001-of-00004.gguf"
        )
        model_pattern = f"*{quant}*"
        revision = None
        parsed_args = DEFAULT_LLAMA_4_ARGS if args is None else shlex.split(args)
    elif model.lower() == "deepseek-r1":
        model_name = "DeepSeek-R1-GGUF"
        quant = "UD-IQ1_S"
        model_entrypoint_file = (
            f"{model}-{quant}/DeepSeek-R1-{quant}-00001-of-00003.gguf"
        )
        model_pattern = f"*{quant}*"
        revision = "02656f62d2aa9da4d3f0cdb34c341d30dd87c3b6"
        parsed_args = DEFAULT_DEEPSEEK_R1_ARGS if args is None else shlex.split(args)
    else:
        raise ValueError(f"Unknown model {model}")

    repo_id = f"{org_name}/{model_name}"
    download_model.remote(repo_id, [model_pattern], revision)

    # call out to a `.remote` Function on Modal for inference
    for i in range(100):
        result = llama_cpp_inference.remote(
            model_entrypoint_file,
            prompt,
            n_predict,
            parsed_args,
            store_output=model.lower() == "deepseek-r1",
        )
        output_path = Path("/tmp") / f"llama-cpp-{model}.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"🦙 writing response to {output_path}")
        output_path.write_text(result)