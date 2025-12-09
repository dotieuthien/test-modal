
from pathlib import Path

import modal


tensorrt_image = modal.Image.from_registry(
    "nvcr.io/nvidia/tritonserver:24.12-py3",
    add_python="3.10", # modal python, not use
).entrypoint([])

# Install pip package for triton own python3 that is used for python backend
tensorrt_image = tensorrt_image.run_commands(
    "/usr/bin/python3 -m pip install cuda-python==12.9.1 tritonclient[grpc] tensorrt-llm==0.16.0 torchaudio==2.5.1 jieba pypinyin librosa vocos hf-transfer==0.1.9 huggingface_hub==0.28.1",
)

tensorrt_image = tensorrt_image.pip_install(
    "cuda-python==12.9.1",
    "tritonclient[grpc]",
    "tensorrt-llm==0.16.0",
    "torchaudio==2.5.1",
    "jieba",
    "pypinyin",
    "librosa",
    "vocos",
    "hf-transfer==0.1.9",
    "huggingface_hub==0.28.1",
)

volume = modal.Volume.from_name(
    "example-trtllm-inference-volume", create_if_missing=True
)

VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"
F5_TTS_PATH = MODELS_PATH / "F5-TTS"
F5_TTS_CHECKPOINT_PATH = MODELS_PATH / "trtllm_ckpt"
F5_TTS_ENGINE_PATH = MODELS_PATH / "f5_trt_llm_engine"
VOCODER_ENGINE_PATH = MODELS_PATH / "vocos_vocoder.plan"
OUTPUT_AUDIO_PATH = VOLUME_PATH / "output_audio"

MODEL_ID = "SWivid/F5-TTS"

tensorrt_image = tensorrt_image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": str(MODELS_PATH),
    }
)

scripts_dir = Path(__file__).parent / "scripts"
scripts_remote_dir = Path("/root/scripts")
scripts_mount = modal.Mount.from_local_dir(
    scripts_dir,
    remote_path=scripts_remote_dir,
)

patch_dir = Path(__file__).parent / "patch"
patch_remote_dir = Path("/root/patch")
patch_mount = modal.Mount.from_local_dir(
    patch_dir,
    remote_path=patch_remote_dir,
)

model_repo_dir = Path(__file__).parent / "model_repo_f5_tts"
model_repo_remote_dir = Path("/root/model_repo_f5_tts")
model_repo_mount = modal.Mount.from_local_dir(
    model_repo_dir,
    remote_path=model_repo_remote_dir,
)

client_http_path = Path(__file__).parent / "client_http.py"
client_http_remote_path = Path("/root/client_http.py")

client_http_mount = modal.Mount.from_local_file(
    client_http_path,
    remote_path=client_http_remote_path,
)

client_http_path = Path(__file__).parent / "client_http.py"
client_http_remote_path = Path("/root/client_http.py")

client_http_mount = modal.Mount.from_local_file(
    client_http_path,
    remote_path=client_http_remote_path,
)

ref_audio_path = Path(__file__).parent / "basic_ref_en.wav"
ref_audio_remote_path = Path("/root/basic_ref_en.wav")

ref_audio_mount = modal.Mount.from_local_file(
    ref_audio_path,
    remote_path=ref_audio_remote_path,
)

N_GPUS = 1
GPU_CONFIG = f"L4:{N_GPUS}"
MINUTES = 60

app = modal.App(
    "example-trtllm-f5",
    mounts=[
        scripts_mount,
        patch_mount,
        model_repo_mount,
        client_http_mount,
        ref_audio_mount
    ]
)


@app.function(
    image=tensorrt_image,
    volumes={VOLUME_PATH: volume},
    timeout=60 * MINUTES,
)
def download_f5_tts(model_name: str = "F5TTS_v1_Base"):
    from huggingface_hub import snapshot_download

    print(f"Downloading F5-TTS model: {model_name}")
    snapshot_download(
        MODEL_ID,
        local_dir=F5_TTS_PATH,
    )
    volume.commit()
    print(f"Model downloaded to {F5_TTS_PATH}")


@app.function(
    image=tensorrt_image,
    gpu=GPU_CONFIG,
    volumes={VOLUME_PATH: volume},
    timeout=60 * MINUTES,
)
def build_trtllm_engine(model_name: str = "F5TTS_v1_Base"):
    """Stage 1: Convert checkpoint and build TensorRT-LLM engine"""    
    import subprocess
    
    

    model_checkpoint = F5_TTS_PATH / model_name / "model_1250000.safetensors"

    # Step 1: Convert checkpoint
    print("Converting checkpoint")
    result = subprocess.run([
        "python3", str(scripts_remote_dir / "convert_checkpoint.py"),
        "--timm_ckpt", str(model_checkpoint),
        "--output_dir", str(F5_TTS_CHECKPOINT_PATH),
        "--model_name", model_name
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)

    print("✓ Checkpoint converted successfully")
    if result.stdout:
        print("STDOUT:", result.stdout)

    # Step 2: Copy patch files to tensorrt_llm/models
    import shutil

    # Find tensorrt_llm package location
    import tensorrt_llm
    trtllm_path = Path(tensorrt_llm.__file__).parent
    target_dir = trtllm_path / "models"

    print(f"TensorRT-LLM path: {trtllm_path}")
    print(f"Target models directory: {target_dir}")

    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    patch_dir = Path("/root/patch")

    # Check if there are any files in patch directory
    patch_files = list(patch_dir.glob('*'))
    if patch_files:
        print(f"Copying {len(patch_files)} patch file(s) to tensorrt_llm/models")

        for patch_file in patch_files:
            target_path = target_dir / patch_file.name
            if patch_file.is_file():
                shutil.copy2(patch_file, target_path)
                print(f"  Copied: {patch_file.name}")
            elif patch_file.is_dir():
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(patch_file, target_path)
                print(f"  Copied directory: {patch_file.name}")

        print(f"✓ Patch files copied successfully")
    else:
        print(f"⚠ No patch files found in {patch_dir}")

    # Step 3: Build TensorRT-LLM engine
    print("Building TensorRT-LLM engine")
    subprocess.run([
        "trtllm-build",
        "--checkpoint_dir", str(F5_TTS_CHECKPOINT_PATH),
        "--max_batch_size", "8",
        "--output_dir", str(F5_TTS_ENGINE_PATH),
        "--remove_input_padding", "disable"
    ], check=True)

    volume.commit()
    print(f"Engine built at {F5_TTS_ENGINE_PATH}")

@app.function(
    image=tensorrt_image,
    gpu=GPU_CONFIG,
    volumes={VOLUME_PATH: volume},
    timeout=30 * MINUTES,
)
def export_vocoder():
    """Stage 2: Export vocos vocoder to TensorRT"""
    import subprocess

    onnx_path = MODELS_PATH / "vocos_vocoder.onnx"

    # Step 1: Export vocoder to ONNX
    print("Exporting vocos vocoder to ONNX")
    subprocess.run([
        "python3", str(scripts_remote_dir / "export_vocoder_to_onnx.py"),
        "--vocoder", "vocos",
        "--output-path", str(onnx_path)
    ], check=True)

    # Step 2: Convert ONNX to TensorRT
    print("Converting vocoder to TensorRT")
    subprocess.run([
        "bash", str(scripts_remote_dir / "export_vocos_trt.sh"),
        str(onnx_path), str(VOCODER_ENGINE_PATH)
    ], check=True)

    volume.commit()
    print(f"Vocoder engine saved at {VOCODER_ENGINE_PATH}")


@app.function(
    image=tensorrt_image,
    gpu=GPU_CONFIG,
    volumes={VOLUME_PATH: volume},
    container_idle_timeout=20 * MINUTES,
)
def start_and_test_triton_server(model_name: str = "F5TTS_v1_Base"):
    """Stage 3: Start Triton Inference Server"""
    import subprocess
    import shutil
    import os
    import time
    import requests

    # Set Python path to use /usr/bin/python3 that is triton python path for python backend
    os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
    os.environ["PYTHON_EXECUTABLE"] = "/usr/bin/python3"

    print(f"Python path set to: {os.environ['PATH']}")
    print(f"Python executable: /usr/bin/python3")

    model_repo_source = Path("/root/model_repo_f5_tts")
    model_repo_dest = MODELS_PATH / "model_repo"

    print("\nBuilding triton server model repository")

    # Remove existing model repo if it exists
    if model_repo_dest.exists():
        print(f"Removing existing model repo: {model_repo_dest}")
        shutil.rmtree(model_repo_dest)

    # Copy model_repo_f5_tts to volume
    print(f"Copying model repo from {model_repo_source} to {model_repo_dest}")
    shutil.copytree(model_repo_source, model_repo_dest)

    # Fill template with actual paths
    config_file = model_repo_dest / "f5_tts" / "config.pbtxt"
    vocab_path = F5_TTS_PATH / model_name / "vocab.txt"
    model_checkpoint = F5_TTS_PATH / model_name / "model_1250000.safetensors"

    print(f"Filling template in {config_file}")
    subprocess.run([
        "python3", str(scripts_remote_dir / "fill_template.py"),
        "-i", str(config_file),
        f"vocab:{vocab_path},model:{model_checkpoint},trtllm:{F5_TTS_ENGINE_PATH},vocoder:vocos"
    ], check=True)

    # Copy vocoder engine
    vocoder_dest = model_repo_dest / "vocoder" / "1" / "vocoder.plan"
    vocoder_dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying vocoder engine from {VOCODER_ENGINE_PATH} to {vocoder_dest}")
    shutil.copy2(VOCODER_ENGINE_PATH, vocoder_dest)

    print(f"✓ Model repository prepared at {model_repo_dest}")

    # Commit changes to volume
    volume.commit()

    # Start tritonserver in background
    triton_process = subprocess.Popen([
        "tritonserver",
        f"--model-repository={model_repo_dest}"
    ])

    # Wait for server to be ready with health check
    print("Waiting for Triton server to start...")
    max_retries = 60
    retry_count = 0
    server_ready = False

    while retry_count < max_retries:
        try:
            response = requests.get("http://localhost:8000/v2/health/ready")
            if response.status_code == 200:
                server_ready = True
                print("✓ Triton server is ready")
                break
        except:
            pass
        time.sleep(2)
        retry_count += 1
        print(f"  Checking server health... ({retry_count}/{max_retries})")

    if not server_ready:
        triton_process.kill()
        raise RuntimeError("Triton server failed to start within timeout")

    # Test triton server with HTTP client
    print("\nTesting Triton server with HTTP client")

    # Create output audio directory
    OUTPUT_AUDIO_PATH.mkdir(parents=True, exist_ok=True)
    output_audio_file = OUTPUT_AUDIO_PATH / "test_output.wav"

    reference_text = "Some call me nature, others call me mother nature."
    target_text = "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring."
    reference_audio = "/root/basic_ref_en.wav"

    subprocess.run([
        "python3", "/root/client_http.py",
        "--reference-audio", reference_audio,
        "--reference-text", reference_text,
        "--target-text", target_text,
        "--output-audio", str(output_audio_file)
    ], check=True)

    print(f"✓ HTTP client test completed. Output saved to {output_audio_file}")

    # Commit volume to save output audio
    volume.commit()

    return


@app.function(
    image=tensorrt_image,
    gpu=GPU_CONFIG,
    volumes={VOLUME_PATH: volume},
    container_idle_timeout=20 * MINUTES,
)
@modal.concurrent(
    max_inputs=10
)
@modal.web_server(port=8000, startup_timeout=10 * MINUTES)
def serve(model_name: str = "F5TTS_v1_Base"):
    """Stage 3: Start Triton Inference Server"""
    import subprocess
    import shutil
    import os
    import time
    import requests

    # Set Python path to use /usr/bin/python3 that is triton python path for python backend
    os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
    os.environ["PYTHON_EXECUTABLE"] = "/usr/bin/python3"

    print(f"Python path set to: {os.environ['PATH']}")
    print(f"Python executable: /usr/bin/python3")

    model_repo_source = Path("/root/model_repo_f5_tts")
    model_repo_dest = MODELS_PATH / "model_repo"

    print("\nBuilding triton server model repository")

    # Remove existing model repo if it exists
    if model_repo_dest.exists():
        print(f"Removing existing model repo: {model_repo_dest}")
        shutil.rmtree(model_repo_dest)

    # Copy model_repo_f5_tts to volume
    print(f"Copying model repo from {model_repo_source} to {model_repo_dest}")
    shutil.copytree(model_repo_source, model_repo_dest)

    # Fill template with actual paths
    config_file = model_repo_dest / "f5_tts" / "config.pbtxt"
    vocab_path = F5_TTS_PATH / model_name / "vocab.txt"
    model_checkpoint = F5_TTS_PATH / model_name / "model_1250000.safetensors"

    print(f"Filling template in {config_file}")
    subprocess.run([
        "python3", str(scripts_remote_dir / "fill_template.py"),
        "-i", str(config_file),
        f"vocab:{vocab_path},model:{model_checkpoint},trtllm:{F5_TTS_ENGINE_PATH},vocoder:vocos"
    ], check=True)

    # Copy vocoder engine
    vocoder_dest = model_repo_dest / "vocoder" / "1" / "vocoder.plan"
    vocoder_dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying vocoder engine from {VOCODER_ENGINE_PATH} to {vocoder_dest}")
    shutil.copy2(VOCODER_ENGINE_PATH, vocoder_dest)

    print(f"✓ Model repository prepared at {model_repo_dest}")

    # Commit changes to volume
    volume.commit()

    print(f"Starting Triton server with model repository: {model_repo_dest}")
    cmd = [
        "tritonserver",
        f"--model-repository={model_repo_dest}"
    ]
    
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
def main():
    # Stage 0: Download model
    print("Stage 0: Downloading F5-TTS model")
    download_f5_tts.remote()
    print("✓ Model downloaded")

    # Stage 1: Build TensorRT-LLM engine
    print("\nStage 1: Building TensorRT-LLM engine")
    engine_path = build_trtllm_engine.remote()
    print(f"✓ Engine built: {engine_path}")

    # Stage 2: Export vocoder
    print("\nStage 2: Exporting vocoder")
    vocoder_path = export_vocoder.remote()
    print(f"✓ Vocoder exported: {vocoder_path}")

    print("\n✓ F5-TTS preparation complete!")

    # Stage 3: Start Triton server
    print("\nStage 3: Starting Triton Inference Server")
    start_and_test_triton_server.remote()
    
    
    
    