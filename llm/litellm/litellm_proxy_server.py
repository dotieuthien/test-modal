import modal

image = modal.Image.debian_slim().pip_install("litellm[proxy]")
app = modal.App(name="example-litellm-proxy", image=image)


N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=image,
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    # mount config.yaml
    mounts=[modal.Mount.from_local_file("config.yaml", remote_path="/root/config.yaml")]
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = ["litellm", "--config", "config.yaml", "--host", "0.0.0.0", "--port", str(VLLM_PORT), "--detailed_debug"]
    subprocess.Popen(" ".join(cmd), shell=True)