import argparse

import numpy as np
import requests
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--server-url",
        type=str,
        default="https://dotieuthien--example-trtllm-f5-serve.modal.run",
        help="Address of the server",
    )

    parser.add_argument(
        "--reference-audio",
        type=str,
        default="basic_ref_en.wav",
        help="Path to a single audio file. It can't be specified at the same time with --manifest-dir",
    )

    parser.add_argument(
        "--reference-text",
        type=str,
        default="Some call me nature, others call me mother nature.",
        help="",
    )

    parser.add_argument(
        "--target-text",
        type=str,
        default="I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring.",
        help="",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="f5_tts",
        choices=["f5_tts", "spark_tts"],
        help="triton model_repo module name to request",
    )

    parser.add_argument(
        "--output-audio",
        type=str,
        default="output.wav",
        help="Path to save the output audio",
    )
    return parser.parse_args()


def prepare_request(
    samples,
    reference_text,
    target_text,
    sample_rate=24000,
    audio_save_dir: str = "./",
):
    assert len(samples.shape) == 1, "samples should be 1D"
    lengths = np.array([[len(samples)]], dtype=np.int32)
    samples = samples.reshape(1, -1).astype(np.float32)

    data = {
        "inputs": [
            {"name": "reference_wav", "shape": samples.shape, "datatype": "FP32", "data": samples.tolist()},
            {
                "name": "reference_wav_len",
                "shape": lengths.shape,
                "datatype": "INT32",
                "data": lengths.tolist(),
            },
            {"name": "reference_text", "shape": [1, 1], "datatype": "BYTES", "data": [reference_text]},
            {"name": "target_text", "shape": [1, 1], "datatype": "BYTES", "data": [target_text]},
        ]
    }

    return data


def load_audio(wav_path, target_sample_rate=24000):
    assert target_sample_rate == 24000, "hard coding in server"
    if isinstance(wav_path, dict):
        samples = wav_path["array"]
        sample_rate = wav_path["sampling_rate"]
    else:
        samples, sample_rate = sf.read(wav_path)
    if sample_rate != target_sample_rate:
        from scipy.signal import resample

        num_samples = int(len(samples) * (target_sample_rate / sample_rate))
        samples = resample(samples, num_samples)
    return samples, target_sample_rate


def check_health(server_url, max_retries=30, retry_interval=2):
    """Check if the server is healthy and ready, with retry logic."""
    import time

    health_url = f"{server_url}/v2/health/ready"
    print(f"Checking server health at {health_url}...")

    for attempt in range(1, max_retries + 1):
        try:
            rsp = requests.get(health_url, verify=False, timeout=10)
            if rsp.status_code == 200:
                print(f"✓ Server is healthy and ready")
                return True
            else:
                print(f"Attempt {attempt}/{max_retries}: Server returned status code {rsp.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt}/{max_retries}: Failed to connect - {e}")

        if attempt < max_retries:
            time.sleep(retry_interval)

    print(f"✗ Server health check failed after {max_retries} attempts")
    return False


if __name__ == "__main__":
    args = get_args()
    server_url = args.server_url
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"

    # Check server health first
    if not check_health(server_url):
        print("Exiting due to health check failure")
        exit(1)

    url = f"{server_url}/v2/models/{args.model_name}/infer"
    samples, sr = load_audio(args.reference_audio)
    assert sr == 24000, "sample rate hardcoded in server"

    samples = np.array(samples, dtype=np.float32)
    data = prepare_request(samples, args.reference_text, args.target_text)

    print(f"Sending inference request to {url}...")
    rsp = requests.post(
        url, headers={"Content-Type": "application/json"}, json=data, verify=False, params={"request_id": "0"}
    )
    result = rsp.json()
    audio = result["outputs"][0]["data"]
    audio = np.array(audio, dtype=np.float32)
    sf.write(args.output_audio, audio, 24000, "PCM_16")
    print(f"✓ Audio saved to {args.output_audio}")
