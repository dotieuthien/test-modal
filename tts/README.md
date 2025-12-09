# F5-TTS with TensorRT-LLM on Modal

Deploy F5-TTS model using TensorRT-LLM and Triton Inference Server on Modal.

## Prerequisites

- Modal CLI: `pip install modal`
- Modal auth: `modal setup`
- L4 GPU quota

## Quick Start

Run the complete pipeline:

```bash
modal run trtllm_f5_tts.py
```

This will:
1. Download F5-TTS model from HuggingFace
2. Build TensorRT-LLM engine
3. Export Vocos vocoder to TensorRT
4. Start Triton server and run test inference

Output audio saved to `/output_audio/test_output.wav`

## Deploy Modal app

Deploy the Modal service:

```bash
modal deploy trtllm_f5_tts.py
```

This will deploy the Modal app and make it available as a web service.

## Call Modal service

Once deployed, test the service with:

```bash
python test_client_http.py
```

This will send a TTS inference request to the deployed Modal endpoint and save the generated audio to `output.wav`.

## Model Details

- **Model**: [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS)
- **GPU**: L4
- **Framework**: TensorRT-LLM 0.16.0
- **Server**: NVIDIA Triton Server 24.12