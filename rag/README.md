# Vision RAG

This directory contains the Vision RAG implementation.

## Setup Instructions

1. Set up Hugging Face Access Token in Modal secrets
![HF](images/hf_secret.png)
2. Set up OpenAI API Key in Modal secrets
![OpenAI](images/openai_secret.png)
3. Deploy gradio app
   ```bash
   modal deploy main.py
   ```

## Example Result
![Vision RAG result](images/example.png)