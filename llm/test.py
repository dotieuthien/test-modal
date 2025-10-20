import requests
import logging


def chat_completion(image_url): 
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail like an image generation prompt"
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

        payload = {
            "model": "Qwen2.5-VL-7B-Instruct-AWQ",
            "messages": messages,
            "for_service": "internal_cv_team"
        }

        headers = {
            "Authorization": "Bearer super-secret-token-fake",
            "Accept": "application/json",
        }

        response = requests.post(
            "https://styleme--example-vllm-openai-compatible-serve.modal.run/generate", json=payload, headers=headers
        )
        response.raise_for_status()
        prompt = response.json()["choices"][0]["message"]["content"]
        
        return prompt
    
        
if __name__ == "__main__":
    chat_completion("https://gpbuympmmntlwaopnaov.supabase.co/storage/v1/object/public/galleries/test/source_outpainting_img.png")