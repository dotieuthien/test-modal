import logging
import random

import locust


image_url = "https://modal-public-assets.s3.amazonaws.com/golden-gate-bridge.jpg"
messages = [
    {
        "role": "system",
        "content": "You are a salesman for Modal, the cloud-native serverless Python computing platform.",
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image in detail."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    },
]


class WebsiteUser(locust.HttpUser):
    wait_time = locust.between(1, 5)
    headers = {
        "Authorization": "Bearer super-secret-token",
        "Accept": "application/json",
    }

    @locust.task
    def chat_completion(self):
        payload = {
            "model": "Qwen2-VL-7B-Instruct",
            "messages": messages,
        }

        response = self.client.request(
            "POST", "/v1/chat/completions", json=payload, headers=self.headers
        )
        response.raise_for_status()
        if random.random() < 0.01:
            logging.info(response.json()["choices"][0]["message"]["content"])