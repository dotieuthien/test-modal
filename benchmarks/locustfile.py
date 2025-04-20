import logging
import random
import uuid
import locust
import glob
import base64
from PIL import Image
import io


image_list = glob.glob("/Users/dotieuthien/Documents/rnd/test-modal/benchmarks/images/test/*.png")


class WebsiteUser(locust.HttpUser):
    wait_time = locust.between(1, 5)
    headers = {
        "Authorization": "Bearer super-secret-token",
        "Accept": "application/json",
    }

    @locust.task
    def chat_completion(self):
        request_id = str(uuid.uuid4())
        image_path = random.choice(image_list)

        image = Image.open(image_path).convert("RGB")
        # convert image to base64
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        base64_image = base64.b64encode(image_bytes.getvalue()).decode(
            "utf-8")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """Return the results in JSON format for:
                        {
                            "sender_name": "The sender name, not number, empty if not present", 
                            "receiver_name": "The receiver name or business name, not number, empty if not present", 
                            "sender_bank_account": "The sender's bank account numbers, empty if not present", 
                            "receiver_bank_account": "The receiver's bank account numbers, empty if not present", 
                            "bank_name": "The name of the bank or service provider, empty if not present",
                            "transaction_id" :"The transaction ID, empty if not present",
                            "value": "The transaction amount", 
                            "type": "The transaction type: expense when send money to others or income when receive money from others, always return income or expense", 
                            "category": "The transaction purpose, categorized as bills, entertainment, education, shopping, others if cannot be categorized, always return value", 
                            "time": "The transaction timestamp", 
                            "noted": "Notes or messages in the transaction"
                        } 
                        """,
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ],
            },
        ]

        payload = {
            "model": "Qwen2.5-VL-7B-Instruct-AWQ",
            "messages": messages,
        }

        response = self.client.request(
            "POST", "/v1/chat/completions", json=payload, headers=self.headers
        )
        response.raise_for_status()
        logging.info(response.json()["choices"][0]["message"]["content"])